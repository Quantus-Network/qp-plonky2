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

theorem publicBatchWrapper86_f8397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 880 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6843)) := by
  have c54735 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54736 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54737 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54738 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54739 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54740 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_880_10 := arithEq_of_rows h (row := 880) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_10
  simp only [← c54735, ← c54736, ← c54737] at e_880_10
  have e_880_11 := arithEq_of_rows h (row := 880) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_11
  simp only [← c54738, ← c54739, k2, ← c54740] at e_880_11
  have hr := e_880_11
  simp only [e_880_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 880 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6844)) := by
  have c54741 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54742 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54743 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54744 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54745 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54746 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_880_12 := arithEq_of_rows h (row := 880) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_12
  simp only [← c54741, ← c54742, ← c54743] at e_880_12
  have e_880_13 := arithEq_of_rows h (row := 880) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_13
  simp only [← c54744, ← c54745, k2, ← c54746] at e_880_13
  have hr := e_880_13
  simp only [e_880_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 880 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6845)) := by
  have c54747 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54748 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54749 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54750 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54751 := (publicBatchWrapper86_copies1710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54752 := (publicBatchWrapper86_copies1711 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_880_14 := arithEq_of_rows h (row := 880) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_14
  simp only [← c54747, ← c54748, ← c54749] at e_880_14
  have e_880_15 := arithEq_of_rows h (row := 880) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_15
  simp only [← c54750, ← c54751, k2, ← c54752] at e_880_15
  have hr := e_880_15
  simp only [e_880_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 880 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6846)) := by
  have c54753 := (publicBatchWrapper86_copies1711 a h).2.1
  have c54754 := (publicBatchWrapper86_copies1711 a h).2.2.1
  have c54755 := (publicBatchWrapper86_copies1711 a h).2.2.2.1
  have c54756 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.1
  have c54757 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.1
  have c54758 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_880_16 := arithEq_of_rows h (row := 880) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_16
  simp only [← c54753, ← c54754, ← c54755] at e_880_16
  have e_880_17 := arithEq_of_rows h (row := 880) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_17
  simp only [← c54756, ← c54757, k2, ← c54758] at e_880_17
  have hr := e_880_17
  simp only [e_880_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 880 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6847)) := by
  have c54759 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.1
  have c54760 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.1
  have c54761 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.1
  have c54762 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54763 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54764 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_880_18 := arithEq_of_rows h (row := 880) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_18
  simp only [← c54759, ← c54760, ← c54761] at e_880_18
  have e_880_19 := arithEq_of_rows h (row := 880) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_880_19
  simp only [← c54762, ← c54763, k2, ← c54764] at e_880_19
  have hr := e_880_19
  simp only [e_880_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6848)) := by
  have c54765 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54766 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54767 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54768 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54769 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54770 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_0 := arithEq_of_rows h (row := 881) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_0
  simp only [← c54765, ← c54766, ← c54767] at e_881_0
  have e_881_1 := arithEq_of_rows h (row := 881) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_1
  simp only [← c54768, ← c54769, k2, ← c54770] at e_881_1
  have hr := e_881_1
  simp only [e_881_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6849)) := by
  have c54771 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54772 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54773 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54774 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54775 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54776 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_2 := arithEq_of_rows h (row := 881) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_2
  simp only [← c54771, ← c54772, ← c54773] at e_881_2
  have e_881_3 := arithEq_of_rows h (row := 881) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_3
  simp only [← c54774, ← c54775, k2, ← c54776] at e_881_3
  have hr := e_881_3
  simp only [e_881_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6850)) := by
  have c54777 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54778 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54779 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54780 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54781 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54782 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_4 := arithEq_of_rows h (row := 881) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_4
  simp only [← c54777, ← c54778, ← c54779] at e_881_4
  have e_881_5 := arithEq_of_rows h (row := 881) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_5
  simp only [← c54780, ← c54781, k2, ← c54782] at e_881_5
  have hr := e_881_5
  simp only [e_881_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6851)) := by
  have c54783 := (publicBatchWrapper86_copies1711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54784 := (publicBatchWrapper86_copies1712 a h).1
  have c54785 := (publicBatchWrapper86_copies1712 a h).2.1
  have c54786 := (publicBatchWrapper86_copies1712 a h).2.2.1
  have c54787 := (publicBatchWrapper86_copies1712 a h).2.2.2.1
  have c54788 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_6 := arithEq_of_rows h (row := 881) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_6
  simp only [← c54783, ← c54784, ← c54785] at e_881_6
  have e_881_7 := arithEq_of_rows h (row := 881) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_7
  simp only [← c54786, ← c54787, k2, ← c54788] at e_881_7
  have hr := e_881_7
  simp only [e_881_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6852)) := by
  have c54789 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.1
  have c54790 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.1
  have c54791 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.1
  have c54792 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.1
  have c54793 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.1
  have c54794 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_8 := arithEq_of_rows h (row := 881) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_8
  simp only [← c54789, ← c54790, ← c54791] at e_881_8
  have e_881_9 := arithEq_of_rows h (row := 881) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_9
  simp only [← c54792, ← c54793, k2, ← c54794] at e_881_9
  have hr := e_881_9
  simp only [e_881_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6853)) := by
  have c54795 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54796 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54797 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54798 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54799 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54800 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_10 := arithEq_of_rows h (row := 881) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_10
  simp only [← c54795, ← c54796, ← c54797] at e_881_10
  have e_881_11 := arithEq_of_rows h (row := 881) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_11
  simp only [← c54798, ← c54799, k2, ← c54800] at e_881_11
  have hr := e_881_11
  simp only [e_881_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6854)) := by
  have c54801 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54802 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54803 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54804 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54805 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54806 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_12 := arithEq_of_rows h (row := 881) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_12
  simp only [← c54801, ← c54802, ← c54803] at e_881_12
  have e_881_13 := arithEq_of_rows h (row := 881) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_13
  simp only [← c54804, ← c54805, k2, ← c54806] at e_881_13
  have hr := e_881_13
  simp only [e_881_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6855)) := by
  have c54807 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54808 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54809 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54810 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54811 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54812 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_14 := arithEq_of_rows h (row := 881) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_14
  simp only [← c54807, ← c54808, ← c54809] at e_881_14
  have e_881_15 := arithEq_of_rows h (row := 881) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_15
  simp only [← c54810, ← c54811, k2, ← c54812] at e_881_15
  have hr := e_881_15
  simp only [e_881_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6856)) := by
  have c54813 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54814 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54815 := (publicBatchWrapper86_copies1712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54816 := (publicBatchWrapper86_copies1713 a h).1
  have c54817 := (publicBatchWrapper86_copies1713 a h).2.1
  have c54818 := (publicBatchWrapper86_copies1713 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_16 := arithEq_of_rows h (row := 881) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_16
  simp only [← c54813, ← c54814, ← c54815] at e_881_16
  have e_881_17 := arithEq_of_rows h (row := 881) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_17
  simp only [← c54816, ← c54817, k2, ← c54818] at e_881_17
  have hr := e_881_17
  simp only [e_881_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 881 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6857)) := by
  have c54819 := (publicBatchWrapper86_copies1713 a h).2.2.2.1
  have c54820 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.1
  have c54821 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.1
  have c54822 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.1
  have c54823 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.1
  have c54824 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_881_18 := arithEq_of_rows h (row := 881) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_18
  simp only [← c54819, ← c54820, ← c54821] at e_881_18
  have e_881_19 := arithEq_of_rows h (row := 881) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_881_19
  simp only [← c54822, ← c54823, k2, ← c54824] at e_881_19
  have hr := e_881_19
  simp only [e_881_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6858)) := by
  have c54825 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.1
  have c54826 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54827 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54828 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54829 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54830 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_0 := arithEq_of_rows h (row := 882) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_0
  simp only [← c54825, ← c54826, ← c54827] at e_882_0
  have e_882_1 := arithEq_of_rows h (row := 882) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_1
  simp only [← c54828, ← c54829, k2, ← c54830] at e_882_1
  have hr := e_882_1
  simp only [e_882_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6859)) := by
  have c54831 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54832 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54833 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54834 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54835 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54836 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_2 := arithEq_of_rows h (row := 882) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_2
  simp only [← c54831, ← c54832, ← c54833] at e_882_2
  have e_882_3 := arithEq_of_rows h (row := 882) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_3
  simp only [← c54834, ← c54835, k2, ← c54836] at e_882_3
  have hr := e_882_3
  simp only [e_882_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6860)) := by
  have c54837 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54838 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54839 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54840 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54841 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54842 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_4 := arithEq_of_rows h (row := 882) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_4
  simp only [← c54837, ← c54838, ← c54839] at e_882_4
  have e_882_5 := arithEq_of_rows h (row := 882) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_5
  simp only [← c54840, ← c54841, k2, ← c54842] at e_882_5
  have hr := e_882_5
  simp only [e_882_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6861)) := by
  have c54843 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54844 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54845 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54846 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54847 := (publicBatchWrapper86_copies1713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54848 := (publicBatchWrapper86_copies1714 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_6 := arithEq_of_rows h (row := 882) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_6
  simp only [← c54843, ← c54844, ← c54845] at e_882_6
  have e_882_7 := arithEq_of_rows h (row := 882) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_7
  simp only [← c54846, ← c54847, k2, ← c54848] at e_882_7
  have hr := e_882_7
  simp only [e_882_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6862)) := by
  have c54849 := (publicBatchWrapper86_copies1714 a h).2.1
  have c54850 := (publicBatchWrapper86_copies1714 a h).2.2.1
  have c54851 := (publicBatchWrapper86_copies1714 a h).2.2.2.1
  have c54852 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.1
  have c54853 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.1
  have c54854 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_8 := arithEq_of_rows h (row := 882) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_8
  simp only [← c54849, ← c54850, ← c54851] at e_882_8
  have e_882_9 := arithEq_of_rows h (row := 882) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_9
  simp only [← c54852, ← c54853, k2, ← c54854] at e_882_9
  have hr := e_882_9
  simp only [e_882_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6863)) := by
  have c54855 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.1
  have c54856 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.1
  have c54857 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.1
  have c54858 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54859 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54860 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_10 := arithEq_of_rows h (row := 882) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_10
  simp only [← c54855, ← c54856, ← c54857] at e_882_10
  have e_882_11 := arithEq_of_rows h (row := 882) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_11
  simp only [← c54858, ← c54859, k2, ← c54860] at e_882_11
  have hr := e_882_11
  simp only [e_882_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6864)) := by
  have c54861 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54862 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54863 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54864 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54865 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54866 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_12 := arithEq_of_rows h (row := 882) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_12
  simp only [← c54861, ← c54862, ← c54863] at e_882_12
  have e_882_13 := arithEq_of_rows h (row := 882) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_13
  simp only [← c54864, ← c54865, k2, ← c54866] at e_882_13
  have hr := e_882_13
  simp only [e_882_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6865)) := by
  have c54867 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54868 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54869 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54870 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54871 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54872 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_14 := arithEq_of_rows h (row := 882) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_14
  simp only [← c54867, ← c54868, ← c54869] at e_882_14
  have e_882_15 := arithEq_of_rows h (row := 882) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_15
  simp only [← c54870, ← c54871, k2, ← c54872] at e_882_15
  have hr := e_882_15
  simp only [e_882_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6866)) := by
  have c54873 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54874 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54875 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54876 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54877 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54878 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_16 := arithEq_of_rows h (row := 882) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_16
  simp only [← c54873, ← c54874, ← c54875] at e_882_16
  have e_882_17 := arithEq_of_rows h (row := 882) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_17
  simp only [← c54876, ← c54877, k2, ← c54878] at e_882_17
  have hr := e_882_17
  simp only [e_882_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 882 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6867)) := by
  have c54879 := (publicBatchWrapper86_copies1714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54880 := (publicBatchWrapper86_copies1715 a h).1
  have c54881 := (publicBatchWrapper86_copies1715 a h).2.1
  have c54882 := (publicBatchWrapper86_copies1715 a h).2.2.1
  have c54883 := (publicBatchWrapper86_copies1715 a h).2.2.2.1
  have c54884 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_882_18 := arithEq_of_rows h (row := 882) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_18
  simp only [← c54879, ← c54880, ← c54881] at e_882_18
  have e_882_19 := arithEq_of_rows h (row := 882) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_882_19
  simp only [← c54882, ← c54883, k2, ← c54884] at e_882_19
  have hr := e_882_19
  simp only [e_882_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6868)) := by
  have c54885 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.1
  have c54886 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.1
  have c54887 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.1
  have c54888 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.1
  have c54889 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.1
  have c54890 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_0 := arithEq_of_rows h (row := 883) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_0
  simp only [← c54885, ← c54886, ← c54887] at e_883_0
  have e_883_1 := arithEq_of_rows h (row := 883) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_1
  simp only [← c54888, ← c54889, k2, ← c54890] at e_883_1
  have hr := e_883_1
  simp only [e_883_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6869)) := by
  have c54891 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54892 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54893 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54894 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54895 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54896 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_2 := arithEq_of_rows h (row := 883) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_2
  simp only [← c54891, ← c54892, ← c54893] at e_883_2
  have e_883_3 := arithEq_of_rows h (row := 883) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_3
  simp only [← c54894, ← c54895, k2, ← c54896] at e_883_3
  have hr := e_883_3
  simp only [e_883_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6870)) := by
  have c54897 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54898 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54899 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54900 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54901 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54902 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_4 := arithEq_of_rows h (row := 883) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_4
  simp only [← c54897, ← c54898, ← c54899] at e_883_4
  have e_883_5 := arithEq_of_rows h (row := 883) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_5
  simp only [← c54900, ← c54901, k2, ← c54902] at e_883_5
  have hr := e_883_5
  simp only [e_883_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6871)) := by
  have c54903 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54904 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54905 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54906 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54907 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54908 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_6 := arithEq_of_rows h (row := 883) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_6
  simp only [← c54903, ← c54904, ← c54905] at e_883_6
  have e_883_7 := arithEq_of_rows h (row := 883) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_7
  simp only [← c54906, ← c54907, k2, ← c54908] at e_883_7
  have hr := e_883_7
  simp only [e_883_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6872)) := by
  have c54909 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54910 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54911 := (publicBatchWrapper86_copies1715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54912 := (publicBatchWrapper86_copies1716 a h).1
  have c54913 := (publicBatchWrapper86_copies1716 a h).2.1
  have c54914 := (publicBatchWrapper86_copies1716 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_8 := arithEq_of_rows h (row := 883) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_8
  simp only [← c54909, ← c54910, ← c54911] at e_883_8
  have e_883_9 := arithEq_of_rows h (row := 883) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_9
  simp only [← c54912, ← c54913, k2, ← c54914] at e_883_9
  have hr := e_883_9
  simp only [e_883_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6873)) := by
  have c54915 := (publicBatchWrapper86_copies1716 a h).2.2.2.1
  have c54916 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.1
  have c54917 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.1
  have c54918 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.1
  have c54919 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.1
  have c54920 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_10 := arithEq_of_rows h (row := 883) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_10
  simp only [← c54915, ← c54916, ← c54917] at e_883_10
  have e_883_11 := arithEq_of_rows h (row := 883) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_11
  simp only [← c54918, ← c54919, k2, ← c54920] at e_883_11
  have hr := e_883_11
  simp only [e_883_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6874)) := by
  have c54921 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.1
  have c54922 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54923 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54924 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54925 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54926 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_12 := arithEq_of_rows h (row := 883) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_12
  simp only [← c54921, ← c54922, ← c54923] at e_883_12
  have e_883_13 := arithEq_of_rows h (row := 883) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_13
  simp only [← c54924, ← c54925, k2, ← c54926] at e_883_13
  have hr := e_883_13
  simp only [e_883_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6875)) := by
  have c54927 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54928 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54929 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54930 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54931 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54932 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_14 := arithEq_of_rows h (row := 883) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_14
  simp only [← c54927, ← c54928, ← c54929] at e_883_14
  have e_883_15 := arithEq_of_rows h (row := 883) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_15
  simp only [← c54930, ← c54931, k2, ← c54932] at e_883_15
  have hr := e_883_15
  simp only [e_883_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6876)) := by
  have c54933 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54934 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54935 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54936 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54937 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54938 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_16 := arithEq_of_rows h (row := 883) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_16
  simp only [← c54933, ← c54934, ← c54935] at e_883_16
  have e_883_17 := arithEq_of_rows h (row := 883) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_17
  simp only [← c54936, ← c54937, k2, ← c54938] at e_883_17
  have hr := e_883_17
  simp only [e_883_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 883 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6877)) := by
  have c54939 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54940 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54941 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54942 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54943 := (publicBatchWrapper86_copies1716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54944 := (publicBatchWrapper86_copies1717 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_883_18 := arithEq_of_rows h (row := 883) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_18
  simp only [← c54939, ← c54940, ← c54941] at e_883_18
  have e_883_19 := arithEq_of_rows h (row := 883) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_883_19
  simp only [← c54942, ← c54943, k2, ← c54944] at e_883_19
  have hr := e_883_19
  simp only [e_883_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6878)) := by
  have c54945 := (publicBatchWrapper86_copies1717 a h).2.1
  have c54946 := (publicBatchWrapper86_copies1717 a h).2.2.1
  have c54947 := (publicBatchWrapper86_copies1717 a h).2.2.2.1
  have c54948 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.1
  have c54949 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.1
  have c54950 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_0 := arithEq_of_rows h (row := 884) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_0
  simp only [← c54945, ← c54946, ← c54947] at e_884_0
  have e_884_1 := arithEq_of_rows h (row := 884) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_1
  simp only [← c54948, ← c54949, k2, ← c54950] at e_884_1
  have hr := e_884_1
  simp only [e_884_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6879)) := by
  have c54951 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.1
  have c54952 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.1
  have c54953 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.1
  have c54954 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c54955 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54956 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_2 := arithEq_of_rows h (row := 884) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_2
  simp only [← c54951, ← c54952, ← c54953] at e_884_2
  have e_884_3 := arithEq_of_rows h (row := 884) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_3
  simp only [← c54954, ← c54955, k2, ← c54956] at e_884_3
  have hr := e_884_3
  simp only [e_884_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6880)) := by
  have c54957 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54958 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54959 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54960 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54961 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54962 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_4 := arithEq_of_rows h (row := 884) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_4
  simp only [← c54957, ← c54958, ← c54959] at e_884_4
  have e_884_5 := arithEq_of_rows h (row := 884) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_5
  simp only [← c54960, ← c54961, k2, ← c54962] at e_884_5
  have hr := e_884_5
  simp only [e_884_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6881)) := by
  have c54963 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54964 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54965 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54966 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54967 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54968 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_6 := arithEq_of_rows h (row := 884) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_6
  simp only [← c54963, ← c54964, ← c54965] at e_884_6
  have e_884_7 := arithEq_of_rows h (row := 884) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_7
  simp only [← c54966, ← c54967, k2, ← c54968] at e_884_7
  have hr := e_884_7
  simp only [e_884_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6882)) := by
  have c54969 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54970 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54971 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54972 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54973 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54974 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_8 := arithEq_of_rows h (row := 884) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_8
  simp only [← c54969, ← c54970, ← c54971] at e_884_8
  have e_884_9 := arithEq_of_rows h (row := 884) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_9
  simp only [← c54972, ← c54973, k2, ← c54974] at e_884_9
  have hr := e_884_9
  simp only [e_884_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6883)) := by
  have c54975 := (publicBatchWrapper86_copies1717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c54976 := (publicBatchWrapper86_copies1718 a h).1
  have c54977 := (publicBatchWrapper86_copies1718 a h).2.1
  have c54978 := (publicBatchWrapper86_copies1718 a h).2.2.1
  have c54979 := (publicBatchWrapper86_copies1718 a h).2.2.2.1
  have c54980 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_10 := arithEq_of_rows h (row := 884) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_10
  simp only [← c54975, ← c54976, ← c54977] at e_884_10
  have e_884_11 := arithEq_of_rows h (row := 884) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_11
  simp only [← c54978, ← c54979, k2, ← c54980] at e_884_11
  have hr := e_884_11
  simp only [e_884_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6884)) := by
  have c54981 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.1
  have c54982 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.1
  have c54983 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.1
  have c54984 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.1
  have c54985 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.1
  have c54986 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_12 := arithEq_of_rows h (row := 884) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_12
  simp only [← c54981, ← c54982, ← c54983] at e_884_12
  have e_884_13 := arithEq_of_rows h (row := 884) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_13
  simp only [← c54984, ← c54985, k2, ← c54986] at e_884_13
  have hr := e_884_13
  simp only [e_884_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6885)) := by
  have c54987 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c54988 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54989 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54990 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54991 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54992 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_14 := arithEq_of_rows h (row := 884) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_14
  simp only [← c54987, ← c54988, ← c54989] at e_884_14
  have e_884_15 := arithEq_of_rows h (row := 884) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_15
  simp only [← c54990, ← c54991, k2, ← c54992] at e_884_15
  have hr := e_884_15
  simp only [e_884_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6886)) := by
  have c54993 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54994 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54995 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54996 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54997 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c54998 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_16 := arithEq_of_rows h (row := 884) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_16
  simp only [← c54993, ← c54994, ← c54995] at e_884_16
  have e_884_17 := arithEq_of_rows h (row := 884) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_17
  simp only [← c54996, ← c54997, k2, ← c54998] at e_884_17
  have hr := e_884_17
  simp only [e_884_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 884 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6887)) := by
  have c54999 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55000 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55001 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55002 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55003 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55004 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_884_18 := arithEq_of_rows h (row := 884) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_18
  simp only [← c54999, ← c55000, ← c55001] at e_884_18
  have e_884_19 := arithEq_of_rows h (row := 884) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_884_19
  simp only [← c55002, ← c55003, k2, ← c55004] at e_884_19
  have hr := e_884_19
  simp only [e_884_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6888)) := by
  have c55005 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55006 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55007 := (publicBatchWrapper86_copies1718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55008 := (publicBatchWrapper86_copies1719 a h).1
  have c55009 := (publicBatchWrapper86_copies1719 a h).2.1
  have c55010 := (publicBatchWrapper86_copies1719 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_0 := arithEq_of_rows h (row := 885) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_0
  simp only [← c55005, ← c55006, ← c55007] at e_885_0
  have e_885_1 := arithEq_of_rows h (row := 885) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_1
  simp only [← c55008, ← c55009, k2, ← c55010] at e_885_1
  have hr := e_885_1
  simp only [e_885_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6889)) := by
  have c55011 := (publicBatchWrapper86_copies1719 a h).2.2.2.1
  have c55012 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.1
  have c55013 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.1
  have c55014 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.1
  have c55015 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.1
  have c55016 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_2 := arithEq_of_rows h (row := 885) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_2
  simp only [← c55011, ← c55012, ← c55013] at e_885_2
  have e_885_3 := arithEq_of_rows h (row := 885) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_3
  simp only [← c55014, ← c55015, k2, ← c55016] at e_885_3
  have hr := e_885_3
  simp only [e_885_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6890)) := by
  have c55017 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.1
  have c55018 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55019 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55020 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55021 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55022 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_4 := arithEq_of_rows h (row := 885) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_4
  simp only [← c55017, ← c55018, ← c55019] at e_885_4
  have e_885_5 := arithEq_of_rows h (row := 885) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_5
  simp only [← c55020, ← c55021, k2, ← c55022] at e_885_5
  have hr := e_885_5
  simp only [e_885_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6891)) := by
  have c55023 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55024 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55025 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55026 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55027 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55028 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_6 := arithEq_of_rows h (row := 885) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_6
  simp only [← c55023, ← c55024, ← c55025] at e_885_6
  have e_885_7 := arithEq_of_rows h (row := 885) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_7
  simp only [← c55026, ← c55027, k2, ← c55028] at e_885_7
  have hr := e_885_7
  simp only [e_885_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6892)) := by
  have c55029 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55030 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55031 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55032 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55033 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55034 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_8 := arithEq_of_rows h (row := 885) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_8
  simp only [← c55029, ← c55030, ← c55031] at e_885_8
  have e_885_9 := arithEq_of_rows h (row := 885) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_9
  simp only [← c55032, ← c55033, k2, ← c55034] at e_885_9
  have hr := e_885_9
  simp only [e_885_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6893)) := by
  have c55035 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55036 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55037 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55038 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55039 := (publicBatchWrapper86_copies1719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55040 := (publicBatchWrapper86_copies1720 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_10 := arithEq_of_rows h (row := 885) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_10
  simp only [← c55035, ← c55036, ← c55037] at e_885_10
  have e_885_11 := arithEq_of_rows h (row := 885) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_11
  simp only [← c55038, ← c55039, k2, ← c55040] at e_885_11
  have hr := e_885_11
  simp only [e_885_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6894)) := by
  have c55041 := (publicBatchWrapper86_copies1720 a h).2.1
  have c55042 := (publicBatchWrapper86_copies1720 a h).2.2.1
  have c55043 := (publicBatchWrapper86_copies1720 a h).2.2.2.1
  have c55044 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.1
  have c55045 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.1
  have c55046 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_12 := arithEq_of_rows h (row := 885) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_12
  simp only [← c55041, ← c55042, ← c55043] at e_885_12
  have e_885_13 := arithEq_of_rows h (row := 885) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_13
  simp only [← c55044, ← c55045, k2, ← c55046] at e_885_13
  have hr := e_885_13
  simp only [e_885_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6895)) := by
  have c55047 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.1
  have c55048 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.1
  have c55049 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.1
  have c55050 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55051 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55052 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_14 := arithEq_of_rows h (row := 885) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_14
  simp only [← c55047, ← c55048, ← c55049] at e_885_14
  have e_885_15 := arithEq_of_rows h (row := 885) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_15
  simp only [← c55050, ← c55051, k2, ← c55052] at e_885_15
  have hr := e_885_15
  simp only [e_885_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6896)) := by
  have c55053 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55054 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55055 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55056 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55057 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55058 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_16 := arithEq_of_rows h (row := 885) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_16
  simp only [← c55053, ← c55054, ← c55055] at e_885_16
  have e_885_17 := arithEq_of_rows h (row := 885) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_17
  simp only [← c55056, ← c55057, k2, ← c55058] at e_885_17
  have hr := e_885_17
  simp only [e_885_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 885 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6897)) := by
  have c55059 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55060 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55061 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55062 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55063 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55064 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_885_18 := arithEq_of_rows h (row := 885) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_18
  simp only [← c55059, ← c55060, ← c55061] at e_885_18
  have e_885_19 := arithEq_of_rows h (row := 885) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_885_19
  simp only [← c55062, ← c55063, k2, ← c55064] at e_885_19
  have hr := e_885_19
  simp only [e_885_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6898)) := by
  have c55065 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55066 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55067 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55068 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55069 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55070 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_0 := arithEq_of_rows h (row := 886) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_0
  simp only [← c55065, ← c55066, ← c55067] at e_886_0
  have e_886_1 := arithEq_of_rows h (row := 886) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_1
  simp only [← c55068, ← c55069, k2, ← c55070] at e_886_1
  have hr := e_886_1
  simp only [e_886_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6899)) := by
  have c55071 := (publicBatchWrapper86_copies1720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55072 := (publicBatchWrapper86_copies1721 a h).1
  have c55073 := (publicBatchWrapper86_copies1721 a h).2.1
  have c55074 := (publicBatchWrapper86_copies1721 a h).2.2.1
  have c55075 := (publicBatchWrapper86_copies1721 a h).2.2.2.1
  have c55076 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_2 := arithEq_of_rows h (row := 886) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_2
  simp only [← c55071, ← c55072, ← c55073] at e_886_2
  have e_886_3 := arithEq_of_rows h (row := 886) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_3
  simp only [← c55074, ← c55075, k2, ← c55076] at e_886_3
  have hr := e_886_3
  simp only [e_886_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6900)) := by
  have c55077 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.1
  have c55078 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.1
  have c55079 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.1
  have c55080 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.1
  have c55081 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.1
  have c55082 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_4 := arithEq_of_rows h (row := 886) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_4
  simp only [← c55077, ← c55078, ← c55079] at e_886_4
  have e_886_5 := arithEq_of_rows h (row := 886) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_5
  simp only [← c55080, ← c55081, k2, ← c55082] at e_886_5
  have hr := e_886_5
  simp only [e_886_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6901)) := by
  have c55083 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55084 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55085 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55086 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55087 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55088 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_6 := arithEq_of_rows h (row := 886) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_6
  simp only [← c55083, ← c55084, ← c55085] at e_886_6
  have e_886_7 := arithEq_of_rows h (row := 886) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_7
  simp only [← c55086, ← c55087, k2, ← c55088] at e_886_7
  have hr := e_886_7
  simp only [e_886_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6902)) := by
  have c55089 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55090 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55091 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55092 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55093 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55094 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_8 := arithEq_of_rows h (row := 886) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_8
  simp only [← c55089, ← c55090, ← c55091] at e_886_8
  have e_886_9 := arithEq_of_rows h (row := 886) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_9
  simp only [← c55092, ← c55093, k2, ← c55094] at e_886_9
  have hr := e_886_9
  simp only [e_886_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6903)) := by
  have c55095 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55096 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55097 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55098 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55099 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55100 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_10 := arithEq_of_rows h (row := 886) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_10
  simp only [← c55095, ← c55096, ← c55097] at e_886_10
  have e_886_11 := arithEq_of_rows h (row := 886) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_11
  simp only [← c55098, ← c55099, k2, ← c55100] at e_886_11
  have hr := e_886_11
  simp only [e_886_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6904)) := by
  have c55101 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55102 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55103 := (publicBatchWrapper86_copies1721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55104 := (publicBatchWrapper86_copies1722 a h).1
  have c55105 := (publicBatchWrapper86_copies1722 a h).2.1
  have c55106 := (publicBatchWrapper86_copies1722 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_12 := arithEq_of_rows h (row := 886) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_12
  simp only [← c55101, ← c55102, ← c55103] at e_886_12
  have e_886_13 := arithEq_of_rows h (row := 886) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_13
  simp only [← c55104, ← c55105, k2, ← c55106] at e_886_13
  have hr := e_886_13
  simp only [e_886_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6905)) := by
  have c55107 := (publicBatchWrapper86_copies1722 a h).2.2.2.1
  have c55108 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.1
  have c55109 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.1
  have c55110 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.1
  have c55111 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.1
  have c55112 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_14 := arithEq_of_rows h (row := 886) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_14
  simp only [← c55107, ← c55108, ← c55109] at e_886_14
  have e_886_15 := arithEq_of_rows h (row := 886) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_15
  simp only [← c55110, ← c55111, k2, ← c55112] at e_886_15
  have hr := e_886_15
  simp only [e_886_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6906)) := by
  have c55113 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.1
  have c55114 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55115 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55116 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55117 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55118 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_16 := arithEq_of_rows h (row := 886) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_16
  simp only [← c55113, ← c55114, ← c55115] at e_886_16
  have e_886_17 := arithEq_of_rows h (row := 886) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_17
  simp only [← c55116, ← c55117, k2, ← c55118] at e_886_17
  have hr := e_886_17
  simp only [e_886_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 886 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6907)) := by
  have c55119 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55120 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55121 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55122 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55123 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55124 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_886_18 := arithEq_of_rows h (row := 886) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_18
  simp only [← c55119, ← c55120, ← c55121] at e_886_18
  have e_886_19 := arithEq_of_rows h (row := 886) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_886_19
  simp only [← c55122, ← c55123, k2, ← c55124] at e_886_19
  have hr := e_886_19
  simp only [e_886_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6908)) := by
  have c55125 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55126 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55127 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55128 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55129 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55130 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_0 := arithEq_of_rows h (row := 887) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_0
  simp only [← c55125, ← c55126, ← c55127] at e_887_0
  have e_887_1 := arithEq_of_rows h (row := 887) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_1
  simp only [← c55128, ← c55129, k2, ← c55130] at e_887_1
  have hr := e_887_1
  simp only [e_887_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6909)) := by
  have c55131 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55132 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55133 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55134 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55135 := (publicBatchWrapper86_copies1722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55136 := (publicBatchWrapper86_copies1723 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_2 := arithEq_of_rows h (row := 887) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_2
  simp only [← c55131, ← c55132, ← c55133] at e_887_2
  have e_887_3 := arithEq_of_rows h (row := 887) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_3
  simp only [← c55134, ← c55135, k2, ← c55136] at e_887_3
  have hr := e_887_3
  simp only [e_887_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6910)) := by
  have c55137 := (publicBatchWrapper86_copies1723 a h).2.1
  have c55138 := (publicBatchWrapper86_copies1723 a h).2.2.1
  have c55139 := (publicBatchWrapper86_copies1723 a h).2.2.2.1
  have c55140 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.1
  have c55141 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.1
  have c55142 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_4 := arithEq_of_rows h (row := 887) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_4
  simp only [← c55137, ← c55138, ← c55139] at e_887_4
  have e_887_5 := arithEq_of_rows h (row := 887) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_5
  simp only [← c55140, ← c55141, k2, ← c55142] at e_887_5
  have hr := e_887_5
  simp only [e_887_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6911)) := by
  have c55143 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.1
  have c55144 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.1
  have c55145 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.1
  have c55146 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55147 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55148 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_6 := arithEq_of_rows h (row := 887) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_6
  simp only [← c55143, ← c55144, ← c55145] at e_887_6
  have e_887_7 := arithEq_of_rows h (row := 887) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_7
  simp only [← c55146, ← c55147, k2, ← c55148] at e_887_7
  have hr := e_887_7
  simp only [e_887_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6912)) := by
  have c55149 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55150 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55151 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55152 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55153 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55154 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_8 := arithEq_of_rows h (row := 887) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_8
  simp only [← c55149, ← c55150, ← c55151] at e_887_8
  have e_887_9 := arithEq_of_rows h (row := 887) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_9
  simp only [← c55152, ← c55153, k2, ← c55154] at e_887_9
  have hr := e_887_9
  simp only [e_887_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6913)) := by
  have c55155 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55156 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55157 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55158 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55159 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55160 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_10 := arithEq_of_rows h (row := 887) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_10
  simp only [← c55155, ← c55156, ← c55157] at e_887_10
  have e_887_11 := arithEq_of_rows h (row := 887) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_11
  simp only [← c55158, ← c55159, k2, ← c55160] at e_887_11
  have hr := e_887_11
  simp only [e_887_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6914)) := by
  have c55161 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55162 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55163 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55164 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55165 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55166 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_12 := arithEq_of_rows h (row := 887) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_12
  simp only [← c55161, ← c55162, ← c55163] at e_887_12
  have e_887_13 := arithEq_of_rows h (row := 887) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_13
  simp only [← c55164, ← c55165, k2, ← c55166] at e_887_13
  have hr := e_887_13
  simp only [e_887_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6915)) := by
  have c55167 := (publicBatchWrapper86_copies1723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55168 := (publicBatchWrapper86_copies1724 a h).1
  have c55169 := (publicBatchWrapper86_copies1724 a h).2.1
  have c55170 := (publicBatchWrapper86_copies1724 a h).2.2.1
  have c55171 := (publicBatchWrapper86_copies1724 a h).2.2.2.1
  have c55172 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_14 := arithEq_of_rows h (row := 887) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_14
  simp only [← c55167, ← c55168, ← c55169] at e_887_14
  have e_887_15 := arithEq_of_rows h (row := 887) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_15
  simp only [← c55170, ← c55171, k2, ← c55172] at e_887_15
  have hr := e_887_15
  simp only [e_887_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6916)) := by
  have c55173 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.1
  have c55174 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.1
  have c55175 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.1
  have c55176 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.1
  have c55177 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.1
  have c55178 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_16 := arithEq_of_rows h (row := 887) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_16
  simp only [← c55173, ← c55174, ← c55175] at e_887_16
  have e_887_17 := arithEq_of_rows h (row := 887) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_17
  simp only [← c55176, ← c55177, k2, ← c55178] at e_887_17
  have hr := e_887_17
  simp only [e_887_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 887 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6917)) := by
  have c55179 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55180 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55181 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55182 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55183 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55184 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_887_18 := arithEq_of_rows h (row := 887) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_18
  simp only [← c55179, ← c55180, ← c55181] at e_887_18
  have e_887_19 := arithEq_of_rows h (row := 887) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_887_19
  simp only [← c55182, ← c55183, k2, ← c55184] at e_887_19
  have hr := e_887_19
  simp only [e_887_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6918)) := by
  have c55185 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55186 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55187 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55188 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55189 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55190 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_0 := arithEq_of_rows h (row := 888) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_0
  simp only [← c55185, ← c55186, ← c55187] at e_888_0
  have e_888_1 := arithEq_of_rows h (row := 888) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_1
  simp only [← c55188, ← c55189, k2, ← c55190] at e_888_1
  have hr := e_888_1
  simp only [e_888_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6919)) := by
  have c55191 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55192 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55193 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55194 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55195 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55196 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_2 := arithEq_of_rows h (row := 888) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_2
  simp only [← c55191, ← c55192, ← c55193] at e_888_2
  have e_888_3 := arithEq_of_rows h (row := 888) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_3
  simp only [← c55194, ← c55195, k2, ← c55196] at e_888_3
  have hr := e_888_3
  simp only [e_888_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6920)) := by
  have c55197 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55198 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55199 := (publicBatchWrapper86_copies1724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55200 := (publicBatchWrapper86_copies1725 a h).1
  have c55201 := (publicBatchWrapper86_copies1725 a h).2.1
  have c55202 := (publicBatchWrapper86_copies1725 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_4 := arithEq_of_rows h (row := 888) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_4
  simp only [← c55197, ← c55198, ← c55199] at e_888_4
  have e_888_5 := arithEq_of_rows h (row := 888) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_5
  simp only [← c55200, ← c55201, k2, ← c55202] at e_888_5
  have hr := e_888_5
  simp only [e_888_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6921)) := by
  have c55203 := (publicBatchWrapper86_copies1725 a h).2.2.2.1
  have c55204 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.1
  have c55205 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.1
  have c55206 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.1
  have c55207 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.1
  have c55208 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_6 := arithEq_of_rows h (row := 888) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_6
  simp only [← c55203, ← c55204, ← c55205] at e_888_6
  have e_888_7 := arithEq_of_rows h (row := 888) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_7
  simp only [← c55206, ← c55207, k2, ← c55208] at e_888_7
  have hr := e_888_7
  simp only [e_888_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6922)) := by
  have c55209 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.1
  have c55210 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55211 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55212 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55213 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55214 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_8 := arithEq_of_rows h (row := 888) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_8
  simp only [← c55209, ← c55210, ← c55211] at e_888_8
  have e_888_9 := arithEq_of_rows h (row := 888) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_9
  simp only [← c55212, ← c55213, k2, ← c55214] at e_888_9
  have hr := e_888_9
  simp only [e_888_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6923)) := by
  have c55215 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55216 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55217 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55218 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55219 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55220 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_10 := arithEq_of_rows h (row := 888) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_10
  simp only [← c55215, ← c55216, ← c55217] at e_888_10
  have e_888_11 := arithEq_of_rows h (row := 888) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_11
  simp only [← c55218, ← c55219, k2, ← c55220] at e_888_11
  have hr := e_888_11
  simp only [e_888_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6924)) := by
  have c55221 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55222 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55223 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55224 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55225 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55226 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_12 := arithEq_of_rows h (row := 888) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_12
  simp only [← c55221, ← c55222, ← c55223] at e_888_12
  have e_888_13 := arithEq_of_rows h (row := 888) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_13
  simp only [← c55224, ← c55225, k2, ← c55226] at e_888_13
  have hr := e_888_13
  simp only [e_888_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6925)) := by
  have c55227 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55228 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55229 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55230 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55231 := (publicBatchWrapper86_copies1725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55232 := (publicBatchWrapper86_copies1726 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_14 := arithEq_of_rows h (row := 888) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_14
  simp only [← c55227, ← c55228, ← c55229] at e_888_14
  have e_888_15 := arithEq_of_rows h (row := 888) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_15
  simp only [← c55230, ← c55231, k2, ← c55232] at e_888_15
  have hr := e_888_15
  simp only [e_888_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6926)) := by
  have c55233 := (publicBatchWrapper86_copies1726 a h).2.1
  have c55234 := (publicBatchWrapper86_copies1726 a h).2.2.1
  have c55235 := (publicBatchWrapper86_copies1726 a h).2.2.2.1
  have c55236 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.1
  have c55237 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.1
  have c55238 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_16 := arithEq_of_rows h (row := 888) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_16
  simp only [← c55233, ← c55234, ← c55235] at e_888_16
  have e_888_17 := arithEq_of_rows h (row := 888) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_17
  simp only [← c55236, ← c55237, k2, ← c55238] at e_888_17
  have hr := e_888_17
  simp only [e_888_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 888 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6927)) := by
  have c55239 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.1
  have c55240 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.1
  have c55241 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.1
  have c55242 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55243 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55244 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_888_18 := arithEq_of_rows h (row := 888) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_18
  simp only [← c55239, ← c55240, ← c55241] at e_888_18
  have e_888_19 := arithEq_of_rows h (row := 888) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_888_19
  simp only [← c55242, ← c55243, k2, ← c55244] at e_888_19
  have hr := e_888_19
  simp only [e_888_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6928)) := by
  have c55245 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55246 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55247 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55248 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55249 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55250 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_0 := arithEq_of_rows h (row := 889) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_0
  simp only [← c55245, ← c55246, ← c55247] at e_889_0
  have e_889_1 := arithEq_of_rows h (row := 889) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_1
  simp only [← c55248, ← c55249, k2, ← c55250] at e_889_1
  have hr := e_889_1
  simp only [e_889_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6929)) := by
  have c55251 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55252 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55253 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55254 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55255 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55256 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_2 := arithEq_of_rows h (row := 889) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_2
  simp only [← c55251, ← c55252, ← c55253] at e_889_2
  have e_889_3 := arithEq_of_rows h (row := 889) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_3
  simp only [← c55254, ← c55255, k2, ← c55256] at e_889_3
  have hr := e_889_3
  simp only [e_889_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6930)) := by
  have c55257 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55258 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55259 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55260 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55261 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55262 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_4 := arithEq_of_rows h (row := 889) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_4
  simp only [← c55257, ← c55258, ← c55259] at e_889_4
  have e_889_5 := arithEq_of_rows h (row := 889) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_5
  simp only [← c55260, ← c55261, k2, ← c55262] at e_889_5
  have hr := e_889_5
  simp only [e_889_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6931)) := by
  have c55263 := (publicBatchWrapper86_copies1726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55264 := (publicBatchWrapper86_copies1727 a h).1
  have c55265 := (publicBatchWrapper86_copies1727 a h).2.1
  have c55266 := (publicBatchWrapper86_copies1727 a h).2.2.1
  have c55267 := (publicBatchWrapper86_copies1727 a h).2.2.2.1
  have c55268 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_6 := arithEq_of_rows h (row := 889) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_6
  simp only [← c55263, ← c55264, ← c55265] at e_889_6
  have e_889_7 := arithEq_of_rows h (row := 889) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_7
  simp only [← c55266, ← c55267, k2, ← c55268] at e_889_7
  have hr := e_889_7
  simp only [e_889_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6932)) := by
  have c55269 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.1
  have c55270 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.1
  have c55271 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.1
  have c55272 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.1
  have c55273 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.1
  have c55274 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_8 := arithEq_of_rows h (row := 889) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_8
  simp only [← c55269, ← c55270, ← c55271] at e_889_8
  have e_889_9 := arithEq_of_rows h (row := 889) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_9
  simp only [← c55272, ← c55273, k2, ← c55274] at e_889_9
  have hr := e_889_9
  simp only [e_889_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6933)) := by
  have c55275 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55276 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55277 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55278 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55279 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55280 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_10 := arithEq_of_rows h (row := 889) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_10
  simp only [← c55275, ← c55276, ← c55277] at e_889_10
  have e_889_11 := arithEq_of_rows h (row := 889) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_11
  simp only [← c55278, ← c55279, k2, ← c55280] at e_889_11
  have hr := e_889_11
  simp only [e_889_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6934)) := by
  have c55281 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55282 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55283 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55284 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55285 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55286 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_12 := arithEq_of_rows h (row := 889) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_12
  simp only [← c55281, ← c55282, ← c55283] at e_889_12
  have e_889_13 := arithEq_of_rows h (row := 889) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_13
  simp only [← c55284, ← c55285, k2, ← c55286] at e_889_13
  have hr := e_889_13
  simp only [e_889_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6935)) := by
  have c55287 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55288 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55289 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55290 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55291 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55292 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_14 := arithEq_of_rows h (row := 889) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_14
  simp only [← c55287, ← c55288, ← c55289] at e_889_14
  have e_889_15 := arithEq_of_rows h (row := 889) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_15
  simp only [← c55290, ← c55291, k2, ← c55292] at e_889_15
  have hr := e_889_15
  simp only [e_889_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6936)) := by
  have c55293 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55294 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55295 := (publicBatchWrapper86_copies1727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55296 := (publicBatchWrapper86_copies1728 a h).1
  have c55297 := (publicBatchWrapper86_copies1728 a h).2.1
  have c55298 := (publicBatchWrapper86_copies1728 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_16 := arithEq_of_rows h (row := 889) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_16
  simp only [← c55293, ← c55294, ← c55295] at e_889_16
  have e_889_17 := arithEq_of_rows h (row := 889) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_17
  simp only [← c55296, ← c55297, k2, ← c55298] at e_889_17
  have hr := e_889_17
  simp only [e_889_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 889 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6937)) := by
  have c55299 := (publicBatchWrapper86_copies1728 a h).2.2.2.1
  have c55300 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.1
  have c55301 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.1
  have c55302 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.1
  have c55303 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.1
  have c55304 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_889_18 := arithEq_of_rows h (row := 889) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_18
  simp only [← c55299, ← c55300, ← c55301] at e_889_18
  have e_889_19 := arithEq_of_rows h (row := 889) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_889_19
  simp only [← c55302, ← c55303, k2, ← c55304] at e_889_19
  have hr := e_889_19
  simp only [e_889_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6938)) := by
  have c55305 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.1
  have c55306 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55307 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55308 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55309 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55310 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_0 := arithEq_of_rows h (row := 890) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_0
  simp only [← c55305, ← c55306, ← c55307] at e_890_0
  have e_890_1 := arithEq_of_rows h (row := 890) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_1
  simp only [← c55308, ← c55309, k2, ← c55310] at e_890_1
  have hr := e_890_1
  simp only [e_890_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6939)) := by
  have c55311 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55312 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55313 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55314 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55315 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55316 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_2 := arithEq_of_rows h (row := 890) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_2
  simp only [← c55311, ← c55312, ← c55313] at e_890_2
  have e_890_3 := arithEq_of_rows h (row := 890) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_3
  simp only [← c55314, ← c55315, k2, ← c55316] at e_890_3
  have hr := e_890_3
  simp only [e_890_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6940)) := by
  have c55317 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55318 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55319 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55320 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55321 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55322 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_4 := arithEq_of_rows h (row := 890) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_4
  simp only [← c55317, ← c55318, ← c55319] at e_890_4
  have e_890_5 := arithEq_of_rows h (row := 890) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_5
  simp only [← c55320, ← c55321, k2, ← c55322] at e_890_5
  have hr := e_890_5
  simp only [e_890_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6941)) := by
  have c55323 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55324 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55325 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55326 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55327 := (publicBatchWrapper86_copies1728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55328 := (publicBatchWrapper86_copies1729 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_6 := arithEq_of_rows h (row := 890) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_6
  simp only [← c55323, ← c55324, ← c55325] at e_890_6
  have e_890_7 := arithEq_of_rows h (row := 890) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_7
  simp only [← c55326, ← c55327, k2, ← c55328] at e_890_7
  have hr := e_890_7
  simp only [e_890_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6942)) := by
  have c55329 := (publicBatchWrapper86_copies1729 a h).2.1
  have c55330 := (publicBatchWrapper86_copies1729 a h).2.2.1
  have c55331 := (publicBatchWrapper86_copies1729 a h).2.2.2.1
  have c55332 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.1
  have c55333 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.1
  have c55334 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_8 := arithEq_of_rows h (row := 890) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_8
  simp only [← c55329, ← c55330, ← c55331] at e_890_8
  have e_890_9 := arithEq_of_rows h (row := 890) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_9
  simp only [← c55332, ← c55333, k2, ← c55334] at e_890_9
  have hr := e_890_9
  simp only [e_890_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6943)) := by
  have c55335 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.1
  have c55336 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.1
  have c55337 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.1
  have c55338 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55339 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55340 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_10 := arithEq_of_rows h (row := 890) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_10
  simp only [← c55335, ← c55336, ← c55337] at e_890_10
  have e_890_11 := arithEq_of_rows h (row := 890) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_11
  simp only [← c55338, ← c55339, k2, ← c55340] at e_890_11
  have hr := e_890_11
  simp only [e_890_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6944)) := by
  have c55341 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55342 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55343 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55344 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55345 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55346 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_12 := arithEq_of_rows h (row := 890) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_12
  simp only [← c55341, ← c55342, ← c55343] at e_890_12
  have e_890_13 := arithEq_of_rows h (row := 890) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_13
  simp only [← c55344, ← c55345, k2, ← c55346] at e_890_13
  have hr := e_890_13
  simp only [e_890_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6945)) := by
  have c55347 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55348 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55349 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55350 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55351 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55352 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_14 := arithEq_of_rows h (row := 890) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_14
  simp only [← c55347, ← c55348, ← c55349] at e_890_14
  have e_890_15 := arithEq_of_rows h (row := 890) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_15
  simp only [← c55350, ← c55351, k2, ← c55352] at e_890_15
  have hr := e_890_15
  simp only [e_890_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6946)) := by
  have c55353 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55354 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55355 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55356 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55357 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55358 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_16 := arithEq_of_rows h (row := 890) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_16
  simp only [← c55353, ← c55354, ← c55355] at e_890_16
  have e_890_17 := arithEq_of_rows h (row := 890) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_17
  simp only [← c55356, ← c55357, k2, ← c55358] at e_890_17
  have hr := e_890_17
  simp only [e_890_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 890 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6947)) := by
  have c55359 := (publicBatchWrapper86_copies1729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55360 := (publicBatchWrapper86_copies1730 a h).1
  have c55361 := (publicBatchWrapper86_copies1730 a h).2.1
  have c55362 := (publicBatchWrapper86_copies1730 a h).2.2.1
  have c55363 := (publicBatchWrapper86_copies1730 a h).2.2.2.1
  have c55364 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_890_18 := arithEq_of_rows h (row := 890) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_18
  simp only [← c55359, ← c55360, ← c55361] at e_890_18
  have e_890_19 := arithEq_of_rows h (row := 890) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_890_19
  simp only [← c55362, ← c55363, k2, ← c55364] at e_890_19
  have hr := e_890_19
  simp only [e_890_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6948)) := by
  have c55365 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.1
  have c55366 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.1
  have c55367 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.1
  have c55368 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.1
  have c55369 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.1
  have c55370 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_0 := arithEq_of_rows h (row := 891) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_0
  simp only [← c55365, ← c55366, ← c55367] at e_891_0
  have e_891_1 := arithEq_of_rows h (row := 891) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_1
  simp only [← c55368, ← c55369, k2, ← c55370] at e_891_1
  have hr := e_891_1
  simp only [e_891_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6949)) := by
  have c55371 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55372 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55373 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55374 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55375 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55376 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_2 := arithEq_of_rows h (row := 891) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_2
  simp only [← c55371, ← c55372, ← c55373] at e_891_2
  have e_891_3 := arithEq_of_rows h (row := 891) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_3
  simp only [← c55374, ← c55375, k2, ← c55376] at e_891_3
  have hr := e_891_3
  simp only [e_891_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6950)) := by
  have c55377 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55378 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55379 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55380 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55381 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55382 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_4 := arithEq_of_rows h (row := 891) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_4
  simp only [← c55377, ← c55378, ← c55379] at e_891_4
  have e_891_5 := arithEq_of_rows h (row := 891) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_5
  simp only [← c55380, ← c55381, k2, ← c55382] at e_891_5
  have hr := e_891_5
  simp only [e_891_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6951)) := by
  have c55383 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55384 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55385 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55386 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55387 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55388 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_6 := arithEq_of_rows h (row := 891) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_6
  simp only [← c55383, ← c55384, ← c55385] at e_891_6
  have e_891_7 := arithEq_of_rows h (row := 891) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_7
  simp only [← c55386, ← c55387, k2, ← c55388] at e_891_7
  have hr := e_891_7
  simp only [e_891_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6952)) := by
  have c55389 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55390 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55391 := (publicBatchWrapper86_copies1730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55392 := (publicBatchWrapper86_copies1731 a h).1
  have c55393 := (publicBatchWrapper86_copies1731 a h).2.1
  have c55394 := (publicBatchWrapper86_copies1731 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_8 := arithEq_of_rows h (row := 891) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_8
  simp only [← c55389, ← c55390, ← c55391] at e_891_8
  have e_891_9 := arithEq_of_rows h (row := 891) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_9
  simp only [← c55392, ← c55393, k2, ← c55394] at e_891_9
  have hr := e_891_9
  simp only [e_891_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6953)) := by
  have c55395 := (publicBatchWrapper86_copies1731 a h).2.2.2.1
  have c55396 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.1
  have c55397 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.1
  have c55398 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.1
  have c55399 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.1
  have c55400 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_10 := arithEq_of_rows h (row := 891) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_10
  simp only [← c55395, ← c55396, ← c55397] at e_891_10
  have e_891_11 := arithEq_of_rows h (row := 891) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_11
  simp only [← c55398, ← c55399, k2, ← c55400] at e_891_11
  have hr := e_891_11
  simp only [e_891_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6954)) := by
  have c55401 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.1
  have c55402 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55403 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55404 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55405 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55406 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_12 := arithEq_of_rows h (row := 891) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_12
  simp only [← c55401, ← c55402, ← c55403] at e_891_12
  have e_891_13 := arithEq_of_rows h (row := 891) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_13
  simp only [← c55404, ← c55405, k2, ← c55406] at e_891_13
  have hr := e_891_13
  simp only [e_891_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6955)) := by
  have c55407 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55408 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55409 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55410 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55411 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55412 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_14 := arithEq_of_rows h (row := 891) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_14
  simp only [← c55407, ← c55408, ← c55409] at e_891_14
  have e_891_15 := arithEq_of_rows h (row := 891) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_15
  simp only [← c55410, ← c55411, k2, ← c55412] at e_891_15
  have hr := e_891_15
  simp only [e_891_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6956)) := by
  have c55413 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55414 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55415 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55416 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55417 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55418 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_16 := arithEq_of_rows h (row := 891) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_16
  simp only [← c55413, ← c55414, ← c55415] at e_891_16
  have e_891_17 := arithEq_of_rows h (row := 891) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_17
  simp only [← c55416, ← c55417, k2, ← c55418] at e_891_17
  have hr := e_891_17
  simp only [e_891_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 891 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6957)) := by
  have c55419 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55420 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55421 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55422 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55423 := (publicBatchWrapper86_copies1731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55424 := (publicBatchWrapper86_copies1732 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_891_18 := arithEq_of_rows h (row := 891) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_18
  simp only [← c55419, ← c55420, ← c55421] at e_891_18
  have e_891_19 := arithEq_of_rows h (row := 891) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_891_19
  simp only [← c55422, ← c55423, k2, ← c55424] at e_891_19
  have hr := e_891_19
  simp only [e_891_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6958)) := by
  have c55425 := (publicBatchWrapper86_copies1732 a h).2.1
  have c55426 := (publicBatchWrapper86_copies1732 a h).2.2.1
  have c55427 := (publicBatchWrapper86_copies1732 a h).2.2.2.1
  have c55428 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.1
  have c55429 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.1
  have c55430 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_0 := arithEq_of_rows h (row := 892) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_0
  simp only [← c55425, ← c55426, ← c55427] at e_892_0
  have e_892_1 := arithEq_of_rows h (row := 892) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_1
  simp only [← c55428, ← c55429, k2, ← c55430] at e_892_1
  have hr := e_892_1
  simp only [e_892_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6959)) := by
  have c55431 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.1
  have c55432 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.1
  have c55433 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.1
  have c55434 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55435 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55436 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_2 := arithEq_of_rows h (row := 892) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_2
  simp only [← c55431, ← c55432, ← c55433] at e_892_2
  have e_892_3 := arithEq_of_rows h (row := 892) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_3
  simp only [← c55434, ← c55435, k2, ← c55436] at e_892_3
  have hr := e_892_3
  simp only [e_892_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6960)) := by
  have c55437 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55438 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55439 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55440 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55441 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55442 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_4 := arithEq_of_rows h (row := 892) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_4
  simp only [← c55437, ← c55438, ← c55439] at e_892_4
  have e_892_5 := arithEq_of_rows h (row := 892) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_5
  simp only [← c55440, ← c55441, k2, ← c55442] at e_892_5
  have hr := e_892_5
  simp only [e_892_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6961)) := by
  have c55443 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55444 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55445 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55446 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55447 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55448 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_6 := arithEq_of_rows h (row := 892) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_6
  simp only [← c55443, ← c55444, ← c55445] at e_892_6
  have e_892_7 := arithEq_of_rows h (row := 892) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_7
  simp only [← c55446, ← c55447, k2, ← c55448] at e_892_7
  have hr := e_892_7
  simp only [e_892_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6962)) := by
  have c55449 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55450 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55451 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55452 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55453 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55454 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_8 := arithEq_of_rows h (row := 892) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_8
  simp only [← c55449, ← c55450, ← c55451] at e_892_8
  have e_892_9 := arithEq_of_rows h (row := 892) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_9
  simp only [← c55452, ← c55453, k2, ← c55454] at e_892_9
  have hr := e_892_9
  simp only [e_892_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6963)) := by
  have c55455 := (publicBatchWrapper86_copies1732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55456 := (publicBatchWrapper86_copies1733 a h).1
  have c55457 := (publicBatchWrapper86_copies1733 a h).2.1
  have c55458 := (publicBatchWrapper86_copies1733 a h).2.2.1
  have c55459 := (publicBatchWrapper86_copies1733 a h).2.2.2.1
  have c55460 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_10 := arithEq_of_rows h (row := 892) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_10
  simp only [← c55455, ← c55456, ← c55457] at e_892_10
  have e_892_11 := arithEq_of_rows h (row := 892) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_11
  simp only [← c55458, ← c55459, k2, ← c55460] at e_892_11
  have hr := e_892_11
  simp only [e_892_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6964)) := by
  have c55461 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.1
  have c55462 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.1
  have c55463 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.1
  have c55464 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.1
  have c55465 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.1
  have c55466 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_12 := arithEq_of_rows h (row := 892) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_12
  simp only [← c55461, ← c55462, ← c55463] at e_892_12
  have e_892_13 := arithEq_of_rows h (row := 892) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_13
  simp only [← c55464, ← c55465, k2, ← c55466] at e_892_13
  have hr := e_892_13
  simp only [e_892_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6965)) := by
  have c55467 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55468 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55469 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55470 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55471 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55472 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_14 := arithEq_of_rows h (row := 892) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_14
  simp only [← c55467, ← c55468, ← c55469] at e_892_14
  have e_892_15 := arithEq_of_rows h (row := 892) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_15
  simp only [← c55470, ← c55471, k2, ← c55472] at e_892_15
  have hr := e_892_15
  simp only [e_892_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6966)) := by
  have c55473 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55474 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55475 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55476 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55477 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55478 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_16 := arithEq_of_rows h (row := 892) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_16
  simp only [← c55473, ← c55474, ← c55475] at e_892_16
  have e_892_17 := arithEq_of_rows h (row := 892) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_17
  simp only [← c55476, ← c55477, k2, ← c55478] at e_892_17
  have hr := e_892_17
  simp only [e_892_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 892 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6967)) := by
  have c55479 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55480 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55481 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55482 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55483 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55484 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_892_18 := arithEq_of_rows h (row := 892) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_18
  simp only [← c55479, ← c55480, ← c55481] at e_892_18
  have e_892_19 := arithEq_of_rows h (row := 892) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_892_19
  simp only [← c55482, ← c55483, k2, ← c55484] at e_892_19
  have hr := e_892_19
  simp only [e_892_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6968)) := by
  have c55485 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55486 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55487 := (publicBatchWrapper86_copies1733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55488 := (publicBatchWrapper86_copies1734 a h).1
  have c55489 := (publicBatchWrapper86_copies1734 a h).2.1
  have c55490 := (publicBatchWrapper86_copies1734 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_0 := arithEq_of_rows h (row := 893) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_0
  simp only [← c55485, ← c55486, ← c55487] at e_893_0
  have e_893_1 := arithEq_of_rows h (row := 893) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_1
  simp only [← c55488, ← c55489, k2, ← c55490] at e_893_1
  have hr := e_893_1
  simp only [e_893_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6969)) := by
  have c55491 := (publicBatchWrapper86_copies1734 a h).2.2.2.1
  have c55492 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.1
  have c55493 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.1
  have c55494 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.1
  have c55495 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.1
  have c55496 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_2 := arithEq_of_rows h (row := 893) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_2
  simp only [← c55491, ← c55492, ← c55493] at e_893_2
  have e_893_3 := arithEq_of_rows h (row := 893) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_3
  simp only [← c55494, ← c55495, k2, ← c55496] at e_893_3
  have hr := e_893_3
  simp only [e_893_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7010)) := by
  have c55497 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.1
  have c55498 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55499 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55500 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55501 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55502 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_4 := arithEq_of_rows h (row := 893) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_4
  simp only [← c55497, ← c55498, ← c55499] at e_893_4
  have e_893_5 := arithEq_of_rows h (row := 893) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_5
  simp only [← c55500, ← c55501, k2, ← c55502] at e_893_5
  have hr := e_893_5
  simp only [e_893_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7011)) := by
  have c55503 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55504 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55505 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55506 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55507 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55508 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_6 := arithEq_of_rows h (row := 893) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_6
  simp only [← c55503, ← c55504, ← c55505] at e_893_6
  have e_893_7 := arithEq_of_rows h (row := 893) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_7
  simp only [← c55506, ← c55507, k2, ← c55508] at e_893_7
  have hr := e_893_7
  simp only [e_893_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7012)) := by
  have c55509 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55510 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55511 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55512 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55513 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55514 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_8 := arithEq_of_rows h (row := 893) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_8
  simp only [← c55509, ← c55510, ← c55511] at e_893_8
  have e_893_9 := arithEq_of_rows h (row := 893) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_9
  simp only [← c55512, ← c55513, k2, ← c55514] at e_893_9
  have hr := e_893_9
  simp only [e_893_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7013)) := by
  have c55515 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55516 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55517 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55518 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55519 := (publicBatchWrapper86_copies1734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55520 := (publicBatchWrapper86_copies1735 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_10 := arithEq_of_rows h (row := 893) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_10
  simp only [← c55515, ← c55516, ← c55517] at e_893_10
  have e_893_11 := arithEq_of_rows h (row := 893) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_11
  simp only [← c55518, ← c55519, k2, ← c55520] at e_893_11
  have hr := e_893_11
  simp only [e_893_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7014)) := by
  have c55521 := (publicBatchWrapper86_copies1735 a h).2.1
  have c55522 := (publicBatchWrapper86_copies1735 a h).2.2.1
  have c55523 := (publicBatchWrapper86_copies1735 a h).2.2.2.1
  have c55524 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.1
  have c55525 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.1
  have c55526 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_12 := arithEq_of_rows h (row := 893) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_12
  simp only [← c55521, ← c55522, ← c55523] at e_893_12
  have e_893_13 := arithEq_of_rows h (row := 893) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_13
  simp only [← c55524, ← c55525, k2, ← c55526] at e_893_13
  have hr := e_893_13
  simp only [e_893_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7015)) := by
  have c55527 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.1
  have c55528 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.1
  have c55529 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.1
  have c55530 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55531 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55532 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_14 := arithEq_of_rows h (row := 893) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_14
  simp only [← c55527, ← c55528, ← c55529] at e_893_14
  have e_893_15 := arithEq_of_rows h (row := 893) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_15
  simp only [← c55530, ← c55531, k2, ← c55532] at e_893_15
  have hr := e_893_15
  simp only [e_893_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7016)) := by
  have c55533 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55534 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55535 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55536 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55537 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55538 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_16 := arithEq_of_rows h (row := 893) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_16
  simp only [← c55533, ← c55534, ← c55535] at e_893_16
  have e_893_17 := arithEq_of_rows h (row := 893) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_17
  simp only [← c55536, ← c55537, k2, ← c55538] at e_893_17
  have hr := e_893_17
  simp only [e_893_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 893 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7017)) := by
  have c55539 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55540 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55541 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55542 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55543 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55544 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_893_18 := arithEq_of_rows h (row := 893) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_18
  simp only [← c55539, ← c55540, ← c55541] at e_893_18
  have e_893_19 := arithEq_of_rows h (row := 893) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_893_19
  simp only [← c55542, ← c55543, k2, ← c55544] at e_893_19
  have hr := e_893_19
  simp only [e_893_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7018)) := by
  have c55545 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55546 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55547 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55548 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55549 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55550 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_0 := arithEq_of_rows h (row := 894) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_0
  simp only [← c55545, ← c55546, ← c55547] at e_894_0
  have e_894_1 := arithEq_of_rows h (row := 894) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_1
  simp only [← c55548, ← c55549, k2, ← c55550] at e_894_1
  have hr := e_894_1
  simp only [e_894_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7019)) := by
  have c55551 := (publicBatchWrapper86_copies1735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55552 := (publicBatchWrapper86_copies1736 a h).1
  have c55553 := (publicBatchWrapper86_copies1736 a h).2.1
  have c55554 := (publicBatchWrapper86_copies1736 a h).2.2.1
  have c55555 := (publicBatchWrapper86_copies1736 a h).2.2.2.1
  have c55556 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_2 := arithEq_of_rows h (row := 894) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_2
  simp only [← c55551, ← c55552, ← c55553] at e_894_2
  have e_894_3 := arithEq_of_rows h (row := 894) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_3
  simp only [← c55554, ← c55555, k2, ← c55556] at e_894_3
  have hr := e_894_3
  simp only [e_894_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7020)) := by
  have c55557 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.1
  have c55558 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.1
  have c55559 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.1
  have c55560 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.1
  have c55561 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.1
  have c55562 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_4 := arithEq_of_rows h (row := 894) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_4
  simp only [← c55557, ← c55558, ← c55559] at e_894_4
  have e_894_5 := arithEq_of_rows h (row := 894) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_5
  simp only [← c55560, ← c55561, k2, ← c55562] at e_894_5
  have hr := e_894_5
  simp only [e_894_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7021)) := by
  have c55563 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55564 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55565 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55566 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55567 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55568 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_6 := arithEq_of_rows h (row := 894) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_6
  simp only [← c55563, ← c55564, ← c55565] at e_894_6
  have e_894_7 := arithEq_of_rows h (row := 894) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_7
  simp only [← c55566, ← c55567, k2, ← c55568] at e_894_7
  have hr := e_894_7
  simp only [e_894_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7022)) := by
  have c55569 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55570 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55571 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55572 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55573 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55574 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_8 := arithEq_of_rows h (row := 894) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_8
  simp only [← c55569, ← c55570, ← c55571] at e_894_8
  have e_894_9 := arithEq_of_rows h (row := 894) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_9
  simp only [← c55572, ← c55573, k2, ← c55574] at e_894_9
  have hr := e_894_9
  simp only [e_894_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7023)) := by
  have c55575 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55576 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55577 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55578 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55579 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55580 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_10 := arithEq_of_rows h (row := 894) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_10
  simp only [← c55575, ← c55576, ← c55577] at e_894_10
  have e_894_11 := arithEq_of_rows h (row := 894) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_11
  simp only [← c55578, ← c55579, k2, ← c55580] at e_894_11
  have hr := e_894_11
  simp only [e_894_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7024)) := by
  have c55581 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55582 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55583 := (publicBatchWrapper86_copies1736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55584 := (publicBatchWrapper86_copies1737 a h).1
  have c55585 := (publicBatchWrapper86_copies1737 a h).2.1
  have c55586 := (publicBatchWrapper86_copies1737 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_12 := arithEq_of_rows h (row := 894) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_12
  simp only [← c55581, ← c55582, ← c55583] at e_894_12
  have e_894_13 := arithEq_of_rows h (row := 894) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_13
  simp only [← c55584, ← c55585, k2, ← c55586] at e_894_13
  have hr := e_894_13
  simp only [e_894_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7025)) := by
  have c55587 := (publicBatchWrapper86_copies1737 a h).2.2.2.1
  have c55588 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.1
  have c55589 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.1
  have c55590 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.1
  have c55591 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.1
  have c55592 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_14 := arithEq_of_rows h (row := 894) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_14
  simp only [← c55587, ← c55588, ← c55589] at e_894_14
  have e_894_15 := arithEq_of_rows h (row := 894) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_15
  simp only [← c55590, ← c55591, k2, ← c55592] at e_894_15
  have hr := e_894_15
  simp only [e_894_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7026)) := by
  have c55593 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.1
  have c55594 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55595 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55596 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55597 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55598 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_16 := arithEq_of_rows h (row := 894) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_16
  simp only [← c55593, ← c55594, ← c55595] at e_894_16
  have e_894_17 := arithEq_of_rows h (row := 894) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_17
  simp only [← c55596, ← c55597, k2, ← c55598] at e_894_17
  have hr := e_894_17
  simp only [e_894_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 894 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7027)) := by
  have c55599 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55600 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55601 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55602 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55603 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55604 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_894_18 := arithEq_of_rows h (row := 894) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_18
  simp only [← c55599, ← c55600, ← c55601] at e_894_18
  have e_894_19 := arithEq_of_rows h (row := 894) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_894_19
  simp only [← c55602, ← c55603, k2, ← c55604] at e_894_19
  have hr := e_894_19
  simp only [e_894_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7028)) := by
  have c55605 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55606 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55607 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55608 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55609 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55610 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_0 := arithEq_of_rows h (row := 895) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_0
  simp only [← c55605, ← c55606, ← c55607] at e_895_0
  have e_895_1 := arithEq_of_rows h (row := 895) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_1
  simp only [← c55608, ← c55609, k2, ← c55610] at e_895_1
  have hr := e_895_1
  simp only [e_895_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7029)) := by
  have c55611 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55612 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55613 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55614 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55615 := (publicBatchWrapper86_copies1737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55616 := (publicBatchWrapper86_copies1738 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_2 := arithEq_of_rows h (row := 895) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_2
  simp only [← c55611, ← c55612, ← c55613] at e_895_2
  have e_895_3 := arithEq_of_rows h (row := 895) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_3
  simp only [← c55614, ← c55615, k2, ← c55616] at e_895_3
  have hr := e_895_3
  simp only [e_895_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7030)) := by
  have c55617 := (publicBatchWrapper86_copies1738 a h).2.1
  have c55618 := (publicBatchWrapper86_copies1738 a h).2.2.1
  have c55619 := (publicBatchWrapper86_copies1738 a h).2.2.2.1
  have c55620 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.1
  have c55621 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.1
  have c55622 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_4 := arithEq_of_rows h (row := 895) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_4
  simp only [← c55617, ← c55618, ← c55619] at e_895_4
  have e_895_5 := arithEq_of_rows h (row := 895) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_5
  simp only [← c55620, ← c55621, k2, ← c55622] at e_895_5
  have hr := e_895_5
  simp only [e_895_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7031)) := by
  have c55623 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.1
  have c55624 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.1
  have c55625 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.1
  have c55626 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55627 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55628 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_6 := arithEq_of_rows h (row := 895) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_6
  simp only [← c55623, ← c55624, ← c55625] at e_895_6
  have e_895_7 := arithEq_of_rows h (row := 895) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_7
  simp only [← c55626, ← c55627, k2, ← c55628] at e_895_7
  have hr := e_895_7
  simp only [e_895_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7032)) := by
  have c55629 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55630 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55631 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55632 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55633 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55634 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_8 := arithEq_of_rows h (row := 895) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_8
  simp only [← c55629, ← c55630, ← c55631] at e_895_8
  have e_895_9 := arithEq_of_rows h (row := 895) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_9
  simp only [← c55632, ← c55633, k2, ← c55634] at e_895_9
  have hr := e_895_9
  simp only [e_895_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7033)) := by
  have c55635 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55636 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55637 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55638 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55639 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55640 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_10 := arithEq_of_rows h (row := 895) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_10
  simp only [← c55635, ← c55636, ← c55637] at e_895_10
  have e_895_11 := arithEq_of_rows h (row := 895) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_11
  simp only [← c55638, ← c55639, k2, ← c55640] at e_895_11
  have hr := e_895_11
  simp only [e_895_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7034)) := by
  have c55641 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55642 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55643 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55644 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55645 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55646 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_12 := arithEq_of_rows h (row := 895) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_12
  simp only [← c55641, ← c55642, ← c55643] at e_895_12
  have e_895_13 := arithEq_of_rows h (row := 895) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_13
  simp only [← c55644, ← c55645, k2, ← c55646] at e_895_13
  have hr := e_895_13
  simp only [e_895_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7035)) := by
  have c55647 := (publicBatchWrapper86_copies1738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55648 := (publicBatchWrapper86_copies1739 a h).1
  have c55649 := (publicBatchWrapper86_copies1739 a h).2.1
  have c55650 := (publicBatchWrapper86_copies1739 a h).2.2.1
  have c55651 := (publicBatchWrapper86_copies1739 a h).2.2.2.1
  have c55652 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_14 := arithEq_of_rows h (row := 895) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_14
  simp only [← c55647, ← c55648, ← c55649] at e_895_14
  have e_895_15 := arithEq_of_rows h (row := 895) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_15
  simp only [← c55650, ← c55651, k2, ← c55652] at e_895_15
  have hr := e_895_15
  simp only [e_895_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7036)) := by
  have c55653 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.1
  have c55654 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.1
  have c55655 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.1
  have c55656 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.1
  have c55657 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.1
  have c55658 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_16 := arithEq_of_rows h (row := 895) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_16
  simp only [← c55653, ← c55654, ← c55655] at e_895_16
  have e_895_17 := arithEq_of_rows h (row := 895) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_17
  simp only [← c55656, ← c55657, k2, ← c55658] at e_895_17
  have hr := e_895_17
  simp only [e_895_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 895 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7037)) := by
  have c55659 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55660 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55661 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55662 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55663 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55664 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_895_18 := arithEq_of_rows h (row := 895) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_18
  simp only [← c55659, ← c55660, ← c55661] at e_895_18
  have e_895_19 := arithEq_of_rows h (row := 895) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_895_19
  simp only [← c55662, ← c55663, k2, ← c55664] at e_895_19
  have hr := e_895_19
  simp only [e_895_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7038)) := by
  have c55665 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55666 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55667 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55668 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55669 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55670 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_0 := arithEq_of_rows h (row := 896) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_0
  simp only [← c55665, ← c55666, ← c55667] at e_896_0
  have e_896_1 := arithEq_of_rows h (row := 896) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_1
  simp only [← c55668, ← c55669, k2, ← c55670] at e_896_1
  have hr := e_896_1
  simp only [e_896_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7039)) := by
  have c55671 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55672 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55673 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55674 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55675 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55676 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_2 := arithEq_of_rows h (row := 896) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_2
  simp only [← c55671, ← c55672, ← c55673] at e_896_2
  have e_896_3 := arithEq_of_rows h (row := 896) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_3
  simp only [← c55674, ← c55675, k2, ← c55676] at e_896_3
  have hr := e_896_3
  simp only [e_896_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7040)) := by
  have c55677 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55678 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55679 := (publicBatchWrapper86_copies1739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55680 := (publicBatchWrapper86_copies1740 a h).1
  have c55681 := (publicBatchWrapper86_copies1740 a h).2.1
  have c55682 := (publicBatchWrapper86_copies1740 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_4 := arithEq_of_rows h (row := 896) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_4
  simp only [← c55677, ← c55678, ← c55679] at e_896_4
  have e_896_5 := arithEq_of_rows h (row := 896) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_5
  simp only [← c55680, ← c55681, k2, ← c55682] at e_896_5
  have hr := e_896_5
  simp only [e_896_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7041)) := by
  have c55683 := (publicBatchWrapper86_copies1740 a h).2.2.2.1
  have c55684 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.1
  have c55685 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.1
  have c55686 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.1
  have c55687 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.1
  have c55688 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_6 := arithEq_of_rows h (row := 896) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_6
  simp only [← c55683, ← c55684, ← c55685] at e_896_6
  have e_896_7 := arithEq_of_rows h (row := 896) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_7
  simp only [← c55686, ← c55687, k2, ← c55688] at e_896_7
  have hr := e_896_7
  simp only [e_896_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7042)) := by
  have c55689 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.1
  have c55690 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55691 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55692 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55693 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55694 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_8 := arithEq_of_rows h (row := 896) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_8
  simp only [← c55689, ← c55690, ← c55691] at e_896_8
  have e_896_9 := arithEq_of_rows h (row := 896) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_9
  simp only [← c55692, ← c55693, k2, ← c55694] at e_896_9
  have hr := e_896_9
  simp only [e_896_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7043)) := by
  have c55695 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55696 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55697 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55698 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55699 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55700 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_10 := arithEq_of_rows h (row := 896) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_10
  simp only [← c55695, ← c55696, ← c55697] at e_896_10
  have e_896_11 := arithEq_of_rows h (row := 896) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_11
  simp only [← c55698, ← c55699, k2, ← c55700] at e_896_11
  have hr := e_896_11
  simp only [e_896_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7044)) := by
  have c55701 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55702 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55703 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55704 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55705 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55706 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_12 := arithEq_of_rows h (row := 896) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_12
  simp only [← c55701, ← c55702, ← c55703] at e_896_12
  have e_896_13 := arithEq_of_rows h (row := 896) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_13
  simp only [← c55704, ← c55705, k2, ← c55706] at e_896_13
  have hr := e_896_13
  simp only [e_896_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7045)) := by
  have c55707 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55708 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55709 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55710 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55711 := (publicBatchWrapper86_copies1740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55712 := (publicBatchWrapper86_copies1741 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_14 := arithEq_of_rows h (row := 896) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_14
  simp only [← c55707, ← c55708, ← c55709] at e_896_14
  have e_896_15 := arithEq_of_rows h (row := 896) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_15
  simp only [← c55710, ← c55711, k2, ← c55712] at e_896_15
  have hr := e_896_15
  simp only [e_896_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7046)) := by
  have c55713 := (publicBatchWrapper86_copies1741 a h).2.1
  have c55714 := (publicBatchWrapper86_copies1741 a h).2.2.1
  have c55715 := (publicBatchWrapper86_copies1741 a h).2.2.2.1
  have c55716 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.1
  have c55717 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.1
  have c55718 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_16 := arithEq_of_rows h (row := 896) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_16
  simp only [← c55713, ← c55714, ← c55715] at e_896_16
  have e_896_17 := arithEq_of_rows h (row := 896) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_17
  simp only [← c55716, ← c55717, k2, ← c55718] at e_896_17
  have hr := e_896_17
  simp only [e_896_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 896 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7047)) := by
  have c55719 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.1
  have c55720 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.1
  have c55721 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.1
  have c55722 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55723 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55724 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_896_18 := arithEq_of_rows h (row := 896) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_18
  simp only [← c55719, ← c55720, ← c55721] at e_896_18
  have e_896_19 := arithEq_of_rows h (row := 896) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_896_19
  simp only [← c55722, ← c55723, k2, ← c55724] at e_896_19
  have hr := e_896_19
  simp only [e_896_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7048)) := by
  have c55725 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55726 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55727 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55728 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55729 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55730 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_0 := arithEq_of_rows h (row := 897) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_0
  simp only [← c55725, ← c55726, ← c55727] at e_897_0
  have e_897_1 := arithEq_of_rows h (row := 897) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_1
  simp only [← c55728, ← c55729, k2, ← c55730] at e_897_1
  have hr := e_897_1
  simp only [e_897_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7049)) := by
  have c55731 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55732 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55733 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55734 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55735 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55736 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_2 := arithEq_of_rows h (row := 897) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_2
  simp only [← c55731, ← c55732, ← c55733] at e_897_2
  have e_897_3 := arithEq_of_rows h (row := 897) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_3
  simp only [← c55734, ← c55735, k2, ← c55736] at e_897_3
  have hr := e_897_3
  simp only [e_897_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7050)) := by
  have c55737 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55738 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55739 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55740 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55741 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55742 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_4 := arithEq_of_rows h (row := 897) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_4
  simp only [← c55737, ← c55738, ← c55739] at e_897_4
  have e_897_5 := arithEq_of_rows h (row := 897) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_5
  simp only [← c55740, ← c55741, k2, ← c55742] at e_897_5
  have hr := e_897_5
  simp only [e_897_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7051)) := by
  have c55743 := (publicBatchWrapper86_copies1741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55744 := (publicBatchWrapper86_copies1742 a h).1
  have c55745 := (publicBatchWrapper86_copies1742 a h).2.1
  have c55746 := (publicBatchWrapper86_copies1742 a h).2.2.1
  have c55747 := (publicBatchWrapper86_copies1742 a h).2.2.2.1
  have c55748 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_6 := arithEq_of_rows h (row := 897) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_6
  simp only [← c55743, ← c55744, ← c55745] at e_897_6
  have e_897_7 := arithEq_of_rows h (row := 897) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_7
  simp only [← c55746, ← c55747, k2, ← c55748] at e_897_7
  have hr := e_897_7
  simp only [e_897_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7052)) := by
  have c55749 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.1
  have c55750 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.1
  have c55751 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.1
  have c55752 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.1
  have c55753 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.1
  have c55754 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_8 := arithEq_of_rows h (row := 897) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_8
  simp only [← c55749, ← c55750, ← c55751] at e_897_8
  have e_897_9 := arithEq_of_rows h (row := 897) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_9
  simp only [← c55752, ← c55753, k2, ← c55754] at e_897_9
  have hr := e_897_9
  simp only [e_897_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7053)) := by
  have c55755 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55756 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55757 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55758 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55759 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55760 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_10 := arithEq_of_rows h (row := 897) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_10
  simp only [← c55755, ← c55756, ← c55757] at e_897_10
  have e_897_11 := arithEq_of_rows h (row := 897) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_11
  simp only [← c55758, ← c55759, k2, ← c55760] at e_897_11
  have hr := e_897_11
  simp only [e_897_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7054)) := by
  have c55761 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55762 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55763 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55764 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55765 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55766 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_12 := arithEq_of_rows h (row := 897) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_12
  simp only [← c55761, ← c55762, ← c55763] at e_897_12
  have e_897_13 := arithEq_of_rows h (row := 897) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_13
  simp only [← c55764, ← c55765, k2, ← c55766] at e_897_13
  have hr := e_897_13
  simp only [e_897_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7055)) := by
  have c55767 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55768 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55769 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55770 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55771 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55772 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_14 := arithEq_of_rows h (row := 897) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_14
  simp only [← c55767, ← c55768, ← c55769] at e_897_14
  have e_897_15 := arithEq_of_rows h (row := 897) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_15
  simp only [← c55770, ← c55771, k2, ← c55772] at e_897_15
  have hr := e_897_15
  simp only [e_897_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7056)) := by
  have c55773 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55774 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55775 := (publicBatchWrapper86_copies1742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55776 := (publicBatchWrapper86_copies1743 a h).1
  have c55777 := (publicBatchWrapper86_copies1743 a h).2.1
  have c55778 := (publicBatchWrapper86_copies1743 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_16 := arithEq_of_rows h (row := 897) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_16
  simp only [← c55773, ← c55774, ← c55775] at e_897_16
  have e_897_17 := arithEq_of_rows h (row := 897) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_17
  simp only [← c55776, ← c55777, k2, ← c55778] at e_897_17
  have hr := e_897_17
  simp only [e_897_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 897 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7057)) := by
  have c55779 := (publicBatchWrapper86_copies1743 a h).2.2.2.1
  have c55780 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.1
  have c55781 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.1
  have c55782 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.1
  have c55783 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.1
  have c55784 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_897_18 := arithEq_of_rows h (row := 897) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_18
  simp only [← c55779, ← c55780, ← c55781] at e_897_18
  have e_897_19 := arithEq_of_rows h (row := 897) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_897_19
  simp only [← c55782, ← c55783, k2, ← c55784] at e_897_19
  have hr := e_897_19
  simp only [e_897_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7058)) := by
  have c55785 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.1
  have c55786 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55787 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55788 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55789 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55790 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_0 := arithEq_of_rows h (row := 898) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_0
  simp only [← c55785, ← c55786, ← c55787] at e_898_0
  have e_898_1 := arithEq_of_rows h (row := 898) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_1
  simp only [← c55788, ← c55789, k2, ← c55790] at e_898_1
  have hr := e_898_1
  simp only [e_898_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7059)) := by
  have c55791 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55792 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55793 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55794 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55795 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55796 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_2 := arithEq_of_rows h (row := 898) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_2
  simp only [← c55791, ← c55792, ← c55793] at e_898_2
  have e_898_3 := arithEq_of_rows h (row := 898) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_3
  simp only [← c55794, ← c55795, k2, ← c55796] at e_898_3
  have hr := e_898_3
  simp only [e_898_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7060)) := by
  have c55797 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55798 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55799 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55800 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55801 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55802 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_4 := arithEq_of_rows h (row := 898) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_4
  simp only [← c55797, ← c55798, ← c55799] at e_898_4
  have e_898_5 := arithEq_of_rows h (row := 898) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_5
  simp only [← c55800, ← c55801, k2, ← c55802] at e_898_5
  have hr := e_898_5
  simp only [e_898_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7061)) := by
  have c55803 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55804 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55805 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55806 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55807 := (publicBatchWrapper86_copies1743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55808 := (publicBatchWrapper86_copies1744 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_6 := arithEq_of_rows h (row := 898) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_6
  simp only [← c55803, ← c55804, ← c55805] at e_898_6
  have e_898_7 := arithEq_of_rows h (row := 898) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_7
  simp only [← c55806, ← c55807, k2, ← c55808] at e_898_7
  have hr := e_898_7
  simp only [e_898_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7062)) := by
  have c55809 := (publicBatchWrapper86_copies1744 a h).2.1
  have c55810 := (publicBatchWrapper86_copies1744 a h).2.2.1
  have c55811 := (publicBatchWrapper86_copies1744 a h).2.2.2.1
  have c55812 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.1
  have c55813 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.1
  have c55814 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_8 := arithEq_of_rows h (row := 898) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_8
  simp only [← c55809, ← c55810, ← c55811] at e_898_8
  have e_898_9 := arithEq_of_rows h (row := 898) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_9
  simp only [← c55812, ← c55813, k2, ← c55814] at e_898_9
  have hr := e_898_9
  simp only [e_898_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7063)) := by
  have c55815 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.1
  have c55816 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.1
  have c55817 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.1
  have c55818 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55819 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55820 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_10 := arithEq_of_rows h (row := 898) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_10
  simp only [← c55815, ← c55816, ← c55817] at e_898_10
  have e_898_11 := arithEq_of_rows h (row := 898) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_11
  simp only [← c55818, ← c55819, k2, ← c55820] at e_898_11
  have hr := e_898_11
  simp only [e_898_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7064)) := by
  have c55821 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55822 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55823 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55824 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55825 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55826 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_12 := arithEq_of_rows h (row := 898) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_12
  simp only [← c55821, ← c55822, ← c55823] at e_898_12
  have e_898_13 := arithEq_of_rows h (row := 898) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_13
  simp only [← c55824, ← c55825, k2, ← c55826] at e_898_13
  have hr := e_898_13
  simp only [e_898_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7065)) := by
  have c55827 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55828 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55829 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55830 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55831 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55832 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_14 := arithEq_of_rows h (row := 898) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_14
  simp only [← c55827, ← c55828, ← c55829] at e_898_14
  have e_898_15 := arithEq_of_rows h (row := 898) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_15
  simp only [← c55830, ← c55831, k2, ← c55832] at e_898_15
  have hr := e_898_15
  simp only [e_898_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7066)) := by
  have c55833 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55834 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55835 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55836 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55837 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55838 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_16 := arithEq_of_rows h (row := 898) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_16
  simp only [← c55833, ← c55834, ← c55835] at e_898_16
  have e_898_17 := arithEq_of_rows h (row := 898) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_17
  simp only [← c55836, ← c55837, k2, ← c55838] at e_898_17
  have hr := e_898_17
  simp only [e_898_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 898 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7067)) := by
  have c55839 := (publicBatchWrapper86_copies1744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55840 := (publicBatchWrapper86_copies1745 a h).1
  have c55841 := (publicBatchWrapper86_copies1745 a h).2.1
  have c55842 := (publicBatchWrapper86_copies1745 a h).2.2.1
  have c55843 := (publicBatchWrapper86_copies1745 a h).2.2.2.1
  have c55844 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_898_18 := arithEq_of_rows h (row := 898) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_18
  simp only [← c55839, ← c55840, ← c55841] at e_898_18
  have e_898_19 := arithEq_of_rows h (row := 898) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_898_19
  simp only [← c55842, ← c55843, k2, ← c55844] at e_898_19
  have hr := e_898_19
  simp only [e_898_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7068)) := by
  have c55845 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.1
  have c55846 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.1
  have c55847 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.1
  have c55848 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.1
  have c55849 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.1
  have c55850 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_0 := arithEq_of_rows h (row := 899) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_0
  simp only [← c55845, ← c55846, ← c55847] at e_899_0
  have e_899_1 := arithEq_of_rows h (row := 899) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_1
  simp only [← c55848, ← c55849, k2, ← c55850] at e_899_1
  have hr := e_899_1
  simp only [e_899_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7069)) := by
  have c55851 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55852 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55853 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55854 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55855 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55856 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_2 := arithEq_of_rows h (row := 899) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_2
  simp only [← c55851, ← c55852, ← c55853] at e_899_2
  have e_899_3 := arithEq_of_rows h (row := 899) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_3
  simp only [← c55854, ← c55855, k2, ← c55856] at e_899_3
  have hr := e_899_3
  simp only [e_899_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7070)) := by
  have c55857 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55858 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55859 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55860 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55861 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55862 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_4 := arithEq_of_rows h (row := 899) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_4
  simp only [← c55857, ← c55858, ← c55859] at e_899_4
  have e_899_5 := arithEq_of_rows h (row := 899) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_5
  simp only [← c55860, ← c55861, k2, ← c55862] at e_899_5
  have hr := e_899_5
  simp only [e_899_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7071)) := by
  have c55863 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55864 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55865 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55866 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55867 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55868 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_6 := arithEq_of_rows h (row := 899) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_6
  simp only [← c55863, ← c55864, ← c55865] at e_899_6
  have e_899_7 := arithEq_of_rows h (row := 899) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_7
  simp only [← c55866, ← c55867, k2, ← c55868] at e_899_7
  have hr := e_899_7
  simp only [e_899_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7072)) := by
  have c55869 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55870 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55871 := (publicBatchWrapper86_copies1745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55872 := (publicBatchWrapper86_copies1746 a h).1
  have c55873 := (publicBatchWrapper86_copies1746 a h).2.1
  have c55874 := (publicBatchWrapper86_copies1746 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_8 := arithEq_of_rows h (row := 899) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_8
  simp only [← c55869, ← c55870, ← c55871] at e_899_8
  have e_899_9 := arithEq_of_rows h (row := 899) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_9
  simp only [← c55872, ← c55873, k2, ← c55874] at e_899_9
  have hr := e_899_9
  simp only [e_899_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7073)) := by
  have c55875 := (publicBatchWrapper86_copies1746 a h).2.2.2.1
  have c55876 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.1
  have c55877 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.1
  have c55878 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.1
  have c55879 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.1
  have c55880 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_10 := arithEq_of_rows h (row := 899) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_10
  simp only [← c55875, ← c55876, ← c55877] at e_899_10
  have e_899_11 := arithEq_of_rows h (row := 899) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_11
  simp only [← c55878, ← c55879, k2, ← c55880] at e_899_11
  have hr := e_899_11
  simp only [e_899_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7074)) := by
  have c55881 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.1
  have c55882 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55883 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55884 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55885 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55886 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_12 := arithEq_of_rows h (row := 899) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_12
  simp only [← c55881, ← c55882, ← c55883] at e_899_12
  have e_899_13 := arithEq_of_rows h (row := 899) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_13
  simp only [← c55884, ← c55885, k2, ← c55886] at e_899_13
  have hr := e_899_13
  simp only [e_899_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7075)) := by
  have c55887 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55888 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55889 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55890 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55891 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55892 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_14 := arithEq_of_rows h (row := 899) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_14
  simp only [← c55887, ← c55888, ← c55889] at e_899_14
  have e_899_15 := arithEq_of_rows h (row := 899) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_15
  simp only [← c55890, ← c55891, k2, ← c55892] at e_899_15
  have hr := e_899_15
  simp only [e_899_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7076)) := by
  have c55893 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55894 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55895 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55896 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55897 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55898 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_16 := arithEq_of_rows h (row := 899) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_16
  simp only [← c55893, ← c55894, ← c55895] at e_899_16
  have e_899_17 := arithEq_of_rows h (row := 899) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_17
  simp only [← c55896, ← c55897, k2, ← c55898] at e_899_17
  have hr := e_899_17
  simp only [e_899_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 899 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7077)) := by
  have c55899 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55900 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55901 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55902 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55903 := (publicBatchWrapper86_copies1746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55904 := (publicBatchWrapper86_copies1747 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_899_18 := arithEq_of_rows h (row := 899) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_18
  simp only [← c55899, ← c55900, ← c55901] at e_899_18
  have e_899_19 := arithEq_of_rows h (row := 899) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_899_19
  simp only [← c55902, ← c55903, k2, ← c55904] at e_899_19
  have hr := e_899_19
  simp only [e_899_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7078)) := by
  have c55905 := (publicBatchWrapper86_copies1747 a h).2.1
  have c55906 := (publicBatchWrapper86_copies1747 a h).2.2.1
  have c55907 := (publicBatchWrapper86_copies1747 a h).2.2.2.1
  have c55908 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.1
  have c55909 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.1
  have c55910 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_0 := arithEq_of_rows h (row := 900) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_0
  simp only [← c55905, ← c55906, ← c55907] at e_900_0
  have e_900_1 := arithEq_of_rows h (row := 900) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_1
  simp only [← c55908, ← c55909, k2, ← c55910] at e_900_1
  have hr := e_900_1
  simp only [e_900_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7079)) := by
  have c55911 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.1
  have c55912 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.1
  have c55913 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.1
  have c55914 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55915 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55916 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_2 := arithEq_of_rows h (row := 900) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_2
  simp only [← c55911, ← c55912, ← c55913] at e_900_2
  have e_900_3 := arithEq_of_rows h (row := 900) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_3
  simp only [← c55914, ← c55915, k2, ← c55916] at e_900_3
  have hr := e_900_3
  simp only [e_900_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7080)) := by
  have c55917 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55918 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55919 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55920 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55921 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55922 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_4 := arithEq_of_rows h (row := 900) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_4
  simp only [← c55917, ← c55918, ← c55919] at e_900_4
  have e_900_5 := arithEq_of_rows h (row := 900) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_5
  simp only [← c55920, ← c55921, k2, ← c55922] at e_900_5
  have hr := e_900_5
  simp only [e_900_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7081)) := by
  have c55923 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55924 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55925 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55926 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55927 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55928 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_6 := arithEq_of_rows h (row := 900) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_6
  simp only [← c55923, ← c55924, ← c55925] at e_900_6
  have e_900_7 := arithEq_of_rows h (row := 900) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_7
  simp only [← c55926, ← c55927, k2, ← c55928] at e_900_7
  have hr := e_900_7
  simp only [e_900_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7082)) := by
  have c55929 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55930 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55931 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55932 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55933 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55934 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_8 := arithEq_of_rows h (row := 900) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_8
  simp only [← c55929, ← c55930, ← c55931] at e_900_8
  have e_900_9 := arithEq_of_rows h (row := 900) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_9
  simp only [← c55932, ← c55933, k2, ← c55934] at e_900_9
  have hr := e_900_9
  simp only [e_900_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7083)) := by
  have c55935 := (publicBatchWrapper86_copies1747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55936 := (publicBatchWrapper86_copies1748 a h).1
  have c55937 := (publicBatchWrapper86_copies1748 a h).2.1
  have c55938 := (publicBatchWrapper86_copies1748 a h).2.2.1
  have c55939 := (publicBatchWrapper86_copies1748 a h).2.2.2.1
  have c55940 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_10 := arithEq_of_rows h (row := 900) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_10
  simp only [← c55935, ← c55936, ← c55937] at e_900_10
  have e_900_11 := arithEq_of_rows h (row := 900) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_11
  simp only [← c55938, ← c55939, k2, ← c55940] at e_900_11
  have hr := e_900_11
  simp only [e_900_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7084)) := by
  have c55941 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.1
  have c55942 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.1
  have c55943 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.1
  have c55944 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.1
  have c55945 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.1
  have c55946 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_12 := arithEq_of_rows h (row := 900) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_12
  simp only [← c55941, ← c55942, ← c55943] at e_900_12
  have e_900_13 := arithEq_of_rows h (row := 900) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_13
  simp only [← c55944, ← c55945, k2, ← c55946] at e_900_13
  have hr := e_900_13
  simp only [e_900_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7085)) := by
  have c55947 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55948 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55949 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55950 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55951 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55952 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_14 := arithEq_of_rows h (row := 900) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_14
  simp only [← c55947, ← c55948, ← c55949] at e_900_14
  have e_900_15 := arithEq_of_rows h (row := 900) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_15
  simp only [← c55950, ← c55951, k2, ← c55952] at e_900_15
  have hr := e_900_15
  simp only [e_900_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7086)) := by
  have c55953 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55954 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55955 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55956 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55957 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55958 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_16 := arithEq_of_rows h (row := 900) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_16
  simp only [← c55953, ← c55954, ← c55955] at e_900_16
  have e_900_17 := arithEq_of_rows h (row := 900) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_17
  simp only [← c55956, ← c55957, k2, ← c55958] at e_900_17
  have hr := e_900_17
  simp only [e_900_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 900 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7087)) := by
  have c55959 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55960 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55961 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55962 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55963 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55964 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_900_18 := arithEq_of_rows h (row := 900) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_18
  simp only [← c55959, ← c55960, ← c55961] at e_900_18
  have e_900_19 := arithEq_of_rows h (row := 900) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_900_19
  simp only [← c55962, ← c55963, k2, ← c55964] at e_900_19
  have hr := e_900_19
  simp only [e_900_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7088)) := by
  have c55965 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55966 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55967 := (publicBatchWrapper86_copies1748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c55968 := (publicBatchWrapper86_copies1749 a h).1
  have c55969 := (publicBatchWrapper86_copies1749 a h).2.1
  have c55970 := (publicBatchWrapper86_copies1749 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_0 := arithEq_of_rows h (row := 901) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_0
  simp only [← c55965, ← c55966, ← c55967] at e_901_0
  have e_901_1 := arithEq_of_rows h (row := 901) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_1
  simp only [← c55968, ← c55969, k2, ← c55970] at e_901_1
  have hr := e_901_1
  simp only [e_901_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7089)) := by
  have c55971 := (publicBatchWrapper86_copies1749 a h).2.2.2.1
  have c55972 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.1
  have c55973 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.1
  have c55974 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.1
  have c55975 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.1
  have c55976 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_2 := arithEq_of_rows h (row := 901) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_2
  simp only [← c55971, ← c55972, ← c55973] at e_901_2
  have e_901_3 := arithEq_of_rows h (row := 901) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_3
  simp only [← c55974, ← c55975, k2, ← c55976] at e_901_3
  have hr := e_901_3
  simp only [e_901_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7090)) := by
  have c55977 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.1
  have c55978 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c55979 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c55980 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55981 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55982 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_4 := arithEq_of_rows h (row := 901) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_4
  simp only [← c55977, ← c55978, ← c55979] at e_901_4
  have e_901_5 := arithEq_of_rows h (row := 901) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_5
  simp only [← c55980, ← c55981, k2, ← c55982] at e_901_5
  have hr := e_901_5
  simp only [e_901_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7091)) := by
  have c55983 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55984 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55985 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55986 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55987 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55988 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_6 := arithEq_of_rows h (row := 901) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_6
  simp only [← c55983, ← c55984, ← c55985] at e_901_6
  have e_901_7 := arithEq_of_rows h (row := 901) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_7
  simp only [← c55986, ← c55987, k2, ← c55988] at e_901_7
  have hr := e_901_7
  simp only [e_901_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7092)) := by
  have c55989 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55990 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55991 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55992 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55993 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55994 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_8 := arithEq_of_rows h (row := 901) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_8
  simp only [← c55989, ← c55990, ← c55991] at e_901_8
  have e_901_9 := arithEq_of_rows h (row := 901) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_9
  simp only [← c55992, ← c55993, k2, ← c55994] at e_901_9
  have hr := e_901_9
  simp only [e_901_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7093)) := by
  have c55995 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55996 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55997 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55998 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c55999 := (publicBatchWrapper86_copies1749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56000 := (publicBatchWrapper86_copies1750 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_10 := arithEq_of_rows h (row := 901) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_10
  simp only [← c55995, ← c55996, ← c55997] at e_901_10
  have e_901_11 := arithEq_of_rows h (row := 901) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_11
  simp only [← c55998, ← c55999, k2, ← c56000] at e_901_11
  have hr := e_901_11
  simp only [e_901_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7094)) := by
  have c56001 := (publicBatchWrapper86_copies1750 a h).2.1
  have c56002 := (publicBatchWrapper86_copies1750 a h).2.2.1
  have c56003 := (publicBatchWrapper86_copies1750 a h).2.2.2.1
  have c56004 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.1
  have c56005 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.1
  have c56006 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_12 := arithEq_of_rows h (row := 901) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_12
  simp only [← c56001, ← c56002, ← c56003] at e_901_12
  have e_901_13 := arithEq_of_rows h (row := 901) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_13
  simp only [← c56004, ← c56005, k2, ← c56006] at e_901_13
  have hr := e_901_13
  simp only [e_901_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7095)) := by
  have c56007 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.1
  have c56008 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.1
  have c56009 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.1
  have c56010 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56011 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56012 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_14 := arithEq_of_rows h (row := 901) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_14
  simp only [← c56007, ← c56008, ← c56009] at e_901_14
  have e_901_15 := arithEq_of_rows h (row := 901) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_15
  simp only [← c56010, ← c56011, k2, ← c56012] at e_901_15
  have hr := e_901_15
  simp only [e_901_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7096)) := by
  have c56013 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56014 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56015 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56016 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56017 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56018 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_16 := arithEq_of_rows h (row := 901) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_16
  simp only [← c56013, ← c56014, ← c56015] at e_901_16
  have e_901_17 := arithEq_of_rows h (row := 901) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_17
  simp only [← c56016, ← c56017, k2, ← c56018] at e_901_17
  have hr := e_901_17
  simp only [e_901_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 901 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7097)) := by
  have c56019 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56020 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56021 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56022 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56023 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56024 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_901_18 := arithEq_of_rows h (row := 901) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_18
  simp only [← c56019, ← c56020, ← c56021] at e_901_18
  have e_901_19 := arithEq_of_rows h (row := 901) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_901_19
  simp only [← c56022, ← c56023, k2, ← c56024] at e_901_19
  have hr := e_901_19
  simp only [e_901_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7098)) := by
  have c56025 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56026 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56027 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56028 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56029 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56030 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_0 := arithEq_of_rows h (row := 902) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_0
  simp only [← c56025, ← c56026, ← c56027] at e_902_0
  have e_902_1 := arithEq_of_rows h (row := 902) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_1
  simp only [← c56028, ← c56029, k2, ← c56030] at e_902_1
  have hr := e_902_1
  simp only [e_902_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7099)) := by
  have c56031 := (publicBatchWrapper86_copies1750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56032 := (publicBatchWrapper86_copies1751 a h).1
  have c56033 := (publicBatchWrapper86_copies1751 a h).2.1
  have c56034 := (publicBatchWrapper86_copies1751 a h).2.2.1
  have c56035 := (publicBatchWrapper86_copies1751 a h).2.2.2.1
  have c56036 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_2 := arithEq_of_rows h (row := 902) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_2
  simp only [← c56031, ← c56032, ← c56033] at e_902_2
  have e_902_3 := arithEq_of_rows h (row := 902) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_3
  simp only [← c56034, ← c56035, k2, ← c56036] at e_902_3
  have hr := e_902_3
  simp only [e_902_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7100)) := by
  have c56037 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.1
  have c56038 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.1
  have c56039 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.1
  have c56040 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.1
  have c56041 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.1
  have c56042 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_4 := arithEq_of_rows h (row := 902) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_4
  simp only [← c56037, ← c56038, ← c56039] at e_902_4
  have e_902_5 := arithEq_of_rows h (row := 902) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_5
  simp only [← c56040, ← c56041, k2, ← c56042] at e_902_5
  have hr := e_902_5
  simp only [e_902_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7101)) := by
  have c56043 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56044 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56045 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56046 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56047 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56048 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_6 := arithEq_of_rows h (row := 902) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_6
  simp only [← c56043, ← c56044, ← c56045] at e_902_6
  have e_902_7 := arithEq_of_rows h (row := 902) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_7
  simp only [← c56046, ← c56047, k2, ← c56048] at e_902_7
  have hr := e_902_7
  simp only [e_902_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7102)) := by
  have c56049 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56050 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56051 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56052 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56053 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56054 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_8 := arithEq_of_rows h (row := 902) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_8
  simp only [← c56049, ← c56050, ← c56051] at e_902_8
  have e_902_9 := arithEq_of_rows h (row := 902) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_9
  simp only [← c56052, ← c56053, k2, ← c56054] at e_902_9
  have hr := e_902_9
  simp only [e_902_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7103)) := by
  have c56055 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56056 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56057 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56058 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56059 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56060 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_10 := arithEq_of_rows h (row := 902) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_10
  simp only [← c56055, ← c56056, ← c56057] at e_902_10
  have e_902_11 := arithEq_of_rows h (row := 902) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_11
  simp only [← c56058, ← c56059, k2, ← c56060] at e_902_11
  have hr := e_902_11
  simp only [e_902_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7104)) := by
  have c56061 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56062 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56063 := (publicBatchWrapper86_copies1751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56064 := (publicBatchWrapper86_copies1752 a h).1
  have c56065 := (publicBatchWrapper86_copies1752 a h).2.1
  have c56066 := (publicBatchWrapper86_copies1752 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_12 := arithEq_of_rows h (row := 902) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_12
  simp only [← c56061, ← c56062, ← c56063] at e_902_12
  have e_902_13 := arithEq_of_rows h (row := 902) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_13
  simp only [← c56064, ← c56065, k2, ← c56066] at e_902_13
  have hr := e_902_13
  simp only [e_902_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7105)) := by
  have c56067 := (publicBatchWrapper86_copies1752 a h).2.2.2.1
  have c56068 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.1
  have c56069 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.1
  have c56070 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.1
  have c56071 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.1
  have c56072 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_14 := arithEq_of_rows h (row := 902) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_14
  simp only [← c56067, ← c56068, ← c56069] at e_902_14
  have e_902_15 := arithEq_of_rows h (row := 902) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_15
  simp only [← c56070, ← c56071, k2, ← c56072] at e_902_15
  have hr := e_902_15
  simp only [e_902_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7106)) := by
  have c56073 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.1
  have c56074 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56075 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56076 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56077 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56078 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_16 := arithEq_of_rows h (row := 902) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_16
  simp only [← c56073, ← c56074, ← c56075] at e_902_16
  have e_902_17 := arithEq_of_rows h (row := 902) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_17
  simp only [← c56076, ← c56077, k2, ← c56078] at e_902_17
  have hr := e_902_17
  simp only [e_902_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 902 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7107)) := by
  have c56079 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56080 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56081 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56082 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56083 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56084 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_902_18 := arithEq_of_rows h (row := 902) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_18
  simp only [← c56079, ← c56080, ← c56081] at e_902_18
  have e_902_19 := arithEq_of_rows h (row := 902) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_902_19
  simp only [← c56082, ← c56083, k2, ← c56084] at e_902_19
  have hr := e_902_19
  simp only [e_902_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7108)) := by
  have c56085 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56086 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56087 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56088 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56089 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56090 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_0 := arithEq_of_rows h (row := 903) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_0
  simp only [← c56085, ← c56086, ← c56087] at e_903_0
  have e_903_1 := arithEq_of_rows h (row := 903) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_1
  simp only [← c56088, ← c56089, k2, ← c56090] at e_903_1
  have hr := e_903_1
  simp only [e_903_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7109)) := by
  have c56091 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56092 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56093 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56094 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56095 := (publicBatchWrapper86_copies1752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56096 := (publicBatchWrapper86_copies1753 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_2 := arithEq_of_rows h (row := 903) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_2
  simp only [← c56091, ← c56092, ← c56093] at e_903_2
  have e_903_3 := arithEq_of_rows h (row := 903) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_3
  simp only [← c56094, ← c56095, k2, ← c56096] at e_903_3
  have hr := e_903_3
  simp only [e_903_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7110)) := by
  have c56097 := (publicBatchWrapper86_copies1753 a h).2.1
  have c56098 := (publicBatchWrapper86_copies1753 a h).2.2.1
  have c56099 := (publicBatchWrapper86_copies1753 a h).2.2.2.1
  have c56100 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.1
  have c56101 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.1
  have c56102 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_4 := arithEq_of_rows h (row := 903) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_4
  simp only [← c56097, ← c56098, ← c56099] at e_903_4
  have e_903_5 := arithEq_of_rows h (row := 903) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_5
  simp only [← c56100, ← c56101, k2, ← c56102] at e_903_5
  have hr := e_903_5
  simp only [e_903_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7111)) := by
  have c56103 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.1
  have c56104 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.1
  have c56105 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.1
  have c56106 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56107 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56108 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_6 := arithEq_of_rows h (row := 903) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_6
  simp only [← c56103, ← c56104, ← c56105] at e_903_6
  have e_903_7 := arithEq_of_rows h (row := 903) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_7
  simp only [← c56106, ← c56107, k2, ← c56108] at e_903_7
  have hr := e_903_7
  simp only [e_903_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7112)) := by
  have c56109 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56110 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56111 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56112 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56113 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56114 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_8 := arithEq_of_rows h (row := 903) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_8
  simp only [← c56109, ← c56110, ← c56111] at e_903_8
  have e_903_9 := arithEq_of_rows h (row := 903) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_9
  simp only [← c56112, ← c56113, k2, ← c56114] at e_903_9
  have hr := e_903_9
  simp only [e_903_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7113)) := by
  have c56115 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56116 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56117 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56118 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56119 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56120 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_10 := arithEq_of_rows h (row := 903) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_10
  simp only [← c56115, ← c56116, ← c56117] at e_903_10
  have e_903_11 := arithEq_of_rows h (row := 903) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_11
  simp only [← c56118, ← c56119, k2, ← c56120] at e_903_11
  have hr := e_903_11
  simp only [e_903_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7114)) := by
  have c56121 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56122 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56123 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56124 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56125 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56126 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_12 := arithEq_of_rows h (row := 903) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_12
  simp only [← c56121, ← c56122, ← c56123] at e_903_12
  have e_903_13 := arithEq_of_rows h (row := 903) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_13
  simp only [← c56124, ← c56125, k2, ← c56126] at e_903_13
  have hr := e_903_13
  simp only [e_903_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7115)) := by
  have c56127 := (publicBatchWrapper86_copies1753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56128 := (publicBatchWrapper86_copies1754 a h).1
  have c56129 := (publicBatchWrapper86_copies1754 a h).2.1
  have c56130 := (publicBatchWrapper86_copies1754 a h).2.2.1
  have c56131 := (publicBatchWrapper86_copies1754 a h).2.2.2.1
  have c56132 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_14 := arithEq_of_rows h (row := 903) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_14
  simp only [← c56127, ← c56128, ← c56129] at e_903_14
  have e_903_15 := arithEq_of_rows h (row := 903) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_15
  simp only [← c56130, ← c56131, k2, ← c56132] at e_903_15
  have hr := e_903_15
  simp only [e_903_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7116)) := by
  have c56133 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.1
  have c56134 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.1
  have c56135 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.1
  have c56136 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.1
  have c56137 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.1
  have c56138 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_16 := arithEq_of_rows h (row := 903) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_16
  simp only [← c56133, ← c56134, ← c56135] at e_903_16
  have e_903_17 := arithEq_of_rows h (row := 903) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_17
  simp only [← c56136, ← c56137, k2, ← c56138] at e_903_17
  have hr := e_903_17
  simp only [e_903_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 903 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7117)) := by
  have c56139 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56140 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56141 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56142 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56143 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56144 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_903_18 := arithEq_of_rows h (row := 903) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_18
  simp only [← c56139, ← c56140, ← c56141] at e_903_18
  have e_903_19 := arithEq_of_rows h (row := 903) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_903_19
  simp only [← c56142, ← c56143, k2, ← c56144] at e_903_19
  have hr := e_903_19
  simp only [e_903_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7118)) := by
  have c56145 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56146 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56147 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56148 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56149 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56150 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_0 := arithEq_of_rows h (row := 904) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_0
  simp only [← c56145, ← c56146, ← c56147] at e_904_0
  have e_904_1 := arithEq_of_rows h (row := 904) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_1
  simp only [← c56148, ← c56149, k2, ← c56150] at e_904_1
  have hr := e_904_1
  simp only [e_904_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7119)) := by
  have c56151 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56152 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56153 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56154 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56155 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56156 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_2 := arithEq_of_rows h (row := 904) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_2
  simp only [← c56151, ← c56152, ← c56153] at e_904_2
  have e_904_3 := arithEq_of_rows h (row := 904) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_3
  simp only [← c56154, ← c56155, k2, ← c56156] at e_904_3
  have hr := e_904_3
  simp only [e_904_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7120)) := by
  have c56157 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56158 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56159 := (publicBatchWrapper86_copies1754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56160 := (publicBatchWrapper86_copies1755 a h).1
  have c56161 := (publicBatchWrapper86_copies1755 a h).2.1
  have c56162 := (publicBatchWrapper86_copies1755 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_4 := arithEq_of_rows h (row := 904) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_4
  simp only [← c56157, ← c56158, ← c56159] at e_904_4
  have e_904_5 := arithEq_of_rows h (row := 904) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_5
  simp only [← c56160, ← c56161, k2, ← c56162] at e_904_5
  have hr := e_904_5
  simp only [e_904_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7121)) := by
  have c56163 := (publicBatchWrapper86_copies1755 a h).2.2.2.1
  have c56164 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.1
  have c56165 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.1
  have c56166 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.1
  have c56167 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.1
  have c56168 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_6 := arithEq_of_rows h (row := 904) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_6
  simp only [← c56163, ← c56164, ← c56165] at e_904_6
  have e_904_7 := arithEq_of_rows h (row := 904) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_7
  simp only [← c56166, ← c56167, k2, ← c56168] at e_904_7
  have hr := e_904_7
  simp only [e_904_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7122)) := by
  have c56169 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.1
  have c56170 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56171 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56172 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56173 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56174 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_8 := arithEq_of_rows h (row := 904) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_8
  simp only [← c56169, ← c56170, ← c56171] at e_904_8
  have e_904_9 := arithEq_of_rows h (row := 904) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_9
  simp only [← c56172, ← c56173, k2, ← c56174] at e_904_9
  have hr := e_904_9
  simp only [e_904_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7123)) := by
  have c56175 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56176 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56177 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56178 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56179 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56180 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_10 := arithEq_of_rows h (row := 904) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_10
  simp only [← c56175, ← c56176, ← c56177] at e_904_10
  have e_904_11 := arithEq_of_rows h (row := 904) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_11
  simp only [← c56178, ← c56179, k2, ← c56180] at e_904_11
  have hr := e_904_11
  simp only [e_904_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7124)) := by
  have c56181 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56182 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56183 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56184 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56185 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56186 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_12 := arithEq_of_rows h (row := 904) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_12
  simp only [← c56181, ← c56182, ← c56183] at e_904_12
  have e_904_13 := arithEq_of_rows h (row := 904) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_13
  simp only [← c56184, ← c56185, k2, ← c56186] at e_904_13
  have hr := e_904_13
  simp only [e_904_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7125)) := by
  have c56187 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56188 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56189 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56190 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56191 := (publicBatchWrapper86_copies1755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56192 := (publicBatchWrapper86_copies1756 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_14 := arithEq_of_rows h (row := 904) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_14
  simp only [← c56187, ← c56188, ← c56189] at e_904_14
  have e_904_15 := arithEq_of_rows h (row := 904) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_15
  simp only [← c56190, ← c56191, k2, ← c56192] at e_904_15
  have hr := e_904_15
  simp only [e_904_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7126)) := by
  have c56193 := (publicBatchWrapper86_copies1756 a h).2.1
  have c56194 := (publicBatchWrapper86_copies1756 a h).2.2.1
  have c56195 := (publicBatchWrapper86_copies1756 a h).2.2.2.1
  have c56196 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.1
  have c56197 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.1
  have c56198 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_16 := arithEq_of_rows h (row := 904) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_16
  simp only [← c56193, ← c56194, ← c56195] at e_904_16
  have e_904_17 := arithEq_of_rows h (row := 904) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_17
  simp only [← c56196, ← c56197, k2, ← c56198] at e_904_17
  have hr := e_904_17
  simp only [e_904_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 904 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7127)) := by
  have c56199 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.1
  have c56200 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.1
  have c56201 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.1
  have c56202 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56203 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56204 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_904_18 := arithEq_of_rows h (row := 904) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_18
  simp only [← c56199, ← c56200, ← c56201] at e_904_18
  have e_904_19 := arithEq_of_rows h (row := 904) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_904_19
  simp only [← c56202, ← c56203, k2, ← c56204] at e_904_19
  have hr := e_904_19
  simp only [e_904_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7128)) := by
  have c56205 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56206 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56207 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56208 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56209 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56210 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_0 := arithEq_of_rows h (row := 905) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_0
  simp only [← c56205, ← c56206, ← c56207] at e_905_0
  have e_905_1 := arithEq_of_rows h (row := 905) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_1
  simp only [← c56208, ← c56209, k2, ← c56210] at e_905_1
  have hr := e_905_1
  simp only [e_905_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7129)) := by
  have c56211 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56212 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56213 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56214 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56215 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56216 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_2 := arithEq_of_rows h (row := 905) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_2
  simp only [← c56211, ← c56212, ← c56213] at e_905_2
  have e_905_3 := arithEq_of_rows h (row := 905) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_3
  simp only [← c56214, ← c56215, k2, ← c56216] at e_905_3
  have hr := e_905_3
  simp only [e_905_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7130)) := by
  have c56217 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56218 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56219 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56220 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56221 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56222 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_4 := arithEq_of_rows h (row := 905) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_4
  simp only [← c56217, ← c56218, ← c56219] at e_905_4
  have e_905_5 := arithEq_of_rows h (row := 905) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_5
  simp only [← c56220, ← c56221, k2, ← c56222] at e_905_5
  have hr := e_905_5
  simp only [e_905_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7131)) := by
  have c56223 := (publicBatchWrapper86_copies1756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56224 := (publicBatchWrapper86_copies1757 a h).1
  have c56225 := (publicBatchWrapper86_copies1757 a h).2.1
  have c56226 := (publicBatchWrapper86_copies1757 a h).2.2.1
  have c56227 := (publicBatchWrapper86_copies1757 a h).2.2.2.1
  have c56228 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_6 := arithEq_of_rows h (row := 905) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_6
  simp only [← c56223, ← c56224, ← c56225] at e_905_6
  have e_905_7 := arithEq_of_rows h (row := 905) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_7
  simp only [← c56226, ← c56227, k2, ← c56228] at e_905_7
  have hr := e_905_7
  simp only [e_905_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7132)) := by
  have c56229 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.1
  have c56230 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.1
  have c56231 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.1
  have c56232 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.1
  have c56233 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.1
  have c56234 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_8 := arithEq_of_rows h (row := 905) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_8
  simp only [← c56229, ← c56230, ← c56231] at e_905_8
  have e_905_9 := arithEq_of_rows h (row := 905) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_9
  simp only [← c56232, ← c56233, k2, ← c56234] at e_905_9
  have hr := e_905_9
  simp only [e_905_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7133)) := by
  have c56235 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56236 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56237 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56238 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56239 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56240 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_10 := arithEq_of_rows h (row := 905) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_10
  simp only [← c56235, ← c56236, ← c56237] at e_905_10
  have e_905_11 := arithEq_of_rows h (row := 905) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_11
  simp only [← c56238, ← c56239, k2, ← c56240] at e_905_11
  have hr := e_905_11
  simp only [e_905_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7134)) := by
  have c56241 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56242 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56243 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56244 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56245 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56246 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_12 := arithEq_of_rows h (row := 905) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_12
  simp only [← c56241, ← c56242, ← c56243] at e_905_12
  have e_905_13 := arithEq_of_rows h (row := 905) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_13
  simp only [← c56244, ← c56245, k2, ← c56246] at e_905_13
  have hr := e_905_13
  simp only [e_905_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7135)) := by
  have c56247 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56248 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56249 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56250 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56251 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56252 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_14 := arithEq_of_rows h (row := 905) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_14
  simp only [← c56247, ← c56248, ← c56249] at e_905_14
  have e_905_15 := arithEq_of_rows h (row := 905) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_15
  simp only [← c56250, ← c56251, k2, ← c56252] at e_905_15
  have hr := e_905_15
  simp only [e_905_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7136)) := by
  have c56253 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56254 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56255 := (publicBatchWrapper86_copies1757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56256 := (publicBatchWrapper86_copies1758 a h).1
  have c56257 := (publicBatchWrapper86_copies1758 a h).2.1
  have c56258 := (publicBatchWrapper86_copies1758 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_16 := arithEq_of_rows h (row := 905) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_16
  simp only [← c56253, ← c56254, ← c56255] at e_905_16
  have e_905_17 := arithEq_of_rows h (row := 905) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_17
  simp only [← c56256, ← c56257, k2, ← c56258] at e_905_17
  have hr := e_905_17
  simp only [e_905_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 905 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7137)) := by
  have c56259 := (publicBatchWrapper86_copies1758 a h).2.2.2.1
  have c56260 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.1
  have c56261 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.1
  have c56262 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.1
  have c56263 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.1
  have c56264 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_905_18 := arithEq_of_rows h (row := 905) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_18
  simp only [← c56259, ← c56260, ← c56261] at e_905_18
  have e_905_19 := arithEq_of_rows h (row := 905) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_905_19
  simp only [← c56262, ← c56263, k2, ← c56264] at e_905_19
  have hr := e_905_19
  simp only [e_905_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7138)) := by
  have c56265 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.1
  have c56266 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56267 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56268 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56269 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56270 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_0 := arithEq_of_rows h (row := 906) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_0
  simp only [← c56265, ← c56266, ← c56267] at e_906_0
  have e_906_1 := arithEq_of_rows h (row := 906) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_1
  simp only [← c56268, ← c56269, k2, ← c56270] at e_906_1
  have hr := e_906_1
  simp only [e_906_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7139)) := by
  have c56271 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56272 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56273 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56274 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56275 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56276 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_2 := arithEq_of_rows h (row := 906) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_2
  simp only [← c56271, ← c56272, ← c56273] at e_906_2
  have e_906_3 := arithEq_of_rows h (row := 906) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_3
  simp only [← c56274, ← c56275, k2, ← c56276] at e_906_3
  have hr := e_906_3
  simp only [e_906_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7140)) := by
  have c56277 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56278 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56279 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56280 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56281 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56282 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_4 := arithEq_of_rows h (row := 906) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_4
  simp only [← c56277, ← c56278, ← c56279] at e_906_4
  have e_906_5 := arithEq_of_rows h (row := 906) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_5
  simp only [← c56280, ← c56281, k2, ← c56282] at e_906_5
  have hr := e_906_5
  simp only [e_906_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7141)) := by
  have c56283 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56284 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56285 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56286 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56287 := (publicBatchWrapper86_copies1758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56288 := (publicBatchWrapper86_copies1759 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_6 := arithEq_of_rows h (row := 906) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_6
  simp only [← c56283, ← c56284, ← c56285] at e_906_6
  have e_906_7 := arithEq_of_rows h (row := 906) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_7
  simp only [← c56286, ← c56287, k2, ← c56288] at e_906_7
  have hr := e_906_7
  simp only [e_906_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7142)) := by
  have c56289 := (publicBatchWrapper86_copies1759 a h).2.1
  have c56290 := (publicBatchWrapper86_copies1759 a h).2.2.1
  have c56291 := (publicBatchWrapper86_copies1759 a h).2.2.2.1
  have c56292 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.1
  have c56293 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.1
  have c56294 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_8 := arithEq_of_rows h (row := 906) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_8
  simp only [← c56289, ← c56290, ← c56291] at e_906_8
  have e_906_9 := arithEq_of_rows h (row := 906) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_9
  simp only [← c56292, ← c56293, k2, ← c56294] at e_906_9
  have hr := e_906_9
  simp only [e_906_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7143)) := by
  have c56295 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.1
  have c56296 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.1
  have c56297 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.1
  have c56298 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56299 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56300 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_10 := arithEq_of_rows h (row := 906) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_10
  simp only [← c56295, ← c56296, ← c56297] at e_906_10
  have e_906_11 := arithEq_of_rows h (row := 906) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_11
  simp only [← c56298, ← c56299, k2, ← c56300] at e_906_11
  have hr := e_906_11
  simp only [e_906_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7144)) := by
  have c56301 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56302 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56303 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56304 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56305 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56306 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_12 := arithEq_of_rows h (row := 906) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_12
  simp only [← c56301, ← c56302, ← c56303] at e_906_12
  have e_906_13 := arithEq_of_rows h (row := 906) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_13
  simp only [← c56304, ← c56305, k2, ← c56306] at e_906_13
  have hr := e_906_13
  simp only [e_906_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7145)) := by
  have c56307 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56308 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56309 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56310 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56311 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56312 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_14 := arithEq_of_rows h (row := 906) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_14
  simp only [← c56307, ← c56308, ← c56309] at e_906_14
  have e_906_15 := arithEq_of_rows h (row := 906) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_15
  simp only [← c56310, ← c56311, k2, ← c56312] at e_906_15
  have hr := e_906_15
  simp only [e_906_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7146)) := by
  have c56313 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56314 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56315 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56316 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56317 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56318 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_16 := arithEq_of_rows h (row := 906) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_16
  simp only [← c56313, ← c56314, ← c56315] at e_906_16
  have e_906_17 := arithEq_of_rows h (row := 906) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_17
  simp only [← c56316, ← c56317, k2, ← c56318] at e_906_17
  have hr := e_906_17
  simp only [e_906_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 906 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7147)) := by
  have c56319 := (publicBatchWrapper86_copies1759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56320 := (publicBatchWrapper86_copies1760 a h).1
  have c56321 := (publicBatchWrapper86_copies1760 a h).2.1
  have c56322 := (publicBatchWrapper86_copies1760 a h).2.2.1
  have c56323 := (publicBatchWrapper86_copies1760 a h).2.2.2.1
  have c56324 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_906_18 := arithEq_of_rows h (row := 906) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_18
  simp only [← c56319, ← c56320, ← c56321] at e_906_18
  have e_906_19 := arithEq_of_rows h (row := 906) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_906_19
  simp only [← c56322, ← c56323, k2, ← c56324] at e_906_19
  have hr := e_906_19
  simp only [e_906_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7148)) := by
  have c56325 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.1
  have c56326 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.1
  have c56327 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.1
  have c56328 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.1
  have c56329 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.1
  have c56330 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_0 := arithEq_of_rows h (row := 907) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_0
  simp only [← c56325, ← c56326, ← c56327] at e_907_0
  have e_907_1 := arithEq_of_rows h (row := 907) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_1
  simp only [← c56328, ← c56329, k2, ← c56330] at e_907_1
  have hr := e_907_1
  simp only [e_907_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7149)) := by
  have c56331 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56332 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56333 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56334 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56335 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56336 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_2 := arithEq_of_rows h (row := 907) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_2
  simp only [← c56331, ← c56332, ← c56333] at e_907_2
  have e_907_3 := arithEq_of_rows h (row := 907) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_3
  simp only [← c56334, ← c56335, k2, ← c56336] at e_907_3
  have hr := e_907_3
  simp only [e_907_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7150)) := by
  have c56337 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56338 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56339 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56340 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56341 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56342 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_4 := arithEq_of_rows h (row := 907) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_4
  simp only [← c56337, ← c56338, ← c56339] at e_907_4
  have e_907_5 := arithEq_of_rows h (row := 907) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_5
  simp only [← c56340, ← c56341, k2, ← c56342] at e_907_5
  have hr := e_907_5
  simp only [e_907_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7151)) := by
  have c56343 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56344 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56345 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56346 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56347 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56348 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_6 := arithEq_of_rows h (row := 907) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_6
  simp only [← c56343, ← c56344, ← c56345] at e_907_6
  have e_907_7 := arithEq_of_rows h (row := 907) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_7
  simp only [← c56346, ← c56347, k2, ← c56348] at e_907_7
  have hr := e_907_7
  simp only [e_907_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7152)) := by
  have c56349 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56350 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56351 := (publicBatchWrapper86_copies1760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56352 := (publicBatchWrapper86_copies1761 a h).1
  have c56353 := (publicBatchWrapper86_copies1761 a h).2.1
  have c56354 := (publicBatchWrapper86_copies1761 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_8 := arithEq_of_rows h (row := 907) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_8
  simp only [← c56349, ← c56350, ← c56351] at e_907_8
  have e_907_9 := arithEq_of_rows h (row := 907) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_9
  simp only [← c56352, ← c56353, k2, ← c56354] at e_907_9
  have hr := e_907_9
  simp only [e_907_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7153)) := by
  have c56355 := (publicBatchWrapper86_copies1761 a h).2.2.2.1
  have c56356 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.1
  have c56357 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.1
  have c56358 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.1
  have c56359 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.1
  have c56360 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_10 := arithEq_of_rows h (row := 907) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_10
  simp only [← c56355, ← c56356, ← c56357] at e_907_10
  have e_907_11 := arithEq_of_rows h (row := 907) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_11
  simp only [← c56358, ← c56359, k2, ← c56360] at e_907_11
  have hr := e_907_11
  simp only [e_907_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7154)) := by
  have c56361 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.1
  have c56362 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56363 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56364 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56365 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56366 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_12 := arithEq_of_rows h (row := 907) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_12
  simp only [← c56361, ← c56362, ← c56363] at e_907_12
  have e_907_13 := arithEq_of_rows h (row := 907) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_13
  simp only [← c56364, ← c56365, k2, ← c56366] at e_907_13
  have hr := e_907_13
  simp only [e_907_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7155)) := by
  have c56367 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56368 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56369 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56370 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56371 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56372 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_14 := arithEq_of_rows h (row := 907) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_14
  simp only [← c56367, ← c56368, ← c56369] at e_907_14
  have e_907_15 := arithEq_of_rows h (row := 907) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_15
  simp only [← c56370, ← c56371, k2, ← c56372] at e_907_15
  have hr := e_907_15
  simp only [e_907_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7156)) := by
  have c56373 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56374 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56375 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56376 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56377 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56378 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_16 := arithEq_of_rows h (row := 907) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_16
  simp only [← c56373, ← c56374, ← c56375] at e_907_16
  have e_907_17 := arithEq_of_rows h (row := 907) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_17
  simp only [← c56376, ← c56377, k2, ← c56378] at e_907_17
  have hr := e_907_17
  simp only [e_907_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 907 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7157)) := by
  have c56379 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56380 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56381 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56382 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56383 := (publicBatchWrapper86_copies1761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56384 := (publicBatchWrapper86_copies1762 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_907_18 := arithEq_of_rows h (row := 907) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_18
  simp only [← c56379, ← c56380, ← c56381] at e_907_18
  have e_907_19 := arithEq_of_rows h (row := 907) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_907_19
  simp only [← c56382, ← c56383, k2, ← c56384] at e_907_19
  have hr := e_907_19
  simp only [e_907_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7158)) := by
  have c56385 := (publicBatchWrapper86_copies1762 a h).2.1
  have c56386 := (publicBatchWrapper86_copies1762 a h).2.2.1
  have c56387 := (publicBatchWrapper86_copies1762 a h).2.2.2.1
  have c56388 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.1
  have c56389 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.1
  have c56390 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_0 := arithEq_of_rows h (row := 908) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_0
  simp only [← c56385, ← c56386, ← c56387] at e_908_0
  have e_908_1 := arithEq_of_rows h (row := 908) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_1
  simp only [← c56388, ← c56389, k2, ← c56390] at e_908_1
  have hr := e_908_1
  simp only [e_908_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7159)) := by
  have c56391 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.1
  have c56392 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.1
  have c56393 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.1
  have c56394 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56395 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56396 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_2 := arithEq_of_rows h (row := 908) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_2
  simp only [← c56391, ← c56392, ← c56393] at e_908_2
  have e_908_3 := arithEq_of_rows h (row := 908) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_3
  simp only [← c56394, ← c56395, k2, ← c56396] at e_908_3
  have hr := e_908_3
  simp only [e_908_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7160)) := by
  have c56397 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56398 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56399 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56400 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56401 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56402 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_4 := arithEq_of_rows h (row := 908) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_4
  simp only [← c56397, ← c56398, ← c56399] at e_908_4
  have e_908_5 := arithEq_of_rows h (row := 908) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_5
  simp only [← c56400, ← c56401, k2, ← c56402] at e_908_5
  have hr := e_908_5
  simp only [e_908_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7161)) := by
  have c56403 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56404 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56405 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56406 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56407 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56408 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_6 := arithEq_of_rows h (row := 908) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_6
  simp only [← c56403, ← c56404, ← c56405] at e_908_6
  have e_908_7 := arithEq_of_rows h (row := 908) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_7
  simp only [← c56406, ← c56407, k2, ← c56408] at e_908_7
  have hr := e_908_7
  simp only [e_908_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7162)) := by
  have c56409 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56410 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56411 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56412 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56413 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56414 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_8 := arithEq_of_rows h (row := 908) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_8
  simp only [← c56409, ← c56410, ← c56411] at e_908_8
  have e_908_9 := arithEq_of_rows h (row := 908) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_9
  simp only [← c56412, ← c56413, k2, ← c56414] at e_908_9
  have hr := e_908_9
  simp only [e_908_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7163)) := by
  have c56415 := (publicBatchWrapper86_copies1762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56416 := (publicBatchWrapper86_copies1763 a h).1
  have c56417 := (publicBatchWrapper86_copies1763 a h).2.1
  have c56418 := (publicBatchWrapper86_copies1763 a h).2.2.1
  have c56419 := (publicBatchWrapper86_copies1763 a h).2.2.2.1
  have c56420 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_10 := arithEq_of_rows h (row := 908) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_10
  simp only [← c56415, ← c56416, ← c56417] at e_908_10
  have e_908_11 := arithEq_of_rows h (row := 908) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_11
  simp only [← c56418, ← c56419, k2, ← c56420] at e_908_11
  have hr := e_908_11
  simp only [e_908_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7164)) := by
  have c56421 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.1
  have c56422 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.1
  have c56423 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.1
  have c56424 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.1
  have c56425 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.1
  have c56426 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_12 := arithEq_of_rows h (row := 908) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_12
  simp only [← c56421, ← c56422, ← c56423] at e_908_12
  have e_908_13 := arithEq_of_rows h (row := 908) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_13
  simp only [← c56424, ← c56425, k2, ← c56426] at e_908_13
  have hr := e_908_13
  simp only [e_908_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7165)) := by
  have c56427 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56428 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56429 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56430 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56431 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56432 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_14 := arithEq_of_rows h (row := 908) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_14
  simp only [← c56427, ← c56428, ← c56429] at e_908_14
  have e_908_15 := arithEq_of_rows h (row := 908) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_15
  simp only [← c56430, ← c56431, k2, ← c56432] at e_908_15
  have hr := e_908_15
  simp only [e_908_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7166)) := by
  have c56433 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56434 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56435 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56436 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56437 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56438 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_16 := arithEq_of_rows h (row := 908) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_16
  simp only [← c56433, ← c56434, ← c56435] at e_908_16
  have e_908_17 := arithEq_of_rows h (row := 908) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_17
  simp only [← c56436, ← c56437, k2, ← c56438] at e_908_17
  have hr := e_908_17
  simp only [e_908_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 908 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7167)) := by
  have c56439 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56440 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56441 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56442 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56443 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56444 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_908_18 := arithEq_of_rows h (row := 908) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_18
  simp only [← c56439, ← c56440, ← c56441] at e_908_18
  have e_908_19 := arithEq_of_rows h (row := 908) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_908_19
  simp only [← c56442, ← c56443, k2, ← c56444] at e_908_19
  have hr := e_908_19
  simp only [e_908_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7168)) := by
  have c56445 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56446 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56447 := (publicBatchWrapper86_copies1763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56448 := (publicBatchWrapper86_copies1764 a h).1
  have c56449 := (publicBatchWrapper86_copies1764 a h).2.1
  have c56450 := (publicBatchWrapper86_copies1764 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_0 := arithEq_of_rows h (row := 909) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_0
  simp only [← c56445, ← c56446, ← c56447] at e_909_0
  have e_909_1 := arithEq_of_rows h (row := 909) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_1
  simp only [← c56448, ← c56449, k2, ← c56450] at e_909_1
  have hr := e_909_1
  simp only [e_909_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7169)) := by
  have c56451 := (publicBatchWrapper86_copies1764 a h).2.2.2.1
  have c56452 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.1
  have c56453 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.1
  have c56454 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.1
  have c56455 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.1
  have c56456 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_2 := arithEq_of_rows h (row := 909) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_2
  simp only [← c56451, ← c56452, ← c56453] at e_909_2
  have e_909_3 := arithEq_of_rows h (row := 909) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_3
  simp only [← c56454, ← c56455, k2, ← c56456] at e_909_3
  have hr := e_909_3
  simp only [e_909_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7210)) := by
  have c56457 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.1
  have c56458 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56459 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56460 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56461 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56462 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_4 := arithEq_of_rows h (row := 909) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_4
  simp only [← c56457, ← c56458, ← c56459] at e_909_4
  have e_909_5 := arithEq_of_rows h (row := 909) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_5
  simp only [← c56460, ← c56461, k2, ← c56462] at e_909_5
  have hr := e_909_5
  simp only [e_909_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7211)) := by
  have c56463 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56464 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56465 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56466 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56467 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56468 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_6 := arithEq_of_rows h (row := 909) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_6
  simp only [← c56463, ← c56464, ← c56465] at e_909_6
  have e_909_7 := arithEq_of_rows h (row := 909) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_7
  simp only [← c56466, ← c56467, k2, ← c56468] at e_909_7
  have hr := e_909_7
  simp only [e_909_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7212)) := by
  have c56469 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56470 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56471 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56472 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56473 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56474 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_8 := arithEq_of_rows h (row := 909) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_8
  simp only [← c56469, ← c56470, ← c56471] at e_909_8
  have e_909_9 := arithEq_of_rows h (row := 909) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_9
  simp only [← c56472, ← c56473, k2, ← c56474] at e_909_9
  have hr := e_909_9
  simp only [e_909_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7213)) := by
  have c56475 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56476 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56477 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56478 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56479 := (publicBatchWrapper86_copies1764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56480 := (publicBatchWrapper86_copies1765 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_10 := arithEq_of_rows h (row := 909) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_10
  simp only [← c56475, ← c56476, ← c56477] at e_909_10
  have e_909_11 := arithEq_of_rows h (row := 909) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_11
  simp only [← c56478, ← c56479, k2, ← c56480] at e_909_11
  have hr := e_909_11
  simp only [e_909_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7214)) := by
  have c56481 := (publicBatchWrapper86_copies1765 a h).2.1
  have c56482 := (publicBatchWrapper86_copies1765 a h).2.2.1
  have c56483 := (publicBatchWrapper86_copies1765 a h).2.2.2.1
  have c56484 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.1
  have c56485 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.1
  have c56486 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_12 := arithEq_of_rows h (row := 909) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_12
  simp only [← c56481, ← c56482, ← c56483] at e_909_12
  have e_909_13 := arithEq_of_rows h (row := 909) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_13
  simp only [← c56484, ← c56485, k2, ← c56486] at e_909_13
  have hr := e_909_13
  simp only [e_909_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7215)) := by
  have c56487 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.1
  have c56488 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.1
  have c56489 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.1
  have c56490 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56491 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56492 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_14 := arithEq_of_rows h (row := 909) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_14
  simp only [← c56487, ← c56488, ← c56489] at e_909_14
  have e_909_15 := arithEq_of_rows h (row := 909) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_15
  simp only [← c56490, ← c56491, k2, ← c56492] at e_909_15
  have hr := e_909_15
  simp only [e_909_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7216)) := by
  have c56493 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56494 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56495 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56496 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56497 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56498 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_16 := arithEq_of_rows h (row := 909) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_16
  simp only [← c56493, ← c56494, ← c56495] at e_909_16
  have e_909_17 := arithEq_of_rows h (row := 909) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_17
  simp only [← c56496, ← c56497, k2, ← c56498] at e_909_17
  have hr := e_909_17
  simp only [e_909_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 909 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7217)) := by
  have c56499 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56500 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56501 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56502 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56503 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56504 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_909_18 := arithEq_of_rows h (row := 909) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_18
  simp only [← c56499, ← c56500, ← c56501] at e_909_18
  have e_909_19 := arithEq_of_rows h (row := 909) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_909_19
  simp only [← c56502, ← c56503, k2, ← c56504] at e_909_19
  have hr := e_909_19
  simp only [e_909_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7218)) := by
  have c56505 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56506 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56507 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56508 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56509 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56510 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_0 := arithEq_of_rows h (row := 910) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_0
  simp only [← c56505, ← c56506, ← c56507] at e_910_0
  have e_910_1 := arithEq_of_rows h (row := 910) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_1
  simp only [← c56508, ← c56509, k2, ← c56510] at e_910_1
  have hr := e_910_1
  simp only [e_910_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7219)) := by
  have c56511 := (publicBatchWrapper86_copies1765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56512 := (publicBatchWrapper86_copies1766 a h).1
  have c56513 := (publicBatchWrapper86_copies1766 a h).2.1
  have c56514 := (publicBatchWrapper86_copies1766 a h).2.2.1
  have c56515 := (publicBatchWrapper86_copies1766 a h).2.2.2.1
  have c56516 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_2 := arithEq_of_rows h (row := 910) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_2
  simp only [← c56511, ← c56512, ← c56513] at e_910_2
  have e_910_3 := arithEq_of_rows h (row := 910) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_3
  simp only [← c56514, ← c56515, k2, ← c56516] at e_910_3
  have hr := e_910_3
  simp only [e_910_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7220)) := by
  have c56517 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.1
  have c56518 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.1
  have c56519 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.1
  have c56520 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.1
  have c56521 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.1
  have c56522 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_4 := arithEq_of_rows h (row := 910) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_4
  simp only [← c56517, ← c56518, ← c56519] at e_910_4
  have e_910_5 := arithEq_of_rows h (row := 910) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_5
  simp only [← c56520, ← c56521, k2, ← c56522] at e_910_5
  have hr := e_910_5
  simp only [e_910_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7221)) := by
  have c56523 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56524 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56525 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56526 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56527 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56528 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_6 := arithEq_of_rows h (row := 910) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_6
  simp only [← c56523, ← c56524, ← c56525] at e_910_6
  have e_910_7 := arithEq_of_rows h (row := 910) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_7
  simp only [← c56526, ← c56527, k2, ← c56528] at e_910_7
  have hr := e_910_7
  simp only [e_910_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7222)) := by
  have c56529 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56530 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56531 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56532 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56533 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56534 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_8 := arithEq_of_rows h (row := 910) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_8
  simp only [← c56529, ← c56530, ← c56531] at e_910_8
  have e_910_9 := arithEq_of_rows h (row := 910) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_9
  simp only [← c56532, ← c56533, k2, ← c56534] at e_910_9
  have hr := e_910_9
  simp only [e_910_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7223)) := by
  have c56535 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56536 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56537 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56538 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56539 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56540 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_10 := arithEq_of_rows h (row := 910) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_10
  simp only [← c56535, ← c56536, ← c56537] at e_910_10
  have e_910_11 := arithEq_of_rows h (row := 910) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_11
  simp only [← c56538, ← c56539, k2, ← c56540] at e_910_11
  have hr := e_910_11
  simp only [e_910_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7224)) := by
  have c56541 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56542 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56543 := (publicBatchWrapper86_copies1766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56544 := (publicBatchWrapper86_copies1767 a h).1
  have c56545 := (publicBatchWrapper86_copies1767 a h).2.1
  have c56546 := (publicBatchWrapper86_copies1767 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_12 := arithEq_of_rows h (row := 910) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_12
  simp only [← c56541, ← c56542, ← c56543] at e_910_12
  have e_910_13 := arithEq_of_rows h (row := 910) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_13
  simp only [← c56544, ← c56545, k2, ← c56546] at e_910_13
  have hr := e_910_13
  simp only [e_910_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7225)) := by
  have c56547 := (publicBatchWrapper86_copies1767 a h).2.2.2.1
  have c56548 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.1
  have c56549 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.1
  have c56550 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.1
  have c56551 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.1
  have c56552 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_14 := arithEq_of_rows h (row := 910) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_14
  simp only [← c56547, ← c56548, ← c56549] at e_910_14
  have e_910_15 := arithEq_of_rows h (row := 910) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_15
  simp only [← c56550, ← c56551, k2, ← c56552] at e_910_15
  have hr := e_910_15
  simp only [e_910_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7226)) := by
  have c56553 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.1
  have c56554 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56555 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56556 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56557 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56558 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_16 := arithEq_of_rows h (row := 910) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_16
  simp only [← c56553, ← c56554, ← c56555] at e_910_16
  have e_910_17 := arithEq_of_rows h (row := 910) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_17
  simp only [← c56556, ← c56557, k2, ← c56558] at e_910_17
  have hr := e_910_17
  simp only [e_910_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 910 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7227)) := by
  have c56559 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56560 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56561 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56562 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56563 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56564 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_910_18 := arithEq_of_rows h (row := 910) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_18
  simp only [← c56559, ← c56560, ← c56561] at e_910_18
  have e_910_19 := arithEq_of_rows h (row := 910) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_910_19
  simp only [← c56562, ← c56563, k2, ← c56564] at e_910_19
  have hr := e_910_19
  simp only [e_910_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7228)) := by
  have c56565 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56566 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56567 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56568 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56569 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56570 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_0 := arithEq_of_rows h (row := 911) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_0
  simp only [← c56565, ← c56566, ← c56567] at e_911_0
  have e_911_1 := arithEq_of_rows h (row := 911) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_1
  simp only [← c56568, ← c56569, k2, ← c56570] at e_911_1
  have hr := e_911_1
  simp only [e_911_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7229)) := by
  have c56571 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56572 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56573 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56574 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56575 := (publicBatchWrapper86_copies1767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56576 := (publicBatchWrapper86_copies1768 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_2 := arithEq_of_rows h (row := 911) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_2
  simp only [← c56571, ← c56572, ← c56573] at e_911_2
  have e_911_3 := arithEq_of_rows h (row := 911) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_3
  simp only [← c56574, ← c56575, k2, ← c56576] at e_911_3
  have hr := e_911_3
  simp only [e_911_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7230)) := by
  have c56577 := (publicBatchWrapper86_copies1768 a h).2.1
  have c56578 := (publicBatchWrapper86_copies1768 a h).2.2.1
  have c56579 := (publicBatchWrapper86_copies1768 a h).2.2.2.1
  have c56580 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.1
  have c56581 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.1
  have c56582 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_4 := arithEq_of_rows h (row := 911) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_4
  simp only [← c56577, ← c56578, ← c56579] at e_911_4
  have e_911_5 := arithEq_of_rows h (row := 911) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_5
  simp only [← c56580, ← c56581, k2, ← c56582] at e_911_5
  have hr := e_911_5
  simp only [e_911_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7231)) := by
  have c56583 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.1
  have c56584 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.1
  have c56585 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.1
  have c56586 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56587 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56588 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_6 := arithEq_of_rows h (row := 911) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_6
  simp only [← c56583, ← c56584, ← c56585] at e_911_6
  have e_911_7 := arithEq_of_rows h (row := 911) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_7
  simp only [← c56586, ← c56587, k2, ← c56588] at e_911_7
  have hr := e_911_7
  simp only [e_911_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7232)) := by
  have c56589 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56590 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56591 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56592 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56593 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56594 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_8 := arithEq_of_rows h (row := 911) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_8
  simp only [← c56589, ← c56590, ← c56591] at e_911_8
  have e_911_9 := arithEq_of_rows h (row := 911) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_9
  simp only [← c56592, ← c56593, k2, ← c56594] at e_911_9
  have hr := e_911_9
  simp only [e_911_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7233)) := by
  have c56595 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56596 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56597 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56598 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56599 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56600 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_10 := arithEq_of_rows h (row := 911) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_10
  simp only [← c56595, ← c56596, ← c56597] at e_911_10
  have e_911_11 := arithEq_of_rows h (row := 911) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_11
  simp only [← c56598, ← c56599, k2, ← c56600] at e_911_11
  have hr := e_911_11
  simp only [e_911_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7234)) := by
  have c56601 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56602 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56603 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56604 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56605 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56606 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_12 := arithEq_of_rows h (row := 911) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_12
  simp only [← c56601, ← c56602, ← c56603] at e_911_12
  have e_911_13 := arithEq_of_rows h (row := 911) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_13
  simp only [← c56604, ← c56605, k2, ← c56606] at e_911_13
  have hr := e_911_13
  simp only [e_911_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7235)) := by
  have c56607 := (publicBatchWrapper86_copies1768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56608 := (publicBatchWrapper86_copies1769 a h).1
  have c56609 := (publicBatchWrapper86_copies1769 a h).2.1
  have c56610 := (publicBatchWrapper86_copies1769 a h).2.2.1
  have c56611 := (publicBatchWrapper86_copies1769 a h).2.2.2.1
  have c56612 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_14 := arithEq_of_rows h (row := 911) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_14
  simp only [← c56607, ← c56608, ← c56609] at e_911_14
  have e_911_15 := arithEq_of_rows h (row := 911) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_15
  simp only [← c56610, ← c56611, k2, ← c56612] at e_911_15
  have hr := e_911_15
  simp only [e_911_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7236)) := by
  have c56613 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.1
  have c56614 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.1
  have c56615 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.1
  have c56616 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.1
  have c56617 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.1
  have c56618 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_16 := arithEq_of_rows h (row := 911) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_16
  simp only [← c56613, ← c56614, ← c56615] at e_911_16
  have e_911_17 := arithEq_of_rows h (row := 911) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_17
  simp only [← c56616, ← c56617, k2, ← c56618] at e_911_17
  have hr := e_911_17
  simp only [e_911_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 911 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7237)) := by
  have c56619 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56620 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56621 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56622 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56623 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56624 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_911_18 := arithEq_of_rows h (row := 911) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_18
  simp only [← c56619, ← c56620, ← c56621] at e_911_18
  have e_911_19 := arithEq_of_rows h (row := 911) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_911_19
  simp only [← c56622, ← c56623, k2, ← c56624] at e_911_19
  have hr := e_911_19
  simp only [e_911_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7238)) := by
  have c56625 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56626 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56627 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56628 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56629 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56630 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_0 := arithEq_of_rows h (row := 912) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_0
  simp only [← c56625, ← c56626, ← c56627] at e_912_0
  have e_912_1 := arithEq_of_rows h (row := 912) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_1
  simp only [← c56628, ← c56629, k2, ← c56630] at e_912_1
  have hr := e_912_1
  simp only [e_912_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7239)) := by
  have c56631 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56632 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56633 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56634 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56635 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56636 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_2 := arithEq_of_rows h (row := 912) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_2
  simp only [← c56631, ← c56632, ← c56633] at e_912_2
  have e_912_3 := arithEq_of_rows h (row := 912) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_3
  simp only [← c56634, ← c56635, k2, ← c56636] at e_912_3
  have hr := e_912_3
  simp only [e_912_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7240)) := by
  have c56637 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56638 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56639 := (publicBatchWrapper86_copies1769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56640 := (publicBatchWrapper86_copies1770 a h).1
  have c56641 := (publicBatchWrapper86_copies1770 a h).2.1
  have c56642 := (publicBatchWrapper86_copies1770 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_4 := arithEq_of_rows h (row := 912) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_4
  simp only [← c56637, ← c56638, ← c56639] at e_912_4
  have e_912_5 := arithEq_of_rows h (row := 912) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_5
  simp only [← c56640, ← c56641, k2, ← c56642] at e_912_5
  have hr := e_912_5
  simp only [e_912_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7241)) := by
  have c56643 := (publicBatchWrapper86_copies1770 a h).2.2.2.1
  have c56644 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.1
  have c56645 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.1
  have c56646 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.1
  have c56647 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.1
  have c56648 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_6 := arithEq_of_rows h (row := 912) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_6
  simp only [← c56643, ← c56644, ← c56645] at e_912_6
  have e_912_7 := arithEq_of_rows h (row := 912) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_7
  simp only [← c56646, ← c56647, k2, ← c56648] at e_912_7
  have hr := e_912_7
  simp only [e_912_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7242)) := by
  have c56649 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.1
  have c56650 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56651 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56652 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56653 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56654 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_8 := arithEq_of_rows h (row := 912) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_8
  simp only [← c56649, ← c56650, ← c56651] at e_912_8
  have e_912_9 := arithEq_of_rows h (row := 912) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_9
  simp only [← c56652, ← c56653, k2, ← c56654] at e_912_9
  have hr := e_912_9
  simp only [e_912_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7243)) := by
  have c56655 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56656 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56657 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56658 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56659 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56660 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_10 := arithEq_of_rows h (row := 912) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_10
  simp only [← c56655, ← c56656, ← c56657] at e_912_10
  have e_912_11 := arithEq_of_rows h (row := 912) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_11
  simp only [← c56658, ← c56659, k2, ← c56660] at e_912_11
  have hr := e_912_11
  simp only [e_912_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7244)) := by
  have c56661 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56662 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56663 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56664 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56665 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56666 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_12 := arithEq_of_rows h (row := 912) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_12
  simp only [← c56661, ← c56662, ← c56663] at e_912_12
  have e_912_13 := arithEq_of_rows h (row := 912) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_13
  simp only [← c56664, ← c56665, k2, ← c56666] at e_912_13
  have hr := e_912_13
  simp only [e_912_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7245)) := by
  have c56667 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56668 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56669 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56670 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56671 := (publicBatchWrapper86_copies1770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56672 := (publicBatchWrapper86_copies1771 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_14 := arithEq_of_rows h (row := 912) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_14
  simp only [← c56667, ← c56668, ← c56669] at e_912_14
  have e_912_15 := arithEq_of_rows h (row := 912) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_15
  simp only [← c56670, ← c56671, k2, ← c56672] at e_912_15
  have hr := e_912_15
  simp only [e_912_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7246)) := by
  have c56673 := (publicBatchWrapper86_copies1771 a h).2.1
  have c56674 := (publicBatchWrapper86_copies1771 a h).2.2.1
  have c56675 := (publicBatchWrapper86_copies1771 a h).2.2.2.1
  have c56676 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.1
  have c56677 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.1
  have c56678 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_16 := arithEq_of_rows h (row := 912) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_16
  simp only [← c56673, ← c56674, ← c56675] at e_912_16
  have e_912_17 := arithEq_of_rows h (row := 912) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_17
  simp only [← c56676, ← c56677, k2, ← c56678] at e_912_17
  have hr := e_912_17
  simp only [e_912_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 912 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7247)) := by
  have c56679 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.1
  have c56680 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.1
  have c56681 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.1
  have c56682 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56683 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56684 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_912_18 := arithEq_of_rows h (row := 912) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_18
  simp only [← c56679, ← c56680, ← c56681] at e_912_18
  have e_912_19 := arithEq_of_rows h (row := 912) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_912_19
  simp only [← c56682, ← c56683, k2, ← c56684] at e_912_19
  have hr := e_912_19
  simp only [e_912_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7248)) := by
  have c56685 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56686 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56687 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56688 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56689 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56690 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_0 := arithEq_of_rows h (row := 913) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_0
  simp only [← c56685, ← c56686, ← c56687] at e_913_0
  have e_913_1 := arithEq_of_rows h (row := 913) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_1
  simp only [← c56688, ← c56689, k2, ← c56690] at e_913_1
  have hr := e_913_1
  simp only [e_913_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7249)) := by
  have c56691 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56692 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56693 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56694 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56695 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56696 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_2 := arithEq_of_rows h (row := 913) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_2
  simp only [← c56691, ← c56692, ← c56693] at e_913_2
  have e_913_3 := arithEq_of_rows h (row := 913) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_3
  simp only [← c56694, ← c56695, k2, ← c56696] at e_913_3
  have hr := e_913_3
  simp only [e_913_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7250)) := by
  have c56697 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56698 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56699 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56700 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56701 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56702 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_4 := arithEq_of_rows h (row := 913) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_4
  simp only [← c56697, ← c56698, ← c56699] at e_913_4
  have e_913_5 := arithEq_of_rows h (row := 913) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_5
  simp only [← c56700, ← c56701, k2, ← c56702] at e_913_5
  have hr := e_913_5
  simp only [e_913_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7251)) := by
  have c56703 := (publicBatchWrapper86_copies1771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56704 := (publicBatchWrapper86_copies1772 a h).1
  have c56705 := (publicBatchWrapper86_copies1772 a h).2.1
  have c56706 := (publicBatchWrapper86_copies1772 a h).2.2.1
  have c56707 := (publicBatchWrapper86_copies1772 a h).2.2.2.1
  have c56708 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_6 := arithEq_of_rows h (row := 913) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_6
  simp only [← c56703, ← c56704, ← c56705] at e_913_6
  have e_913_7 := arithEq_of_rows h (row := 913) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_7
  simp only [← c56706, ← c56707, k2, ← c56708] at e_913_7
  have hr := e_913_7
  simp only [e_913_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7252)) := by
  have c56709 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.1
  have c56710 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.1
  have c56711 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.1
  have c56712 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.1
  have c56713 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.1
  have c56714 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_8 := arithEq_of_rows h (row := 913) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_8
  simp only [← c56709, ← c56710, ← c56711] at e_913_8
  have e_913_9 := arithEq_of_rows h (row := 913) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_9
  simp only [← c56712, ← c56713, k2, ← c56714] at e_913_9
  have hr := e_913_9
  simp only [e_913_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7253)) := by
  have c56715 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56716 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56717 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56718 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56719 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56720 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_10 := arithEq_of_rows h (row := 913) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_10
  simp only [← c56715, ← c56716, ← c56717] at e_913_10
  have e_913_11 := arithEq_of_rows h (row := 913) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_11
  simp only [← c56718, ← c56719, k2, ← c56720] at e_913_11
  have hr := e_913_11
  simp only [e_913_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7254)) := by
  have c56721 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56722 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56723 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56724 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56725 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56726 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_12 := arithEq_of_rows h (row := 913) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_12
  simp only [← c56721, ← c56722, ← c56723] at e_913_12
  have e_913_13 := arithEq_of_rows h (row := 913) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_13
  simp only [← c56724, ← c56725, k2, ← c56726] at e_913_13
  have hr := e_913_13
  simp only [e_913_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7255)) := by
  have c56727 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56728 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56729 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56730 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56731 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56732 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_14 := arithEq_of_rows h (row := 913) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_14
  simp only [← c56727, ← c56728, ← c56729] at e_913_14
  have e_913_15 := arithEq_of_rows h (row := 913) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_15
  simp only [← c56730, ← c56731, k2, ← c56732] at e_913_15
  have hr := e_913_15
  simp only [e_913_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7256)) := by
  have c56733 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56734 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56735 := (publicBatchWrapper86_copies1772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56736 := (publicBatchWrapper86_copies1773 a h).1
  have c56737 := (publicBatchWrapper86_copies1773 a h).2.1
  have c56738 := (publicBatchWrapper86_copies1773 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_16 := arithEq_of_rows h (row := 913) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_16
  simp only [← c56733, ← c56734, ← c56735] at e_913_16
  have e_913_17 := arithEq_of_rows h (row := 913) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_17
  simp only [← c56736, ← c56737, k2, ← c56738] at e_913_17
  have hr := e_913_17
  simp only [e_913_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 913 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7257)) := by
  have c56739 := (publicBatchWrapper86_copies1773 a h).2.2.2.1
  have c56740 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.1
  have c56741 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.1
  have c56742 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.1
  have c56743 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.1
  have c56744 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_913_18 := arithEq_of_rows h (row := 913) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_18
  simp only [← c56739, ← c56740, ← c56741] at e_913_18
  have e_913_19 := arithEq_of_rows h (row := 913) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_913_19
  simp only [← c56742, ← c56743, k2, ← c56744] at e_913_19
  have hr := e_913_19
  simp only [e_913_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7258)) := by
  have c56745 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.1
  have c56746 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56747 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56748 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56749 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56750 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_0 := arithEq_of_rows h (row := 914) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_0
  simp only [← c56745, ← c56746, ← c56747] at e_914_0
  have e_914_1 := arithEq_of_rows h (row := 914) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_1
  simp only [← c56748, ← c56749, k2, ← c56750] at e_914_1
  have hr := e_914_1
  simp only [e_914_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7259)) := by
  have c56751 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56752 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56753 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56754 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56755 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56756 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_2 := arithEq_of_rows h (row := 914) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_2
  simp only [← c56751, ← c56752, ← c56753] at e_914_2
  have e_914_3 := arithEq_of_rows h (row := 914) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_3
  simp only [← c56754, ← c56755, k2, ← c56756] at e_914_3
  have hr := e_914_3
  simp only [e_914_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7260)) := by
  have c56757 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56758 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56759 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56760 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56761 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56762 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_4 := arithEq_of_rows h (row := 914) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_4
  simp only [← c56757, ← c56758, ← c56759] at e_914_4
  have e_914_5 := arithEq_of_rows h (row := 914) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_5
  simp only [← c56760, ← c56761, k2, ← c56762] at e_914_5
  have hr := e_914_5
  simp only [e_914_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7261)) := by
  have c56763 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56764 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56765 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56766 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56767 := (publicBatchWrapper86_copies1773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56768 := (publicBatchWrapper86_copies1774 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_6 := arithEq_of_rows h (row := 914) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_6
  simp only [← c56763, ← c56764, ← c56765] at e_914_6
  have e_914_7 := arithEq_of_rows h (row := 914) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_7
  simp only [← c56766, ← c56767, k2, ← c56768] at e_914_7
  have hr := e_914_7
  simp only [e_914_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7262)) := by
  have c56769 := (publicBatchWrapper86_copies1774 a h).2.1
  have c56770 := (publicBatchWrapper86_copies1774 a h).2.2.1
  have c56771 := (publicBatchWrapper86_copies1774 a h).2.2.2.1
  have c56772 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.1
  have c56773 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.1
  have c56774 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_8 := arithEq_of_rows h (row := 914) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_8
  simp only [← c56769, ← c56770, ← c56771] at e_914_8
  have e_914_9 := arithEq_of_rows h (row := 914) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_9
  simp only [← c56772, ← c56773, k2, ← c56774] at e_914_9
  have hr := e_914_9
  simp only [e_914_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7263)) := by
  have c56775 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.1
  have c56776 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.1
  have c56777 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.1
  have c56778 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56779 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56780 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_10 := arithEq_of_rows h (row := 914) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_10
  simp only [← c56775, ← c56776, ← c56777] at e_914_10
  have e_914_11 := arithEq_of_rows h (row := 914) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_11
  simp only [← c56778, ← c56779, k2, ← c56780] at e_914_11
  have hr := e_914_11
  simp only [e_914_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7264)) := by
  have c56781 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56782 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56783 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56784 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56785 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56786 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_12 := arithEq_of_rows h (row := 914) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_12
  simp only [← c56781, ← c56782, ← c56783] at e_914_12
  have e_914_13 := arithEq_of_rows h (row := 914) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_13
  simp only [← c56784, ← c56785, k2, ← c56786] at e_914_13
  have hr := e_914_13
  simp only [e_914_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7265)) := by
  have c56787 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56788 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56789 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56790 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56791 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56792 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_14 := arithEq_of_rows h (row := 914) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_14
  simp only [← c56787, ← c56788, ← c56789] at e_914_14
  have e_914_15 := arithEq_of_rows h (row := 914) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_15
  simp only [← c56790, ← c56791, k2, ← c56792] at e_914_15
  have hr := e_914_15
  simp only [e_914_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7266)) := by
  have c56793 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56794 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56795 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56796 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56797 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56798 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_16 := arithEq_of_rows h (row := 914) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_16
  simp only [← c56793, ← c56794, ← c56795] at e_914_16
  have e_914_17 := arithEq_of_rows h (row := 914) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_17
  simp only [← c56796, ← c56797, k2, ← c56798] at e_914_17
  have hr := e_914_17
  simp only [e_914_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 914 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7267)) := by
  have c56799 := (publicBatchWrapper86_copies1774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56800 := (publicBatchWrapper86_copies1775 a h).1
  have c56801 := (publicBatchWrapper86_copies1775 a h).2.1
  have c56802 := (publicBatchWrapper86_copies1775 a h).2.2.1
  have c56803 := (publicBatchWrapper86_copies1775 a h).2.2.2.1
  have c56804 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_914_18 := arithEq_of_rows h (row := 914) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_18
  simp only [← c56799, ← c56800, ← c56801] at e_914_18
  have e_914_19 := arithEq_of_rows h (row := 914) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_914_19
  simp only [← c56802, ← c56803, k2, ← c56804] at e_914_19
  have hr := e_914_19
  simp only [e_914_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7268)) := by
  have c56805 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.1
  have c56806 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.1
  have c56807 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.1
  have c56808 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.1
  have c56809 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.1
  have c56810 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_0 := arithEq_of_rows h (row := 915) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_0
  simp only [← c56805, ← c56806, ← c56807] at e_915_0
  have e_915_1 := arithEq_of_rows h (row := 915) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_1
  simp only [← c56808, ← c56809, k2, ← c56810] at e_915_1
  have hr := e_915_1
  simp only [e_915_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7269)) := by
  have c56811 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56812 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56813 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56814 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56815 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56816 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_2 := arithEq_of_rows h (row := 915) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_2
  simp only [← c56811, ← c56812, ← c56813] at e_915_2
  have e_915_3 := arithEq_of_rows h (row := 915) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_3
  simp only [← c56814, ← c56815, k2, ← c56816] at e_915_3
  have hr := e_915_3
  simp only [e_915_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7270)) := by
  have c56817 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56818 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56819 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56820 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56821 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56822 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_4 := arithEq_of_rows h (row := 915) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_4
  simp only [← c56817, ← c56818, ← c56819] at e_915_4
  have e_915_5 := arithEq_of_rows h (row := 915) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_5
  simp only [← c56820, ← c56821, k2, ← c56822] at e_915_5
  have hr := e_915_5
  simp only [e_915_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7271)) := by
  have c56823 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56824 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56825 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56826 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56827 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56828 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_6 := arithEq_of_rows h (row := 915) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_6
  simp only [← c56823, ← c56824, ← c56825] at e_915_6
  have e_915_7 := arithEq_of_rows h (row := 915) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_7
  simp only [← c56826, ← c56827, k2, ← c56828] at e_915_7
  have hr := e_915_7
  simp only [e_915_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7272)) := by
  have c56829 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56830 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56831 := (publicBatchWrapper86_copies1775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56832 := (publicBatchWrapper86_copies1776 a h).1
  have c56833 := (publicBatchWrapper86_copies1776 a h).2.1
  have c56834 := (publicBatchWrapper86_copies1776 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_8 := arithEq_of_rows h (row := 915) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_8
  simp only [← c56829, ← c56830, ← c56831] at e_915_8
  have e_915_9 := arithEq_of_rows h (row := 915) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_9
  simp only [← c56832, ← c56833, k2, ← c56834] at e_915_9
  have hr := e_915_9
  simp only [e_915_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7273)) := by
  have c56835 := (publicBatchWrapper86_copies1776 a h).2.2.2.1
  have c56836 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.1
  have c56837 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.1
  have c56838 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.1
  have c56839 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.1
  have c56840 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_10 := arithEq_of_rows h (row := 915) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_10
  simp only [← c56835, ← c56836, ← c56837] at e_915_10
  have e_915_11 := arithEq_of_rows h (row := 915) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_11
  simp only [← c56838, ← c56839, k2, ← c56840] at e_915_11
  have hr := e_915_11
  simp only [e_915_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7274)) := by
  have c56841 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.1
  have c56842 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56843 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56844 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56845 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56846 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_12 := arithEq_of_rows h (row := 915) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_12
  simp only [← c56841, ← c56842, ← c56843] at e_915_12
  have e_915_13 := arithEq_of_rows h (row := 915) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_13
  simp only [← c56844, ← c56845, k2, ← c56846] at e_915_13
  have hr := e_915_13
  simp only [e_915_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7275)) := by
  have c56847 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56848 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56849 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56850 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56851 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56852 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_14 := arithEq_of_rows h (row := 915) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_14
  simp only [← c56847, ← c56848, ← c56849] at e_915_14
  have e_915_15 := arithEq_of_rows h (row := 915) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_15
  simp only [← c56850, ← c56851, k2, ← c56852] at e_915_15
  have hr := e_915_15
  simp only [e_915_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7276)) := by
  have c56853 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56854 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56855 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56856 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56857 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56858 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_16 := arithEq_of_rows h (row := 915) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_16
  simp only [← c56853, ← c56854, ← c56855] at e_915_16
  have e_915_17 := arithEq_of_rows h (row := 915) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_17
  simp only [← c56856, ← c56857, k2, ← c56858] at e_915_17
  have hr := e_915_17
  simp only [e_915_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 915 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7277)) := by
  have c56859 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56860 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56861 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56862 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56863 := (publicBatchWrapper86_copies1776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56864 := (publicBatchWrapper86_copies1777 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_915_18 := arithEq_of_rows h (row := 915) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_18
  simp only [← c56859, ← c56860, ← c56861] at e_915_18
  have e_915_19 := arithEq_of_rows h (row := 915) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_915_19
  simp only [← c56862, ← c56863, k2, ← c56864] at e_915_19
  have hr := e_915_19
  simp only [e_915_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7278)) := by
  have c56865 := (publicBatchWrapper86_copies1777 a h).2.1
  have c56866 := (publicBatchWrapper86_copies1777 a h).2.2.1
  have c56867 := (publicBatchWrapper86_copies1777 a h).2.2.2.1
  have c56868 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.1
  have c56869 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.1
  have c56870 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_0 := arithEq_of_rows h (row := 916) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_0
  simp only [← c56865, ← c56866, ← c56867] at e_916_0
  have e_916_1 := arithEq_of_rows h (row := 916) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_1
  simp only [← c56868, ← c56869, k2, ← c56870] at e_916_1
  have hr := e_916_1
  simp only [e_916_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7279)) := by
  have c56871 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.1
  have c56872 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.1
  have c56873 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.1
  have c56874 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56875 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56876 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_2 := arithEq_of_rows h (row := 916) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_2
  simp only [← c56871, ← c56872, ← c56873] at e_916_2
  have e_916_3 := arithEq_of_rows h (row := 916) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_3
  simp only [← c56874, ← c56875, k2, ← c56876] at e_916_3
  have hr := e_916_3
  simp only [e_916_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7280)) := by
  have c56877 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56878 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56879 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56880 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56881 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56882 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_4 := arithEq_of_rows h (row := 916) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_4
  simp only [← c56877, ← c56878, ← c56879] at e_916_4
  have e_916_5 := arithEq_of_rows h (row := 916) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_5
  simp only [← c56880, ← c56881, k2, ← c56882] at e_916_5
  have hr := e_916_5
  simp only [e_916_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7281)) := by
  have c56883 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56884 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56885 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56886 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56887 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56888 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_6 := arithEq_of_rows h (row := 916) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_6
  simp only [← c56883, ← c56884, ← c56885] at e_916_6
  have e_916_7 := arithEq_of_rows h (row := 916) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_7
  simp only [← c56886, ← c56887, k2, ← c56888] at e_916_7
  have hr := e_916_7
  simp only [e_916_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7282)) := by
  have c56889 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56890 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56891 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56892 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56893 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56894 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_8 := arithEq_of_rows h (row := 916) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_8
  simp only [← c56889, ← c56890, ← c56891] at e_916_8
  have e_916_9 := arithEq_of_rows h (row := 916) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_9
  simp only [← c56892, ← c56893, k2, ← c56894] at e_916_9
  have hr := e_916_9
  simp only [e_916_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7283)) := by
  have c56895 := (publicBatchWrapper86_copies1777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56896 := (publicBatchWrapper86_copies1778 a h).1
  have c56897 := (publicBatchWrapper86_copies1778 a h).2.1
  have c56898 := (publicBatchWrapper86_copies1778 a h).2.2.1
  have c56899 := (publicBatchWrapper86_copies1778 a h).2.2.2.1
  have c56900 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_10 := arithEq_of_rows h (row := 916) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_10
  simp only [← c56895, ← c56896, ← c56897] at e_916_10
  have e_916_11 := arithEq_of_rows h (row := 916) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_11
  simp only [← c56898, ← c56899, k2, ← c56900] at e_916_11
  have hr := e_916_11
  simp only [e_916_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7284)) := by
  have c56901 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.1
  have c56902 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.1
  have c56903 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.1
  have c56904 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.1
  have c56905 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.1
  have c56906 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_12 := arithEq_of_rows h (row := 916) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_12
  simp only [← c56901, ← c56902, ← c56903] at e_916_12
  have e_916_13 := arithEq_of_rows h (row := 916) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_13
  simp only [← c56904, ← c56905, k2, ← c56906] at e_916_13
  have hr := e_916_13
  simp only [e_916_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7285)) := by
  have c56907 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56908 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56909 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56910 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56911 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56912 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_14 := arithEq_of_rows h (row := 916) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_14
  simp only [← c56907, ← c56908, ← c56909] at e_916_14
  have e_916_15 := arithEq_of_rows h (row := 916) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_15
  simp only [← c56910, ← c56911, k2, ← c56912] at e_916_15
  have hr := e_916_15
  simp only [e_916_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7286)) := by
  have c56913 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56914 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56915 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56916 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56917 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56918 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_16 := arithEq_of_rows h (row := 916) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_16
  simp only [← c56913, ← c56914, ← c56915] at e_916_16
  have e_916_17 := arithEq_of_rows h (row := 916) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_17
  simp only [← c56916, ← c56917, k2, ← c56918] at e_916_17
  have hr := e_916_17
  simp only [e_916_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 916 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7287)) := by
  have c56919 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56920 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56921 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56922 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56923 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56924 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_916_18 := arithEq_of_rows h (row := 916) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_18
  simp only [← c56919, ← c56920, ← c56921] at e_916_18
  have e_916_19 := arithEq_of_rows h (row := 916) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_916_19
  simp only [← c56922, ← c56923, k2, ← c56924] at e_916_19
  have hr := e_916_19
  simp only [e_916_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7288)) := by
  have c56925 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56926 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56927 := (publicBatchWrapper86_copies1778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56928 := (publicBatchWrapper86_copies1779 a h).1
  have c56929 := (publicBatchWrapper86_copies1779 a h).2.1
  have c56930 := (publicBatchWrapper86_copies1779 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_0 := arithEq_of_rows h (row := 917) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_0
  simp only [← c56925, ← c56926, ← c56927] at e_917_0
  have e_917_1 := arithEq_of_rows h (row := 917) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_1
  simp only [← c56928, ← c56929, k2, ← c56930] at e_917_1
  have hr := e_917_1
  simp only [e_917_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7289)) := by
  have c56931 := (publicBatchWrapper86_copies1779 a h).2.2.2.1
  have c56932 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.1
  have c56933 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.1
  have c56934 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.1
  have c56935 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.1
  have c56936 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_2 := arithEq_of_rows h (row := 917) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_2
  simp only [← c56931, ← c56932, ← c56933] at e_917_2
  have e_917_3 := arithEq_of_rows h (row := 917) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_3
  simp only [← c56934, ← c56935, k2, ← c56936] at e_917_3
  have hr := e_917_3
  simp only [e_917_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7290)) := by
  have c56937 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.1
  have c56938 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56939 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56940 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56941 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56942 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_4 := arithEq_of_rows h (row := 917) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_4
  simp only [← c56937, ← c56938, ← c56939] at e_917_4
  have e_917_5 := arithEq_of_rows h (row := 917) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_5
  simp only [← c56940, ← c56941, k2, ← c56942] at e_917_5
  have hr := e_917_5
  simp only [e_917_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7291)) := by
  have c56943 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56944 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56945 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56946 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56947 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56948 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_6 := arithEq_of_rows h (row := 917) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_6
  simp only [← c56943, ← c56944, ← c56945] at e_917_6
  have e_917_7 := arithEq_of_rows h (row := 917) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_7
  simp only [← c56946, ← c56947, k2, ← c56948] at e_917_7
  have hr := e_917_7
  simp only [e_917_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7292)) := by
  have c56949 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56950 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56951 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56952 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56953 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56954 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_8 := arithEq_of_rows h (row := 917) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_8
  simp only [← c56949, ← c56950, ← c56951] at e_917_8
  have e_917_9 := arithEq_of_rows h (row := 917) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_9
  simp only [← c56952, ← c56953, k2, ← c56954] at e_917_9
  have hr := e_917_9
  simp only [e_917_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7293)) := by
  have c56955 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56956 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56957 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56958 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56959 := (publicBatchWrapper86_copies1779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56960 := (publicBatchWrapper86_copies1780 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_10 := arithEq_of_rows h (row := 917) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_10
  simp only [← c56955, ← c56956, ← c56957] at e_917_10
  have e_917_11 := arithEq_of_rows h (row := 917) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_11
  simp only [← c56958, ← c56959, k2, ← c56960] at e_917_11
  have hr := e_917_11
  simp only [e_917_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7294)) := by
  have c56961 := (publicBatchWrapper86_copies1780 a h).2.1
  have c56962 := (publicBatchWrapper86_copies1780 a h).2.2.1
  have c56963 := (publicBatchWrapper86_copies1780 a h).2.2.2.1
  have c56964 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.1
  have c56965 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.1
  have c56966 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_12 := arithEq_of_rows h (row := 917) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_12
  simp only [← c56961, ← c56962, ← c56963] at e_917_12
  have e_917_13 := arithEq_of_rows h (row := 917) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_13
  simp only [← c56964, ← c56965, k2, ← c56966] at e_917_13
  have hr := e_917_13
  simp only [e_917_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7295)) := by
  have c56967 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.1
  have c56968 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.1
  have c56969 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.1
  have c56970 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c56971 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c56972 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_14 := arithEq_of_rows h (row := 917) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_14
  simp only [← c56967, ← c56968, ← c56969] at e_917_14
  have e_917_15 := arithEq_of_rows h (row := 917) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_15
  simp only [← c56970, ← c56971, k2, ← c56972] at e_917_15
  have hr := e_917_15
  simp only [e_917_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7296)) := by
  have c56973 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56974 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56975 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56976 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56977 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56978 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_16 := arithEq_of_rows h (row := 917) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_16
  simp only [← c56973, ← c56974, ← c56975] at e_917_16
  have e_917_17 := arithEq_of_rows h (row := 917) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_17
  simp only [← c56976, ← c56977, k2, ← c56978] at e_917_17
  have hr := e_917_17
  simp only [e_917_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 917 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7297)) := by
  have c56979 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56980 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56981 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56982 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56983 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56984 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_917_18 := arithEq_of_rows h (row := 917) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_18
  simp only [← c56979, ← c56980, ← c56981] at e_917_18
  have e_917_19 := arithEq_of_rows h (row := 917) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_917_19
  simp only [← c56982, ← c56983, k2, ← c56984] at e_917_19
  have hr := e_917_19
  simp only [e_917_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7298)) := by
  have c56985 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56986 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56987 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56988 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56989 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c56990 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_0 := arithEq_of_rows h (row := 918) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_0
  simp only [← c56985, ← c56986, ← c56987] at e_918_0
  have e_918_1 := arithEq_of_rows h (row := 918) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_1
  simp only [← c56988, ← c56989, k2, ← c56990] at e_918_1
  have hr := e_918_1
  simp only [e_918_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7299)) := by
  have c56991 := (publicBatchWrapper86_copies1780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c56992 := (publicBatchWrapper86_copies1781 a h).1
  have c56993 := (publicBatchWrapper86_copies1781 a h).2.1
  have c56994 := (publicBatchWrapper86_copies1781 a h).2.2.1
  have c56995 := (publicBatchWrapper86_copies1781 a h).2.2.2.1
  have c56996 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_2 := arithEq_of_rows h (row := 918) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_2
  simp only [← c56991, ← c56992, ← c56993] at e_918_2
  have e_918_3 := arithEq_of_rows h (row := 918) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_3
  simp only [← c56994, ← c56995, k2, ← c56996] at e_918_3
  have hr := e_918_3
  simp only [e_918_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7300)) := by
  have c56997 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.1
  have c56998 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.1
  have c56999 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.1
  have c57000 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.1
  have c57001 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.1
  have c57002 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_4 := arithEq_of_rows h (row := 918) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_4
  simp only [← c56997, ← c56998, ← c56999] at e_918_4
  have e_918_5 := arithEq_of_rows h (row := 918) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_5
  simp only [← c57000, ← c57001, k2, ← c57002] at e_918_5
  have hr := e_918_5
  simp only [e_918_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7301)) := by
  have c57003 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57004 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57005 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57006 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57007 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57008 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_6 := arithEq_of_rows h (row := 918) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_6
  simp only [← c57003, ← c57004, ← c57005] at e_918_6
  have e_918_7 := arithEq_of_rows h (row := 918) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_7
  simp only [← c57006, ← c57007, k2, ← c57008] at e_918_7
  have hr := e_918_7
  simp only [e_918_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7302)) := by
  have c57009 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57010 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57011 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57012 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57013 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57014 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_8 := arithEq_of_rows h (row := 918) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_8
  simp only [← c57009, ← c57010, ← c57011] at e_918_8
  have e_918_9 := arithEq_of_rows h (row := 918) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_9
  simp only [← c57012, ← c57013, k2, ← c57014] at e_918_9
  have hr := e_918_9
  simp only [e_918_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7303)) := by
  have c57015 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57016 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57017 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57018 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57019 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57020 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_10 := arithEq_of_rows h (row := 918) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_10
  simp only [← c57015, ← c57016, ← c57017] at e_918_10
  have e_918_11 := arithEq_of_rows h (row := 918) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_11
  simp only [← c57018, ← c57019, k2, ← c57020] at e_918_11
  have hr := e_918_11
  simp only [e_918_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7304)) := by
  have c57021 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57022 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57023 := (publicBatchWrapper86_copies1781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57024 := (publicBatchWrapper86_copies1782 a h).1
  have c57025 := (publicBatchWrapper86_copies1782 a h).2.1
  have c57026 := (publicBatchWrapper86_copies1782 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_12 := arithEq_of_rows h (row := 918) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_12
  simp only [← c57021, ← c57022, ← c57023] at e_918_12
  have e_918_13 := arithEq_of_rows h (row := 918) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_13
  simp only [← c57024, ← c57025, k2, ← c57026] at e_918_13
  have hr := e_918_13
  simp only [e_918_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7305)) := by
  have c57027 := (publicBatchWrapper86_copies1782 a h).2.2.2.1
  have c57028 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.1
  have c57029 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.1
  have c57030 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.1
  have c57031 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.1
  have c57032 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_14 := arithEq_of_rows h (row := 918) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_14
  simp only [← c57027, ← c57028, ← c57029] at e_918_14
  have e_918_15 := arithEq_of_rows h (row := 918) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_15
  simp only [← c57030, ← c57031, k2, ← c57032] at e_918_15
  have hr := e_918_15
  simp only [e_918_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7306)) := by
  have c57033 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.1
  have c57034 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57035 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57036 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57037 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57038 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_16 := arithEq_of_rows h (row := 918) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_16
  simp only [← c57033, ← c57034, ← c57035] at e_918_16
  have e_918_17 := arithEq_of_rows h (row := 918) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_17
  simp only [← c57036, ← c57037, k2, ← c57038] at e_918_17
  have hr := e_918_17
  simp only [e_918_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 918 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7307)) := by
  have c57039 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57040 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57041 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57042 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57043 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57044 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_918_18 := arithEq_of_rows h (row := 918) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_18
  simp only [← c57039, ← c57040, ← c57041] at e_918_18
  have e_918_19 := arithEq_of_rows h (row := 918) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_918_19
  simp only [← c57042, ← c57043, k2, ← c57044] at e_918_19
  have hr := e_918_19
  simp only [e_918_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7308)) := by
  have c57045 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57046 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57047 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57048 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57049 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57050 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_0 := arithEq_of_rows h (row := 919) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_0
  simp only [← c57045, ← c57046, ← c57047] at e_919_0
  have e_919_1 := arithEq_of_rows h (row := 919) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_1
  simp only [← c57048, ← c57049, k2, ← c57050] at e_919_1
  have hr := e_919_1
  simp only [e_919_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7309)) := by
  have c57051 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57052 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57053 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57054 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57055 := (publicBatchWrapper86_copies1782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57056 := (publicBatchWrapper86_copies1783 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_2 := arithEq_of_rows h (row := 919) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_2
  simp only [← c57051, ← c57052, ← c57053] at e_919_2
  have e_919_3 := arithEq_of_rows h (row := 919) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_3
  simp only [← c57054, ← c57055, k2, ← c57056] at e_919_3
  have hr := e_919_3
  simp only [e_919_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7310)) := by
  have c57057 := (publicBatchWrapper86_copies1783 a h).2.1
  have c57058 := (publicBatchWrapper86_copies1783 a h).2.2.1
  have c57059 := (publicBatchWrapper86_copies1783 a h).2.2.2.1
  have c57060 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.1
  have c57061 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.1
  have c57062 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_4 := arithEq_of_rows h (row := 919) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_4
  simp only [← c57057, ← c57058, ← c57059] at e_919_4
  have e_919_5 := arithEq_of_rows h (row := 919) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_5
  simp only [← c57060, ← c57061, k2, ← c57062] at e_919_5
  have hr := e_919_5
  simp only [e_919_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7311)) := by
  have c57063 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.1
  have c57064 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.1
  have c57065 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.1
  have c57066 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57067 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57068 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_6 := arithEq_of_rows h (row := 919) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_6
  simp only [← c57063, ← c57064, ← c57065] at e_919_6
  have e_919_7 := arithEq_of_rows h (row := 919) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_7
  simp only [← c57066, ← c57067, k2, ← c57068] at e_919_7
  have hr := e_919_7
  simp only [e_919_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7312)) := by
  have c57069 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57070 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57071 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57072 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57073 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57074 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_8 := arithEq_of_rows h (row := 919) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_8
  simp only [← c57069, ← c57070, ← c57071] at e_919_8
  have e_919_9 := arithEq_of_rows h (row := 919) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_9
  simp only [← c57072, ← c57073, k2, ← c57074] at e_919_9
  have hr := e_919_9
  simp only [e_919_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7313)) := by
  have c57075 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57076 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57077 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57078 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57079 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57080 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_10 := arithEq_of_rows h (row := 919) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_10
  simp only [← c57075, ← c57076, ← c57077] at e_919_10
  have e_919_11 := arithEq_of_rows h (row := 919) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_11
  simp only [← c57078, ← c57079, k2, ← c57080] at e_919_11
  have hr := e_919_11
  simp only [e_919_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7314)) := by
  have c57081 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57082 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57083 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57084 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57085 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57086 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_12 := arithEq_of_rows h (row := 919) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_12
  simp only [← c57081, ← c57082, ← c57083] at e_919_12
  have e_919_13 := arithEq_of_rows h (row := 919) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_13
  simp only [← c57084, ← c57085, k2, ← c57086] at e_919_13
  have hr := e_919_13
  simp only [e_919_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7315)) := by
  have c57087 := (publicBatchWrapper86_copies1783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57088 := (publicBatchWrapper86_copies1784 a h).1
  have c57089 := (publicBatchWrapper86_copies1784 a h).2.1
  have c57090 := (publicBatchWrapper86_copies1784 a h).2.2.1
  have c57091 := (publicBatchWrapper86_copies1784 a h).2.2.2.1
  have c57092 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_14 := arithEq_of_rows h (row := 919) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_14
  simp only [← c57087, ← c57088, ← c57089] at e_919_14
  have e_919_15 := arithEq_of_rows h (row := 919) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_15
  simp only [← c57090, ← c57091, k2, ← c57092] at e_919_15
  have hr := e_919_15
  simp only [e_919_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7316)) := by
  have c57093 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.1
  have c57094 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.1
  have c57095 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.1
  have c57096 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.1
  have c57097 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.1
  have c57098 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_16 := arithEq_of_rows h (row := 919) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_16
  simp only [← c57093, ← c57094, ← c57095] at e_919_16
  have e_919_17 := arithEq_of_rows h (row := 919) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_17
  simp only [← c57096, ← c57097, k2, ← c57098] at e_919_17
  have hr := e_919_17
  simp only [e_919_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 919 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7317)) := by
  have c57099 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57100 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57101 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57102 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57103 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57104 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_919_18 := arithEq_of_rows h (row := 919) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_18
  simp only [← c57099, ← c57100, ← c57101] at e_919_18
  have e_919_19 := arithEq_of_rows h (row := 919) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_919_19
  simp only [← c57102, ← c57103, k2, ← c57104] at e_919_19
  have hr := e_919_19
  simp only [e_919_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7318)) := by
  have c57105 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57106 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57107 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57108 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57109 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57110 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_0 := arithEq_of_rows h (row := 920) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_0
  simp only [← c57105, ← c57106, ← c57107] at e_920_0
  have e_920_1 := arithEq_of_rows h (row := 920) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_1
  simp only [← c57108, ← c57109, k2, ← c57110] at e_920_1
  have hr := e_920_1
  simp only [e_920_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7319)) := by
  have c57111 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57112 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57113 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57114 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57115 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57116 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_2 := arithEq_of_rows h (row := 920) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_2
  simp only [← c57111, ← c57112, ← c57113] at e_920_2
  have e_920_3 := arithEq_of_rows h (row := 920) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_3
  simp only [← c57114, ← c57115, k2, ← c57116] at e_920_3
  have hr := e_920_3
  simp only [e_920_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7320)) := by
  have c57117 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57118 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57119 := (publicBatchWrapper86_copies1784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57120 := (publicBatchWrapper86_copies1785 a h).1
  have c57121 := (publicBatchWrapper86_copies1785 a h).2.1
  have c57122 := (publicBatchWrapper86_copies1785 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_4 := arithEq_of_rows h (row := 920) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_4
  simp only [← c57117, ← c57118, ← c57119] at e_920_4
  have e_920_5 := arithEq_of_rows h (row := 920) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_5
  simp only [← c57120, ← c57121, k2, ← c57122] at e_920_5
  have hr := e_920_5
  simp only [e_920_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7321)) := by
  have c57123 := (publicBatchWrapper86_copies1785 a h).2.2.2.1
  have c57124 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.1
  have c57125 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.1
  have c57126 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.1
  have c57127 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.1
  have c57128 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_6 := arithEq_of_rows h (row := 920) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_6
  simp only [← c57123, ← c57124, ← c57125] at e_920_6
  have e_920_7 := arithEq_of_rows h (row := 920) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_7
  simp only [← c57126, ← c57127, k2, ← c57128] at e_920_7
  have hr := e_920_7
  simp only [e_920_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7322)) := by
  have c57129 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.1
  have c57130 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57131 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57132 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57133 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57134 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_8 := arithEq_of_rows h (row := 920) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_8
  simp only [← c57129, ← c57130, ← c57131] at e_920_8
  have e_920_9 := arithEq_of_rows h (row := 920) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_9
  simp only [← c57132, ← c57133, k2, ← c57134] at e_920_9
  have hr := e_920_9
  simp only [e_920_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7323)) := by
  have c57135 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57136 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57137 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57138 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57139 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57140 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_10 := arithEq_of_rows h (row := 920) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_10
  simp only [← c57135, ← c57136, ← c57137] at e_920_10
  have e_920_11 := arithEq_of_rows h (row := 920) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_11
  simp only [← c57138, ← c57139, k2, ← c57140] at e_920_11
  have hr := e_920_11
  simp only [e_920_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7324)) := by
  have c57141 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57142 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57143 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57144 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57145 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57146 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_12 := arithEq_of_rows h (row := 920) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_12
  simp only [← c57141, ← c57142, ← c57143] at e_920_12
  have e_920_13 := arithEq_of_rows h (row := 920) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_13
  simp only [← c57144, ← c57145, k2, ← c57146] at e_920_13
  have hr := e_920_13
  simp only [e_920_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7325)) := by
  have c57147 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57148 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57149 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57150 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57151 := (publicBatchWrapper86_copies1785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57152 := (publicBatchWrapper86_copies1786 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_14 := arithEq_of_rows h (row := 920) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_14
  simp only [← c57147, ← c57148, ← c57149] at e_920_14
  have e_920_15 := arithEq_of_rows h (row := 920) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_15
  simp only [← c57150, ← c57151, k2, ← c57152] at e_920_15
  have hr := e_920_15
  simp only [e_920_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7326)) := by
  have c57153 := (publicBatchWrapper86_copies1786 a h).2.1
  have c57154 := (publicBatchWrapper86_copies1786 a h).2.2.1
  have c57155 := (publicBatchWrapper86_copies1786 a h).2.2.2.1
  have c57156 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.1
  have c57157 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.1
  have c57158 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_16 := arithEq_of_rows h (row := 920) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_16
  simp only [← c57153, ← c57154, ← c57155] at e_920_16
  have e_920_17 := arithEq_of_rows h (row := 920) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_17
  simp only [← c57156, ← c57157, k2, ← c57158] at e_920_17
  have hr := e_920_17
  simp only [e_920_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 920 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7327)) := by
  have c57159 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.1
  have c57160 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.1
  have c57161 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.1
  have c57162 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57163 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57164 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_920_18 := arithEq_of_rows h (row := 920) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_18
  simp only [← c57159, ← c57160, ← c57161] at e_920_18
  have e_920_19 := arithEq_of_rows h (row := 920) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_920_19
  simp only [← c57162, ← c57163, k2, ← c57164] at e_920_19
  have hr := e_920_19
  simp only [e_920_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7328)) := by
  have c57165 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57166 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57167 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57168 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57169 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57170 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_0 := arithEq_of_rows h (row := 921) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_0
  simp only [← c57165, ← c57166, ← c57167] at e_921_0
  have e_921_1 := arithEq_of_rows h (row := 921) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_1
  simp only [← c57168, ← c57169, k2, ← c57170] at e_921_1
  have hr := e_921_1
  simp only [e_921_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7329)) := by
  have c57171 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57172 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57173 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57174 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57175 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57176 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_2 := arithEq_of_rows h (row := 921) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_2
  simp only [← c57171, ← c57172, ← c57173] at e_921_2
  have e_921_3 := arithEq_of_rows h (row := 921) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_3
  simp only [← c57174, ← c57175, k2, ← c57176] at e_921_3
  have hr := e_921_3
  simp only [e_921_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7330)) := by
  have c57177 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57178 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57179 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57180 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57181 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57182 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_4 := arithEq_of_rows h (row := 921) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_4
  simp only [← c57177, ← c57178, ← c57179] at e_921_4
  have e_921_5 := arithEq_of_rows h (row := 921) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_5
  simp only [← c57180, ← c57181, k2, ← c57182] at e_921_5
  have hr := e_921_5
  simp only [e_921_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7331)) := by
  have c57183 := (publicBatchWrapper86_copies1786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57184 := (publicBatchWrapper86_copies1787 a h).1
  have c57185 := (publicBatchWrapper86_copies1787 a h).2.1
  have c57186 := (publicBatchWrapper86_copies1787 a h).2.2.1
  have c57187 := (publicBatchWrapper86_copies1787 a h).2.2.2.1
  have c57188 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_6 := arithEq_of_rows h (row := 921) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_6
  simp only [← c57183, ← c57184, ← c57185] at e_921_6
  have e_921_7 := arithEq_of_rows h (row := 921) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_7
  simp only [← c57186, ← c57187, k2, ← c57188] at e_921_7
  have hr := e_921_7
  simp only [e_921_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7332)) := by
  have c57189 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.1
  have c57190 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.1
  have c57191 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.1
  have c57192 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.1
  have c57193 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.1
  have c57194 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_8 := arithEq_of_rows h (row := 921) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_8
  simp only [← c57189, ← c57190, ← c57191] at e_921_8
  have e_921_9 := arithEq_of_rows h (row := 921) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_9
  simp only [← c57192, ← c57193, k2, ← c57194] at e_921_9
  have hr := e_921_9
  simp only [e_921_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7333)) := by
  have c57195 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57196 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57197 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57198 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57199 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57200 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_10 := arithEq_of_rows h (row := 921) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_10
  simp only [← c57195, ← c57196, ← c57197] at e_921_10
  have e_921_11 := arithEq_of_rows h (row := 921) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_11
  simp only [← c57198, ← c57199, k2, ← c57200] at e_921_11
  have hr := e_921_11
  simp only [e_921_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7334)) := by
  have c57201 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57202 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57203 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57204 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57205 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57206 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_12 := arithEq_of_rows h (row := 921) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_12
  simp only [← c57201, ← c57202, ← c57203] at e_921_12
  have e_921_13 := arithEq_of_rows h (row := 921) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_13
  simp only [← c57204, ← c57205, k2, ← c57206] at e_921_13
  have hr := e_921_13
  simp only [e_921_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7335)) := by
  have c57207 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57208 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57209 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57210 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57211 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57212 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_14 := arithEq_of_rows h (row := 921) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_14
  simp only [← c57207, ← c57208, ← c57209] at e_921_14
  have e_921_15 := arithEq_of_rows h (row := 921) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_15
  simp only [← c57210, ← c57211, k2, ← c57212] at e_921_15
  have hr := e_921_15
  simp only [e_921_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7336)) := by
  have c57213 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57214 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57215 := (publicBatchWrapper86_copies1787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57216 := (publicBatchWrapper86_copies1788 a h).1
  have c57217 := (publicBatchWrapper86_copies1788 a h).2.1
  have c57218 := (publicBatchWrapper86_copies1788 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_16 := arithEq_of_rows h (row := 921) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_16
  simp only [← c57213, ← c57214, ← c57215] at e_921_16
  have e_921_17 := arithEq_of_rows h (row := 921) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_17
  simp only [← c57216, ← c57217, k2, ← c57218] at e_921_17
  have hr := e_921_17
  simp only [e_921_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 921 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7337)) := by
  have c57219 := (publicBatchWrapper86_copies1788 a h).2.2.2.1
  have c57220 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.1
  have c57221 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.1
  have c57222 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.1
  have c57223 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.1
  have c57224 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_921_18 := arithEq_of_rows h (row := 921) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_18
  simp only [← c57219, ← c57220, ← c57221] at e_921_18
  have e_921_19 := arithEq_of_rows h (row := 921) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_921_19
  simp only [← c57222, ← c57223, k2, ← c57224] at e_921_19
  have hr := e_921_19
  simp only [e_921_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7338)) := by
  have c57225 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.1
  have c57226 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57227 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57228 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57229 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57230 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_0 := arithEq_of_rows h (row := 922) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_0
  simp only [← c57225, ← c57226, ← c57227] at e_922_0
  have e_922_1 := arithEq_of_rows h (row := 922) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_1
  simp only [← c57228, ← c57229, k2, ← c57230] at e_922_1
  have hr := e_922_1
  simp only [e_922_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7339)) := by
  have c57231 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57232 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57233 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57234 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57235 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57236 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_2 := arithEq_of_rows h (row := 922) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_2
  simp only [← c57231, ← c57232, ← c57233] at e_922_2
  have e_922_3 := arithEq_of_rows h (row := 922) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_3
  simp only [← c57234, ← c57235, k2, ← c57236] at e_922_3
  have hr := e_922_3
  simp only [e_922_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7340)) := by
  have c57237 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57238 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57239 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57240 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57241 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57242 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_4 := arithEq_of_rows h (row := 922) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_4
  simp only [← c57237, ← c57238, ← c57239] at e_922_4
  have e_922_5 := arithEq_of_rows h (row := 922) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_5
  simp only [← c57240, ← c57241, k2, ← c57242] at e_922_5
  have hr := e_922_5
  simp only [e_922_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7341)) := by
  have c57243 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57244 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57245 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57246 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57247 := (publicBatchWrapper86_copies1788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57248 := (publicBatchWrapper86_copies1789 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_6 := arithEq_of_rows h (row := 922) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_6
  simp only [← c57243, ← c57244, ← c57245] at e_922_6
  have e_922_7 := arithEq_of_rows h (row := 922) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_7
  simp only [← c57246, ← c57247, k2, ← c57248] at e_922_7
  have hr := e_922_7
  simp only [e_922_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7342)) := by
  have c57249 := (publicBatchWrapper86_copies1789 a h).2.1
  have c57250 := (publicBatchWrapper86_copies1789 a h).2.2.1
  have c57251 := (publicBatchWrapper86_copies1789 a h).2.2.2.1
  have c57252 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.1
  have c57253 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.1
  have c57254 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_8 := arithEq_of_rows h (row := 922) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_8
  simp only [← c57249, ← c57250, ← c57251] at e_922_8
  have e_922_9 := arithEq_of_rows h (row := 922) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_9
  simp only [← c57252, ← c57253, k2, ← c57254] at e_922_9
  have hr := e_922_9
  simp only [e_922_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7343)) := by
  have c57255 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.1
  have c57256 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.1
  have c57257 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.1
  have c57258 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57259 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57260 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_10 := arithEq_of_rows h (row := 922) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_10
  simp only [← c57255, ← c57256, ← c57257] at e_922_10
  have e_922_11 := arithEq_of_rows h (row := 922) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_11
  simp only [← c57258, ← c57259, k2, ← c57260] at e_922_11
  have hr := e_922_11
  simp only [e_922_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7344)) := by
  have c57261 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57262 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57263 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57264 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57265 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57266 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_12 := arithEq_of_rows h (row := 922) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_12
  simp only [← c57261, ← c57262, ← c57263] at e_922_12
  have e_922_13 := arithEq_of_rows h (row := 922) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_13
  simp only [← c57264, ← c57265, k2, ← c57266] at e_922_13
  have hr := e_922_13
  simp only [e_922_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7345)) := by
  have c57267 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57268 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57269 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57270 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57271 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57272 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_14 := arithEq_of_rows h (row := 922) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_14
  simp only [← c57267, ← c57268, ← c57269] at e_922_14
  have e_922_15 := arithEq_of_rows h (row := 922) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_15
  simp only [← c57270, ← c57271, k2, ← c57272] at e_922_15
  have hr := e_922_15
  simp only [e_922_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7346)) := by
  have c57273 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57274 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57275 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57276 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57277 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57278 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_16 := arithEq_of_rows h (row := 922) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_16
  simp only [← c57273, ← c57274, ← c57275] at e_922_16
  have e_922_17 := arithEq_of_rows h (row := 922) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_17
  simp only [← c57276, ← c57277, k2, ← c57278] at e_922_17
  have hr := e_922_17
  simp only [e_922_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 922 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7347)) := by
  have c57279 := (publicBatchWrapper86_copies1789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57280 := (publicBatchWrapper86_copies1790 a h).1
  have c57281 := (publicBatchWrapper86_copies1790 a h).2.1
  have c57282 := (publicBatchWrapper86_copies1790 a h).2.2.1
  have c57283 := (publicBatchWrapper86_copies1790 a h).2.2.2.1
  have c57284 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_922_18 := arithEq_of_rows h (row := 922) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_18
  simp only [← c57279, ← c57280, ← c57281] at e_922_18
  have e_922_19 := arithEq_of_rows h (row := 922) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_922_19
  simp only [← c57282, ← c57283, k2, ← c57284] at e_922_19
  have hr := e_922_19
  simp only [e_922_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7348)) := by
  have c57285 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.1
  have c57286 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.1
  have c57287 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.1
  have c57288 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.1
  have c57289 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.1
  have c57290 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_0 := arithEq_of_rows h (row := 923) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_0
  simp only [← c57285, ← c57286, ← c57287] at e_923_0
  have e_923_1 := arithEq_of_rows h (row := 923) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_1
  simp only [← c57288, ← c57289, k2, ← c57290] at e_923_1
  have hr := e_923_1
  simp only [e_923_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7349)) := by
  have c57291 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57292 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57293 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57294 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57295 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57296 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_2 := arithEq_of_rows h (row := 923) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_2
  simp only [← c57291, ← c57292, ← c57293] at e_923_2
  have e_923_3 := arithEq_of_rows h (row := 923) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_3
  simp only [← c57294, ← c57295, k2, ← c57296] at e_923_3
  have hr := e_923_3
  simp only [e_923_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7350)) := by
  have c57297 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57298 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57299 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57300 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57301 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57302 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_4 := arithEq_of_rows h (row := 923) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_4
  simp only [← c57297, ← c57298, ← c57299] at e_923_4
  have e_923_5 := arithEq_of_rows h (row := 923) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_5
  simp only [← c57300, ← c57301, k2, ← c57302] at e_923_5
  have hr := e_923_5
  simp only [e_923_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7351)) := by
  have c57303 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57304 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57305 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57306 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57307 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57308 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_6 := arithEq_of_rows h (row := 923) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_6
  simp only [← c57303, ← c57304, ← c57305] at e_923_6
  have e_923_7 := arithEq_of_rows h (row := 923) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_7
  simp only [← c57306, ← c57307, k2, ← c57308] at e_923_7
  have hr := e_923_7
  simp only [e_923_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7352)) := by
  have c57309 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57310 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57311 := (publicBatchWrapper86_copies1790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57312 := (publicBatchWrapper86_copies1791 a h).1
  have c57313 := (publicBatchWrapper86_copies1791 a h).2.1
  have c57314 := (publicBatchWrapper86_copies1791 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_8 := arithEq_of_rows h (row := 923) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_8
  simp only [← c57309, ← c57310, ← c57311] at e_923_8
  have e_923_9 := arithEq_of_rows h (row := 923) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_9
  simp only [← c57312, ← c57313, k2, ← c57314] at e_923_9
  have hr := e_923_9
  simp only [e_923_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7353)) := by
  have c57315 := (publicBatchWrapper86_copies1791 a h).2.2.2.1
  have c57316 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.1
  have c57317 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.1
  have c57318 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.1
  have c57319 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.1
  have c57320 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_10 := arithEq_of_rows h (row := 923) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_10
  simp only [← c57315, ← c57316, ← c57317] at e_923_10
  have e_923_11 := arithEq_of_rows h (row := 923) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_11
  simp only [← c57318, ← c57319, k2, ← c57320] at e_923_11
  have hr := e_923_11
  simp only [e_923_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7354)) := by
  have c57321 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.1
  have c57322 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57323 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57324 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57325 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57326 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_12 := arithEq_of_rows h (row := 923) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_12
  simp only [← c57321, ← c57322, ← c57323] at e_923_12
  have e_923_13 := arithEq_of_rows h (row := 923) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_13
  simp only [← c57324, ← c57325, k2, ← c57326] at e_923_13
  have hr := e_923_13
  simp only [e_923_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7355)) := by
  have c57327 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57328 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57329 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57330 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57331 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57332 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_14 := arithEq_of_rows h (row := 923) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_14
  simp only [← c57327, ← c57328, ← c57329] at e_923_14
  have e_923_15 := arithEq_of_rows h (row := 923) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_15
  simp only [← c57330, ← c57331, k2, ← c57332] at e_923_15
  have hr := e_923_15
  simp only [e_923_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7356)) := by
  have c57333 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57334 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57335 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57336 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57337 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57338 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_16 := arithEq_of_rows h (row := 923) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_16
  simp only [← c57333, ← c57334, ← c57335] at e_923_16
  have e_923_17 := arithEq_of_rows h (row := 923) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_17
  simp only [← c57336, ← c57337, k2, ← c57338] at e_923_17
  have hr := e_923_17
  simp only [e_923_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 923 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7357)) := by
  have c57339 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57340 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57341 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57342 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57343 := (publicBatchWrapper86_copies1791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57344 := (publicBatchWrapper86_copies1792 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_923_18 := arithEq_of_rows h (row := 923) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_18
  simp only [← c57339, ← c57340, ← c57341] at e_923_18
  have e_923_19 := arithEq_of_rows h (row := 923) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_923_19
  simp only [← c57342, ← c57343, k2, ← c57344] at e_923_19
  have hr := e_923_19
  simp only [e_923_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7358)) := by
  have c57345 := (publicBatchWrapper86_copies1792 a h).2.1
  have c57346 := (publicBatchWrapper86_copies1792 a h).2.2.1
  have c57347 := (publicBatchWrapper86_copies1792 a h).2.2.2.1
  have c57348 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.1
  have c57349 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.1
  have c57350 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_0 := arithEq_of_rows h (row := 924) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_0
  simp only [← c57345, ← c57346, ← c57347] at e_924_0
  have e_924_1 := arithEq_of_rows h (row := 924) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_1
  simp only [← c57348, ← c57349, k2, ← c57350] at e_924_1
  have hr := e_924_1
  simp only [e_924_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7359)) := by
  have c57351 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.1
  have c57352 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.1
  have c57353 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.1
  have c57354 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57355 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57356 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_2 := arithEq_of_rows h (row := 924) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_2
  simp only [← c57351, ← c57352, ← c57353] at e_924_2
  have e_924_3 := arithEq_of_rows h (row := 924) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_3
  simp only [← c57354, ← c57355, k2, ← c57356] at e_924_3
  have hr := e_924_3
  simp only [e_924_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7360)) := by
  have c57357 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57358 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57359 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57360 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57361 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57362 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_4 := arithEq_of_rows h (row := 924) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_4
  simp only [← c57357, ← c57358, ← c57359] at e_924_4
  have e_924_5 := arithEq_of_rows h (row := 924) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_5
  simp only [← c57360, ← c57361, k2, ← c57362] at e_924_5
  have hr := e_924_5
  simp only [e_924_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7361)) := by
  have c57363 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57364 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57365 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57366 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57367 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57368 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_6 := arithEq_of_rows h (row := 924) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_6
  simp only [← c57363, ← c57364, ← c57365] at e_924_6
  have e_924_7 := arithEq_of_rows h (row := 924) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_7
  simp only [← c57366, ← c57367, k2, ← c57368] at e_924_7
  have hr := e_924_7
  simp only [e_924_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7362)) := by
  have c57369 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57370 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57371 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57372 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57373 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57374 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_8 := arithEq_of_rows h (row := 924) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_8
  simp only [← c57369, ← c57370, ← c57371] at e_924_8
  have e_924_9 := arithEq_of_rows h (row := 924) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_9
  simp only [← c57372, ← c57373, k2, ← c57374] at e_924_9
  have hr := e_924_9
  simp only [e_924_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7363)) := by
  have c57375 := (publicBatchWrapper86_copies1792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57376 := (publicBatchWrapper86_copies1793 a h).1
  have c57377 := (publicBatchWrapper86_copies1793 a h).2.1
  have c57378 := (publicBatchWrapper86_copies1793 a h).2.2.1
  have c57379 := (publicBatchWrapper86_copies1793 a h).2.2.2.1
  have c57380 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_10 := arithEq_of_rows h (row := 924) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_10
  simp only [← c57375, ← c57376, ← c57377] at e_924_10
  have e_924_11 := arithEq_of_rows h (row := 924) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_11
  simp only [← c57378, ← c57379, k2, ← c57380] at e_924_11
  have hr := e_924_11
  simp only [e_924_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7364)) := by
  have c57381 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.1
  have c57382 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.1
  have c57383 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.1
  have c57384 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.1
  have c57385 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.1
  have c57386 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_12 := arithEq_of_rows h (row := 924) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_12
  simp only [← c57381, ← c57382, ← c57383] at e_924_12
  have e_924_13 := arithEq_of_rows h (row := 924) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_13
  simp only [← c57384, ← c57385, k2, ← c57386] at e_924_13
  have hr := e_924_13
  simp only [e_924_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7365)) := by
  have c57387 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57388 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57389 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57390 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57391 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57392 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_14 := arithEq_of_rows h (row := 924) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_14
  simp only [← c57387, ← c57388, ← c57389] at e_924_14
  have e_924_15 := arithEq_of_rows h (row := 924) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_15
  simp only [← c57390, ← c57391, k2, ← c57392] at e_924_15
  have hr := e_924_15
  simp only [e_924_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7366)) := by
  have c57393 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57394 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57395 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57396 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57397 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57398 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_16 := arithEq_of_rows h (row := 924) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_16
  simp only [← c57393, ← c57394, ← c57395] at e_924_16
  have e_924_17 := arithEq_of_rows h (row := 924) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_17
  simp only [← c57396, ← c57397, k2, ← c57398] at e_924_17
  have hr := e_924_17
  simp only [e_924_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 924 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7367)) := by
  have c57399 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57400 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57401 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57402 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57403 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57404 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_924_18 := arithEq_of_rows h (row := 924) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_18
  simp only [← c57399, ← c57400, ← c57401] at e_924_18
  have e_924_19 := arithEq_of_rows h (row := 924) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_924_19
  simp only [← c57402, ← c57403, k2, ← c57404] at e_924_19
  have hr := e_924_19
  simp only [e_924_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7368)) := by
  have c57405 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57406 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57407 := (publicBatchWrapper86_copies1793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57408 := (publicBatchWrapper86_copies1794 a h).1
  have c57409 := (publicBatchWrapper86_copies1794 a h).2.1
  have c57410 := (publicBatchWrapper86_copies1794 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_0 := arithEq_of_rows h (row := 925) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_0
  simp only [← c57405, ← c57406, ← c57407] at e_925_0
  have e_925_1 := arithEq_of_rows h (row := 925) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_1
  simp only [← c57408, ← c57409, k2, ← c57410] at e_925_1
  have hr := e_925_1
  simp only [e_925_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7369)) := by
  have c57411 := (publicBatchWrapper86_copies1794 a h).2.2.2.1
  have c57412 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.1
  have c57413 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.1
  have c57414 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.1
  have c57415 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.1
  have c57416 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_2 := arithEq_of_rows h (row := 925) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_2
  simp only [← c57411, ← c57412, ← c57413] at e_925_2
  have e_925_3 := arithEq_of_rows h (row := 925) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_3
  simp only [← c57414, ← c57415, k2, ← c57416] at e_925_3
  have hr := e_925_3
  simp only [e_925_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7410)) := by
  have c57417 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.1
  have c57418 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57419 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57420 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57421 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57422 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_4 := arithEq_of_rows h (row := 925) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_4
  simp only [← c57417, ← c57418, ← c57419] at e_925_4
  have e_925_5 := arithEq_of_rows h (row := 925) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_5
  simp only [← c57420, ← c57421, k2, ← c57422] at e_925_5
  have hr := e_925_5
  simp only [e_925_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7411)) := by
  have c57423 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57424 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57425 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57426 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57427 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57428 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_6 := arithEq_of_rows h (row := 925) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_6
  simp only [← c57423, ← c57424, ← c57425] at e_925_6
  have e_925_7 := arithEq_of_rows h (row := 925) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_7
  simp only [← c57426, ← c57427, k2, ← c57428] at e_925_7
  have hr := e_925_7
  simp only [e_925_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7412)) := by
  have c57429 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57430 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57431 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57432 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57433 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57434 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_8 := arithEq_of_rows h (row := 925) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_8
  simp only [← c57429, ← c57430, ← c57431] at e_925_8
  have e_925_9 := arithEq_of_rows h (row := 925) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_9
  simp only [← c57432, ← c57433, k2, ← c57434] at e_925_9
  have hr := e_925_9
  simp only [e_925_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7413)) := by
  have c57435 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57436 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57437 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57438 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57439 := (publicBatchWrapper86_copies1794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57440 := (publicBatchWrapper86_copies1795 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_10 := arithEq_of_rows h (row := 925) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_10
  simp only [← c57435, ← c57436, ← c57437] at e_925_10
  have e_925_11 := arithEq_of_rows h (row := 925) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_11
  simp only [← c57438, ← c57439, k2, ← c57440] at e_925_11
  have hr := e_925_11
  simp only [e_925_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7414)) := by
  have c57441 := (publicBatchWrapper86_copies1795 a h).2.1
  have c57442 := (publicBatchWrapper86_copies1795 a h).2.2.1
  have c57443 := (publicBatchWrapper86_copies1795 a h).2.2.2.1
  have c57444 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.1
  have c57445 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.1
  have c57446 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_12 := arithEq_of_rows h (row := 925) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_12
  simp only [← c57441, ← c57442, ← c57443] at e_925_12
  have e_925_13 := arithEq_of_rows h (row := 925) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_13
  simp only [← c57444, ← c57445, k2, ← c57446] at e_925_13
  have hr := e_925_13
  simp only [e_925_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7415)) := by
  have c57447 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.1
  have c57448 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.1
  have c57449 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.1
  have c57450 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57451 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57452 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_14 := arithEq_of_rows h (row := 925) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_14
  simp only [← c57447, ← c57448, ← c57449] at e_925_14
  have e_925_15 := arithEq_of_rows h (row := 925) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_15
  simp only [← c57450, ← c57451, k2, ← c57452] at e_925_15
  have hr := e_925_15
  simp only [e_925_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7416)) := by
  have c57453 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57454 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57455 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57456 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57457 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57458 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_16 := arithEq_of_rows h (row := 925) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_16
  simp only [← c57453, ← c57454, ← c57455] at e_925_16
  have e_925_17 := arithEq_of_rows h (row := 925) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_17
  simp only [← c57456, ← c57457, k2, ← c57458] at e_925_17
  have hr := e_925_17
  simp only [e_925_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 925 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7417)) := by
  have c57459 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57460 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57461 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57462 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57463 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57464 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_925_18 := arithEq_of_rows h (row := 925) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_18
  simp only [← c57459, ← c57460, ← c57461] at e_925_18
  have e_925_19 := arithEq_of_rows h (row := 925) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_925_19
  simp only [← c57462, ← c57463, k2, ← c57464] at e_925_19
  have hr := e_925_19
  simp only [e_925_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7418)) := by
  have c57465 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57466 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57467 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57468 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57469 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57470 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_0 := arithEq_of_rows h (row := 926) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_0
  simp only [← c57465, ← c57466, ← c57467] at e_926_0
  have e_926_1 := arithEq_of_rows h (row := 926) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_1
  simp only [← c57468, ← c57469, k2, ← c57470] at e_926_1
  have hr := e_926_1
  simp only [e_926_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7419)) := by
  have c57471 := (publicBatchWrapper86_copies1795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57472 := (publicBatchWrapper86_copies1796 a h).1
  have c57473 := (publicBatchWrapper86_copies1796 a h).2.1
  have c57474 := (publicBatchWrapper86_copies1796 a h).2.2.1
  have c57475 := (publicBatchWrapper86_copies1796 a h).2.2.2.1
  have c57476 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_2 := arithEq_of_rows h (row := 926) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_2
  simp only [← c57471, ← c57472, ← c57473] at e_926_2
  have e_926_3 := arithEq_of_rows h (row := 926) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_3
  simp only [← c57474, ← c57475, k2, ← c57476] at e_926_3
  have hr := e_926_3
  simp only [e_926_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7420)) := by
  have c57477 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.1
  have c57478 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.1
  have c57479 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.1
  have c57480 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.1
  have c57481 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.1
  have c57482 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_4 := arithEq_of_rows h (row := 926) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_4
  simp only [← c57477, ← c57478, ← c57479] at e_926_4
  have e_926_5 := arithEq_of_rows h (row := 926) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_5
  simp only [← c57480, ← c57481, k2, ← c57482] at e_926_5
  have hr := e_926_5
  simp only [e_926_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7421)) := by
  have c57483 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57484 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57485 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57486 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57487 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57488 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_6 := arithEq_of_rows h (row := 926) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_6
  simp only [← c57483, ← c57484, ← c57485] at e_926_6
  have e_926_7 := arithEq_of_rows h (row := 926) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_7
  simp only [← c57486, ← c57487, k2, ← c57488] at e_926_7
  have hr := e_926_7
  simp only [e_926_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7422)) := by
  have c57489 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57490 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57491 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57492 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57493 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57494 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_8 := arithEq_of_rows h (row := 926) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_8
  simp only [← c57489, ← c57490, ← c57491] at e_926_8
  have e_926_9 := arithEq_of_rows h (row := 926) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_9
  simp only [← c57492, ← c57493, k2, ← c57494] at e_926_9
  have hr := e_926_9
  simp only [e_926_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7423)) := by
  have c57495 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57496 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57497 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57498 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57499 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57500 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_10 := arithEq_of_rows h (row := 926) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_10
  simp only [← c57495, ← c57496, ← c57497] at e_926_10
  have e_926_11 := arithEq_of_rows h (row := 926) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_11
  simp only [← c57498, ← c57499, k2, ← c57500] at e_926_11
  have hr := e_926_11
  simp only [e_926_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7424)) := by
  have c57501 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57502 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57503 := (publicBatchWrapper86_copies1796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57504 := (publicBatchWrapper86_copies1797 a h).1
  have c57505 := (publicBatchWrapper86_copies1797 a h).2.1
  have c57506 := (publicBatchWrapper86_copies1797 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_12 := arithEq_of_rows h (row := 926) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_12
  simp only [← c57501, ← c57502, ← c57503] at e_926_12
  have e_926_13 := arithEq_of_rows h (row := 926) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_13
  simp only [← c57504, ← c57505, k2, ← c57506] at e_926_13
  have hr := e_926_13
  simp only [e_926_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7425)) := by
  have c57507 := (publicBatchWrapper86_copies1797 a h).2.2.2.1
  have c57508 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.1
  have c57509 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.1
  have c57510 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.1
  have c57511 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.1
  have c57512 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_14 := arithEq_of_rows h (row := 926) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_14
  simp only [← c57507, ← c57508, ← c57509] at e_926_14
  have e_926_15 := arithEq_of_rows h (row := 926) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_15
  simp only [← c57510, ← c57511, k2, ← c57512] at e_926_15
  have hr := e_926_15
  simp only [e_926_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7426)) := by
  have c57513 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.1
  have c57514 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57515 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57516 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57517 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57518 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_16 := arithEq_of_rows h (row := 926) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_16
  simp only [← c57513, ← c57514, ← c57515] at e_926_16
  have e_926_17 := arithEq_of_rows h (row := 926) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_17
  simp only [← c57516, ← c57517, k2, ← c57518] at e_926_17
  have hr := e_926_17
  simp only [e_926_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 926 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7427)) := by
  have c57519 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57520 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57521 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57522 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57523 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57524 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_926_18 := arithEq_of_rows h (row := 926) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_18
  simp only [← c57519, ← c57520, ← c57521] at e_926_18
  have e_926_19 := arithEq_of_rows h (row := 926) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_926_19
  simp only [← c57522, ← c57523, k2, ← c57524] at e_926_19
  have hr := e_926_19
  simp only [e_926_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7428)) := by
  have c57525 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57526 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57527 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57528 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57529 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57530 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_0 := arithEq_of_rows h (row := 927) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_0
  simp only [← c57525, ← c57526, ← c57527] at e_927_0
  have e_927_1 := arithEq_of_rows h (row := 927) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_1
  simp only [← c57528, ← c57529, k2, ← c57530] at e_927_1
  have hr := e_927_1
  simp only [e_927_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7429)) := by
  have c57531 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57532 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57533 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57534 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57535 := (publicBatchWrapper86_copies1797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57536 := (publicBatchWrapper86_copies1798 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_2 := arithEq_of_rows h (row := 927) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_2
  simp only [← c57531, ← c57532, ← c57533] at e_927_2
  have e_927_3 := arithEq_of_rows h (row := 927) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_3
  simp only [← c57534, ← c57535, k2, ← c57536] at e_927_3
  have hr := e_927_3
  simp only [e_927_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7430)) := by
  have c57537 := (publicBatchWrapper86_copies1798 a h).2.1
  have c57538 := (publicBatchWrapper86_copies1798 a h).2.2.1
  have c57539 := (publicBatchWrapper86_copies1798 a h).2.2.2.1
  have c57540 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.1
  have c57541 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.1
  have c57542 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_4 := arithEq_of_rows h (row := 927) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_4
  simp only [← c57537, ← c57538, ← c57539] at e_927_4
  have e_927_5 := arithEq_of_rows h (row := 927) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_5
  simp only [← c57540, ← c57541, k2, ← c57542] at e_927_5
  have hr := e_927_5
  simp only [e_927_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7431)) := by
  have c57543 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.1
  have c57544 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.1
  have c57545 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.1
  have c57546 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57547 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57548 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_6 := arithEq_of_rows h (row := 927) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_6
  simp only [← c57543, ← c57544, ← c57545] at e_927_6
  have e_927_7 := arithEq_of_rows h (row := 927) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_7
  simp only [← c57546, ← c57547, k2, ← c57548] at e_927_7
  have hr := e_927_7
  simp only [e_927_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7432)) := by
  have c57549 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57550 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57551 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57552 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57553 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57554 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_8 := arithEq_of_rows h (row := 927) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_8
  simp only [← c57549, ← c57550, ← c57551] at e_927_8
  have e_927_9 := arithEq_of_rows h (row := 927) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_9
  simp only [← c57552, ← c57553, k2, ← c57554] at e_927_9
  have hr := e_927_9
  simp only [e_927_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7433)) := by
  have c57555 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57556 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57557 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57558 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57559 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57560 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_10 := arithEq_of_rows h (row := 927) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_10
  simp only [← c57555, ← c57556, ← c57557] at e_927_10
  have e_927_11 := arithEq_of_rows h (row := 927) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_11
  simp only [← c57558, ← c57559, k2, ← c57560] at e_927_11
  have hr := e_927_11
  simp only [e_927_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7434)) := by
  have c57561 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57562 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57563 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57564 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57565 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57566 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_12 := arithEq_of_rows h (row := 927) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_12
  simp only [← c57561, ← c57562, ← c57563] at e_927_12
  have e_927_13 := arithEq_of_rows h (row := 927) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_13
  simp only [← c57564, ← c57565, k2, ← c57566] at e_927_13
  have hr := e_927_13
  simp only [e_927_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7435)) := by
  have c57567 := (publicBatchWrapper86_copies1798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57568 := (publicBatchWrapper86_copies1799 a h).1
  have c57569 := (publicBatchWrapper86_copies1799 a h).2.1
  have c57570 := (publicBatchWrapper86_copies1799 a h).2.2.1
  have c57571 := (publicBatchWrapper86_copies1799 a h).2.2.2.1
  have c57572 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_14 := arithEq_of_rows h (row := 927) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_14
  simp only [← c57567, ← c57568, ← c57569] at e_927_14
  have e_927_15 := arithEq_of_rows h (row := 927) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_15
  simp only [← c57570, ← c57571, k2, ← c57572] at e_927_15
  have hr := e_927_15
  simp only [e_927_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7436)) := by
  have c57573 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.1
  have c57574 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.1
  have c57575 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.1
  have c57576 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.1
  have c57577 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.1
  have c57578 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_16 := arithEq_of_rows h (row := 927) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_16
  simp only [← c57573, ← c57574, ← c57575] at e_927_16
  have e_927_17 := arithEq_of_rows h (row := 927) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_17
  simp only [← c57576, ← c57577, k2, ← c57578] at e_927_17
  have hr := e_927_17
  simp only [e_927_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 927 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7437)) := by
  have c57579 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57580 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57581 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57582 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57583 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57584 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_927_18 := arithEq_of_rows h (row := 927) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_18
  simp only [← c57579, ← c57580, ← c57581] at e_927_18
  have e_927_19 := arithEq_of_rows h (row := 927) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_927_19
  simp only [← c57582, ← c57583, k2, ← c57584] at e_927_19
  have hr := e_927_19
  simp only [e_927_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7438)) := by
  have c57585 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57586 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57587 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57588 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57589 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57590 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_0 := arithEq_of_rows h (row := 928) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_0
  simp only [← c57585, ← c57586, ← c57587] at e_928_0
  have e_928_1 := arithEq_of_rows h (row := 928) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_1
  simp only [← c57588, ← c57589, k2, ← c57590] at e_928_1
  have hr := e_928_1
  simp only [e_928_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7439)) := by
  have c57591 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57592 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57593 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57594 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57595 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57596 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_2 := arithEq_of_rows h (row := 928) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_2
  simp only [← c57591, ← c57592, ← c57593] at e_928_2
  have e_928_3 := arithEq_of_rows h (row := 928) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_3
  simp only [← c57594, ← c57595, k2, ← c57596] at e_928_3
  have hr := e_928_3
  simp only [e_928_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7440)) := by
  have c57597 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57598 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57599 := (publicBatchWrapper86_copies1799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57600 := (publicBatchWrapper86_copies1800 a h).1
  have c57601 := (publicBatchWrapper86_copies1800 a h).2.1
  have c57602 := (publicBatchWrapper86_copies1800 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_4 := arithEq_of_rows h (row := 928) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_4
  simp only [← c57597, ← c57598, ← c57599] at e_928_4
  have e_928_5 := arithEq_of_rows h (row := 928) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_5
  simp only [← c57600, ← c57601, k2, ← c57602] at e_928_5
  have hr := e_928_5
  simp only [e_928_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7441)) := by
  have c57603 := (publicBatchWrapper86_copies1800 a h).2.2.2.1
  have c57604 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.1
  have c57605 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.1
  have c57606 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.1
  have c57607 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.1
  have c57608 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_6 := arithEq_of_rows h (row := 928) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_6
  simp only [← c57603, ← c57604, ← c57605] at e_928_6
  have e_928_7 := arithEq_of_rows h (row := 928) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_7
  simp only [← c57606, ← c57607, k2, ← c57608] at e_928_7
  have hr := e_928_7
  simp only [e_928_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7442)) := by
  have c57609 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.1
  have c57610 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57611 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57612 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57613 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57614 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_8 := arithEq_of_rows h (row := 928) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_8
  simp only [← c57609, ← c57610, ← c57611] at e_928_8
  have e_928_9 := arithEq_of_rows h (row := 928) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_9
  simp only [← c57612, ← c57613, k2, ← c57614] at e_928_9
  have hr := e_928_9
  simp only [e_928_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7443)) := by
  have c57615 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57616 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57617 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57618 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57619 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57620 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_10 := arithEq_of_rows h (row := 928) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_10
  simp only [← c57615, ← c57616, ← c57617] at e_928_10
  have e_928_11 := arithEq_of_rows h (row := 928) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_11
  simp only [← c57618, ← c57619, k2, ← c57620] at e_928_11
  have hr := e_928_11
  simp only [e_928_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7444)) := by
  have c57621 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57622 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57623 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57624 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57625 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57626 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_12 := arithEq_of_rows h (row := 928) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_12
  simp only [← c57621, ← c57622, ← c57623] at e_928_12
  have e_928_13 := arithEq_of_rows h (row := 928) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_13
  simp only [← c57624, ← c57625, k2, ← c57626] at e_928_13
  have hr := e_928_13
  simp only [e_928_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7445)) := by
  have c57627 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57628 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57629 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57630 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57631 := (publicBatchWrapper86_copies1800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57632 := (publicBatchWrapper86_copies1801 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_14 := arithEq_of_rows h (row := 928) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_14
  simp only [← c57627, ← c57628, ← c57629] at e_928_14
  have e_928_15 := arithEq_of_rows h (row := 928) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_15
  simp only [← c57630, ← c57631, k2, ← c57632] at e_928_15
  have hr := e_928_15
  simp only [e_928_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7446)) := by
  have c57633 := (publicBatchWrapper86_copies1801 a h).2.1
  have c57634 := (publicBatchWrapper86_copies1801 a h).2.2.1
  have c57635 := (publicBatchWrapper86_copies1801 a h).2.2.2.1
  have c57636 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.1
  have c57637 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.1
  have c57638 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_16 := arithEq_of_rows h (row := 928) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_16
  simp only [← c57633, ← c57634, ← c57635] at e_928_16
  have e_928_17 := arithEq_of_rows h (row := 928) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_17
  simp only [← c57636, ← c57637, k2, ← c57638] at e_928_17
  have hr := e_928_17
  simp only [e_928_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 928 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7447)) := by
  have c57639 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.1
  have c57640 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.1
  have c57641 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.1
  have c57642 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57643 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57644 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_928_18 := arithEq_of_rows h (row := 928) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_18
  simp only [← c57639, ← c57640, ← c57641] at e_928_18
  have e_928_19 := arithEq_of_rows h (row := 928) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_928_19
  simp only [← c57642, ← c57643, k2, ← c57644] at e_928_19
  have hr := e_928_19
  simp only [e_928_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7448)) := by
  have c57645 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57646 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57647 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57648 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57649 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57650 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_0 := arithEq_of_rows h (row := 929) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_0
  simp only [← c57645, ← c57646, ← c57647] at e_929_0
  have e_929_1 := arithEq_of_rows h (row := 929) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_1
  simp only [← c57648, ← c57649, k2, ← c57650] at e_929_1
  have hr := e_929_1
  simp only [e_929_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7449)) := by
  have c57651 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57652 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57653 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57654 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57655 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57656 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_2 := arithEq_of_rows h (row := 929) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_2
  simp only [← c57651, ← c57652, ← c57653] at e_929_2
  have e_929_3 := arithEq_of_rows h (row := 929) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_3
  simp only [← c57654, ← c57655, k2, ← c57656] at e_929_3
  have hr := e_929_3
  simp only [e_929_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7450)) := by
  have c57657 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57658 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57659 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57660 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57661 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57662 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_4 := arithEq_of_rows h (row := 929) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_4
  simp only [← c57657, ← c57658, ← c57659] at e_929_4
  have e_929_5 := arithEq_of_rows h (row := 929) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_5
  simp only [← c57660, ← c57661, k2, ← c57662] at e_929_5
  have hr := e_929_5
  simp only [e_929_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7451)) := by
  have c57663 := (publicBatchWrapper86_copies1801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57664 := (publicBatchWrapper86_copies1802 a h).1
  have c57665 := (publicBatchWrapper86_copies1802 a h).2.1
  have c57666 := (publicBatchWrapper86_copies1802 a h).2.2.1
  have c57667 := (publicBatchWrapper86_copies1802 a h).2.2.2.1
  have c57668 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_6 := arithEq_of_rows h (row := 929) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_6
  simp only [← c57663, ← c57664, ← c57665] at e_929_6
  have e_929_7 := arithEq_of_rows h (row := 929) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_7
  simp only [← c57666, ← c57667, k2, ← c57668] at e_929_7
  have hr := e_929_7
  simp only [e_929_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7452)) := by
  have c57669 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.1
  have c57670 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.1
  have c57671 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.1
  have c57672 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.1
  have c57673 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.1
  have c57674 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_8 := arithEq_of_rows h (row := 929) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_8
  simp only [← c57669, ← c57670, ← c57671] at e_929_8
  have e_929_9 := arithEq_of_rows h (row := 929) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_9
  simp only [← c57672, ← c57673, k2, ← c57674] at e_929_9
  have hr := e_929_9
  simp only [e_929_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7453)) := by
  have c57675 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57676 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57677 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57678 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57679 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57680 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_10 := arithEq_of_rows h (row := 929) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_10
  simp only [← c57675, ← c57676, ← c57677] at e_929_10
  have e_929_11 := arithEq_of_rows h (row := 929) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_11
  simp only [← c57678, ← c57679, k2, ← c57680] at e_929_11
  have hr := e_929_11
  simp only [e_929_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7454)) := by
  have c57681 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57682 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57683 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57684 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57685 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57686 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_12 := arithEq_of_rows h (row := 929) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_12
  simp only [← c57681, ← c57682, ← c57683] at e_929_12
  have e_929_13 := arithEq_of_rows h (row := 929) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_13
  simp only [← c57684, ← c57685, k2, ← c57686] at e_929_13
  have hr := e_929_13
  simp only [e_929_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7455)) := by
  have c57687 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57688 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57689 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57690 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57691 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57692 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_14 := arithEq_of_rows h (row := 929) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_14
  simp only [← c57687, ← c57688, ← c57689] at e_929_14
  have e_929_15 := arithEq_of_rows h (row := 929) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_15
  simp only [← c57690, ← c57691, k2, ← c57692] at e_929_15
  have hr := e_929_15
  simp only [e_929_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7456)) := by
  have c57693 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57694 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57695 := (publicBatchWrapper86_copies1802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57696 := (publicBatchWrapper86_copies1803 a h).1
  have c57697 := (publicBatchWrapper86_copies1803 a h).2.1
  have c57698 := (publicBatchWrapper86_copies1803 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_16 := arithEq_of_rows h (row := 929) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_16
  simp only [← c57693, ← c57694, ← c57695] at e_929_16
  have e_929_17 := arithEq_of_rows h (row := 929) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_17
  simp only [← c57696, ← c57697, k2, ← c57698] at e_929_17
  have hr := e_929_17
  simp only [e_929_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 929 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7457)) := by
  have c57699 := (publicBatchWrapper86_copies1803 a h).2.2.2.1
  have c57700 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.1
  have c57701 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.1
  have c57702 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.1
  have c57703 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.1
  have c57704 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_929_18 := arithEq_of_rows h (row := 929) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_18
  simp only [← c57699, ← c57700, ← c57701] at e_929_18
  have e_929_19 := arithEq_of_rows h (row := 929) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_929_19
  simp only [← c57702, ← c57703, k2, ← c57704] at e_929_19
  have hr := e_929_19
  simp only [e_929_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7458)) := by
  have c57705 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.1
  have c57706 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57707 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57708 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57709 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57710 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_0 := arithEq_of_rows h (row := 930) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_0
  simp only [← c57705, ← c57706, ← c57707] at e_930_0
  have e_930_1 := arithEq_of_rows h (row := 930) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_1
  simp only [← c57708, ← c57709, k2, ← c57710] at e_930_1
  have hr := e_930_1
  simp only [e_930_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7459)) := by
  have c57711 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57712 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57713 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57714 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57715 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57716 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_2 := arithEq_of_rows h (row := 930) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_2
  simp only [← c57711, ← c57712, ← c57713] at e_930_2
  have e_930_3 := arithEq_of_rows h (row := 930) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_3
  simp only [← c57714, ← c57715, k2, ← c57716] at e_930_3
  have hr := e_930_3
  simp only [e_930_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7460)) := by
  have c57717 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57718 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57719 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57720 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57721 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57722 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_4 := arithEq_of_rows h (row := 930) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_4
  simp only [← c57717, ← c57718, ← c57719] at e_930_4
  have e_930_5 := arithEq_of_rows h (row := 930) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_5
  simp only [← c57720, ← c57721, k2, ← c57722] at e_930_5
  have hr := e_930_5
  simp only [e_930_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7461)) := by
  have c57723 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57724 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57725 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57726 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57727 := (publicBatchWrapper86_copies1803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57728 := (publicBatchWrapper86_copies1804 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_6 := arithEq_of_rows h (row := 930) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_6
  simp only [← c57723, ← c57724, ← c57725] at e_930_6
  have e_930_7 := arithEq_of_rows h (row := 930) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_7
  simp only [← c57726, ← c57727, k2, ← c57728] at e_930_7
  have hr := e_930_7
  simp only [e_930_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7462)) := by
  have c57729 := (publicBatchWrapper86_copies1804 a h).2.1
  have c57730 := (publicBatchWrapper86_copies1804 a h).2.2.1
  have c57731 := (publicBatchWrapper86_copies1804 a h).2.2.2.1
  have c57732 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.1
  have c57733 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.1
  have c57734 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_8 := arithEq_of_rows h (row := 930) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_8
  simp only [← c57729, ← c57730, ← c57731] at e_930_8
  have e_930_9 := arithEq_of_rows h (row := 930) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_9
  simp only [← c57732, ← c57733, k2, ← c57734] at e_930_9
  have hr := e_930_9
  simp only [e_930_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7463)) := by
  have c57735 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.1
  have c57736 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.1
  have c57737 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.1
  have c57738 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57739 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57740 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_10 := arithEq_of_rows h (row := 930) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_10
  simp only [← c57735, ← c57736, ← c57737] at e_930_10
  have e_930_11 := arithEq_of_rows h (row := 930) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_11
  simp only [← c57738, ← c57739, k2, ← c57740] at e_930_11
  have hr := e_930_11
  simp only [e_930_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7464)) := by
  have c57741 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57742 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57743 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57744 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57745 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57746 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_12 := arithEq_of_rows h (row := 930) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_12
  simp only [← c57741, ← c57742, ← c57743] at e_930_12
  have e_930_13 := arithEq_of_rows h (row := 930) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_13
  simp only [← c57744, ← c57745, k2, ← c57746] at e_930_13
  have hr := e_930_13
  simp only [e_930_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7465)) := by
  have c57747 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57748 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57749 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57750 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57751 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57752 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_14 := arithEq_of_rows h (row := 930) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_14
  simp only [← c57747, ← c57748, ← c57749] at e_930_14
  have e_930_15 := arithEq_of_rows h (row := 930) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_15
  simp only [← c57750, ← c57751, k2, ← c57752] at e_930_15
  have hr := e_930_15
  simp only [e_930_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7466)) := by
  have c57753 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57754 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57755 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57756 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57757 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57758 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_16 := arithEq_of_rows h (row := 930) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_16
  simp only [← c57753, ← c57754, ← c57755] at e_930_16
  have e_930_17 := arithEq_of_rows h (row := 930) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_17
  simp only [← c57756, ← c57757, k2, ← c57758] at e_930_17
  have hr := e_930_17
  simp only [e_930_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 930 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7467)) := by
  have c57759 := (publicBatchWrapper86_copies1804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57760 := (publicBatchWrapper86_copies1805 a h).1
  have c57761 := (publicBatchWrapper86_copies1805 a h).2.1
  have c57762 := (publicBatchWrapper86_copies1805 a h).2.2.1
  have c57763 := (publicBatchWrapper86_copies1805 a h).2.2.2.1
  have c57764 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_930_18 := arithEq_of_rows h (row := 930) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_18
  simp only [← c57759, ← c57760, ← c57761] at e_930_18
  have e_930_19 := arithEq_of_rows h (row := 930) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_930_19
  simp only [← c57762, ← c57763, k2, ← c57764] at e_930_19
  have hr := e_930_19
  simp only [e_930_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7468)) := by
  have c57765 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.1
  have c57766 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.1
  have c57767 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.1
  have c57768 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.1
  have c57769 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.1
  have c57770 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_0 := arithEq_of_rows h (row := 931) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_0
  simp only [← c57765, ← c57766, ← c57767] at e_931_0
  have e_931_1 := arithEq_of_rows h (row := 931) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_1
  simp only [← c57768, ← c57769, k2, ← c57770] at e_931_1
  have hr := e_931_1
  simp only [e_931_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7469)) := by
  have c57771 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57772 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57773 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57774 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57775 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57776 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_2 := arithEq_of_rows h (row := 931) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_2
  simp only [← c57771, ← c57772, ← c57773] at e_931_2
  have e_931_3 := arithEq_of_rows h (row := 931) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_3
  simp only [← c57774, ← c57775, k2, ← c57776] at e_931_3
  have hr := e_931_3
  simp only [e_931_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7470)) := by
  have c57777 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57778 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57779 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57780 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57781 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57782 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_4 := arithEq_of_rows h (row := 931) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_4
  simp only [← c57777, ← c57778, ← c57779] at e_931_4
  have e_931_5 := arithEq_of_rows h (row := 931) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_5
  simp only [← c57780, ← c57781, k2, ← c57782] at e_931_5
  have hr := e_931_5
  simp only [e_931_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7471)) := by
  have c57783 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57784 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57785 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57786 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57787 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57788 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_6 := arithEq_of_rows h (row := 931) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_6
  simp only [← c57783, ← c57784, ← c57785] at e_931_6
  have e_931_7 := arithEq_of_rows h (row := 931) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_7
  simp only [← c57786, ← c57787, k2, ← c57788] at e_931_7
  have hr := e_931_7
  simp only [e_931_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7472)) := by
  have c57789 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57790 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57791 := (publicBatchWrapper86_copies1805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57792 := (publicBatchWrapper86_copies1806 a h).1
  have c57793 := (publicBatchWrapper86_copies1806 a h).2.1
  have c57794 := (publicBatchWrapper86_copies1806 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_8 := arithEq_of_rows h (row := 931) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_8
  simp only [← c57789, ← c57790, ← c57791] at e_931_8
  have e_931_9 := arithEq_of_rows h (row := 931) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_9
  simp only [← c57792, ← c57793, k2, ← c57794] at e_931_9
  have hr := e_931_9
  simp only [e_931_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7473)) := by
  have c57795 := (publicBatchWrapper86_copies1806 a h).2.2.2.1
  have c57796 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.1
  have c57797 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.1
  have c57798 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.1
  have c57799 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.1
  have c57800 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_10 := arithEq_of_rows h (row := 931) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_10
  simp only [← c57795, ← c57796, ← c57797] at e_931_10
  have e_931_11 := arithEq_of_rows h (row := 931) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_11
  simp only [← c57798, ← c57799, k2, ← c57800] at e_931_11
  have hr := e_931_11
  simp only [e_931_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7474)) := by
  have c57801 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.1
  have c57802 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57803 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57804 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57805 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57806 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_12 := arithEq_of_rows h (row := 931) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_12
  simp only [← c57801, ← c57802, ← c57803] at e_931_12
  have e_931_13 := arithEq_of_rows h (row := 931) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_13
  simp only [← c57804, ← c57805, k2, ← c57806] at e_931_13
  have hr := e_931_13
  simp only [e_931_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7475)) := by
  have c57807 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57808 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57809 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57810 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57811 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57812 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_14 := arithEq_of_rows h (row := 931) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_14
  simp only [← c57807, ← c57808, ← c57809] at e_931_14
  have e_931_15 := arithEq_of_rows h (row := 931) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_15
  simp only [← c57810, ← c57811, k2, ← c57812] at e_931_15
  have hr := e_931_15
  simp only [e_931_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7476)) := by
  have c57813 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57814 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57815 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57816 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57817 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57818 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_16 := arithEq_of_rows h (row := 931) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_16
  simp only [← c57813, ← c57814, ← c57815] at e_931_16
  have e_931_17 := arithEq_of_rows h (row := 931) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_17
  simp only [← c57816, ← c57817, k2, ← c57818] at e_931_17
  have hr := e_931_17
  simp only [e_931_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 931 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7477)) := by
  have c57819 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57820 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57821 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57822 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57823 := (publicBatchWrapper86_copies1806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57824 := (publicBatchWrapper86_copies1807 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_931_18 := arithEq_of_rows h (row := 931) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_18
  simp only [← c57819, ← c57820, ← c57821] at e_931_18
  have e_931_19 := arithEq_of_rows h (row := 931) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_931_19
  simp only [← c57822, ← c57823, k2, ← c57824] at e_931_19
  have hr := e_931_19
  simp only [e_931_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7478)) := by
  have c57825 := (publicBatchWrapper86_copies1807 a h).2.1
  have c57826 := (publicBatchWrapper86_copies1807 a h).2.2.1
  have c57827 := (publicBatchWrapper86_copies1807 a h).2.2.2.1
  have c57828 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.1
  have c57829 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.1
  have c57830 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_0 := arithEq_of_rows h (row := 932) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_0
  simp only [← c57825, ← c57826, ← c57827] at e_932_0
  have e_932_1 := arithEq_of_rows h (row := 932) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_1
  simp only [← c57828, ← c57829, k2, ← c57830] at e_932_1
  have hr := e_932_1
  simp only [e_932_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7479)) := by
  have c57831 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.1
  have c57832 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.1
  have c57833 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.1
  have c57834 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57835 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57836 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_2 := arithEq_of_rows h (row := 932) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_2
  simp only [← c57831, ← c57832, ← c57833] at e_932_2
  have e_932_3 := arithEq_of_rows h (row := 932) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_3
  simp only [← c57834, ← c57835, k2, ← c57836] at e_932_3
  have hr := e_932_3
  simp only [e_932_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7480)) := by
  have c57837 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57838 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57839 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57840 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57841 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57842 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_4 := arithEq_of_rows h (row := 932) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_4
  simp only [← c57837, ← c57838, ← c57839] at e_932_4
  have e_932_5 := arithEq_of_rows h (row := 932) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_5
  simp only [← c57840, ← c57841, k2, ← c57842] at e_932_5
  have hr := e_932_5
  simp only [e_932_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7481)) := by
  have c57843 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57844 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57845 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57846 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57847 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57848 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_6 := arithEq_of_rows h (row := 932) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_6
  simp only [← c57843, ← c57844, ← c57845] at e_932_6
  have e_932_7 := arithEq_of_rows h (row := 932) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_7
  simp only [← c57846, ← c57847, k2, ← c57848] at e_932_7
  have hr := e_932_7
  simp only [e_932_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7482)) := by
  have c57849 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57850 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57851 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57852 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57853 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57854 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_8 := arithEq_of_rows h (row := 932) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_8
  simp only [← c57849, ← c57850, ← c57851] at e_932_8
  have e_932_9 := arithEq_of_rows h (row := 932) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_9
  simp only [← c57852, ← c57853, k2, ← c57854] at e_932_9
  have hr := e_932_9
  simp only [e_932_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7483)) := by
  have c57855 := (publicBatchWrapper86_copies1807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57856 := (publicBatchWrapper86_copies1808 a h).1
  have c57857 := (publicBatchWrapper86_copies1808 a h).2.1
  have c57858 := (publicBatchWrapper86_copies1808 a h).2.2.1
  have c57859 := (publicBatchWrapper86_copies1808 a h).2.2.2.1
  have c57860 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_10 := arithEq_of_rows h (row := 932) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_10
  simp only [← c57855, ← c57856, ← c57857] at e_932_10
  have e_932_11 := arithEq_of_rows h (row := 932) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_11
  simp only [← c57858, ← c57859, k2, ← c57860] at e_932_11
  have hr := e_932_11
  simp only [e_932_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7484)) := by
  have c57861 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.1
  have c57862 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.1
  have c57863 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.1
  have c57864 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.1
  have c57865 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.1
  have c57866 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_12 := arithEq_of_rows h (row := 932) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_12
  simp only [← c57861, ← c57862, ← c57863] at e_932_12
  have e_932_13 := arithEq_of_rows h (row := 932) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_13
  simp only [← c57864, ← c57865, k2, ← c57866] at e_932_13
  have hr := e_932_13
  simp only [e_932_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7485)) := by
  have c57867 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57868 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57869 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57870 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57871 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57872 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_14 := arithEq_of_rows h (row := 932) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_14
  simp only [← c57867, ← c57868, ← c57869] at e_932_14
  have e_932_15 := arithEq_of_rows h (row := 932) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_15
  simp only [← c57870, ← c57871, k2, ← c57872] at e_932_15
  have hr := e_932_15
  simp only [e_932_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7486)) := by
  have c57873 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57874 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57875 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57876 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57877 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57878 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_16 := arithEq_of_rows h (row := 932) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_16
  simp only [← c57873, ← c57874, ← c57875] at e_932_16
  have e_932_17 := arithEq_of_rows h (row := 932) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_17
  simp only [← c57876, ← c57877, k2, ← c57878] at e_932_17
  have hr := e_932_17
  simp only [e_932_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 932 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7487)) := by
  have c57879 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57880 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57881 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57882 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57883 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57884 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_932_18 := arithEq_of_rows h (row := 932) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_18
  simp only [← c57879, ← c57880, ← c57881] at e_932_18
  have e_932_19 := arithEq_of_rows h (row := 932) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_932_19
  simp only [← c57882, ← c57883, k2, ← c57884] at e_932_19
  have hr := e_932_19
  simp only [e_932_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7488)) := by
  have c57885 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57886 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57887 := (publicBatchWrapper86_copies1808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57888 := (publicBatchWrapper86_copies1809 a h).1
  have c57889 := (publicBatchWrapper86_copies1809 a h).2.1
  have c57890 := (publicBatchWrapper86_copies1809 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_0 := arithEq_of_rows h (row := 933) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_0
  simp only [← c57885, ← c57886, ← c57887] at e_933_0
  have e_933_1 := arithEq_of_rows h (row := 933) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_1
  simp only [← c57888, ← c57889, k2, ← c57890] at e_933_1
  have hr := e_933_1
  simp only [e_933_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7489)) := by
  have c57891 := (publicBatchWrapper86_copies1809 a h).2.2.2.1
  have c57892 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.1
  have c57893 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.1
  have c57894 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.1
  have c57895 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.1
  have c57896 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_2 := arithEq_of_rows h (row := 933) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_2
  simp only [← c57891, ← c57892, ← c57893] at e_933_2
  have e_933_3 := arithEq_of_rows h (row := 933) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_3
  simp only [← c57894, ← c57895, k2, ← c57896] at e_933_3
  have hr := e_933_3
  simp only [e_933_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7490)) := by
  have c57897 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.1
  have c57898 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57899 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57900 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57901 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57902 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_4 := arithEq_of_rows h (row := 933) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_4
  simp only [← c57897, ← c57898, ← c57899] at e_933_4
  have e_933_5 := arithEq_of_rows h (row := 933) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_5
  simp only [← c57900, ← c57901, k2, ← c57902] at e_933_5
  have hr := e_933_5
  simp only [e_933_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7491)) := by
  have c57903 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57904 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57905 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57906 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57907 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57908 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_6 := arithEq_of_rows h (row := 933) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_6
  simp only [← c57903, ← c57904, ← c57905] at e_933_6
  have e_933_7 := arithEq_of_rows h (row := 933) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_7
  simp only [← c57906, ← c57907, k2, ← c57908] at e_933_7
  have hr := e_933_7
  simp only [e_933_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7492)) := by
  have c57909 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57910 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57911 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57912 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57913 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57914 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_8 := arithEq_of_rows h (row := 933) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_8
  simp only [← c57909, ← c57910, ← c57911] at e_933_8
  have e_933_9 := arithEq_of_rows h (row := 933) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_9
  simp only [← c57912, ← c57913, k2, ← c57914] at e_933_9
  have hr := e_933_9
  simp only [e_933_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7493)) := by
  have c57915 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57916 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57917 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57918 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57919 := (publicBatchWrapper86_copies1809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57920 := (publicBatchWrapper86_copies1810 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_10 := arithEq_of_rows h (row := 933) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_10
  simp only [← c57915, ← c57916, ← c57917] at e_933_10
  have e_933_11 := arithEq_of_rows h (row := 933) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_11
  simp only [← c57918, ← c57919, k2, ← c57920] at e_933_11
  have hr := e_933_11
  simp only [e_933_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7494)) := by
  have c57921 := (publicBatchWrapper86_copies1810 a h).2.1
  have c57922 := (publicBatchWrapper86_copies1810 a h).2.2.1
  have c57923 := (publicBatchWrapper86_copies1810 a h).2.2.2.1
  have c57924 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.1
  have c57925 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.1
  have c57926 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_12 := arithEq_of_rows h (row := 933) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_12
  simp only [← c57921, ← c57922, ← c57923] at e_933_12
  have e_933_13 := arithEq_of_rows h (row := 933) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_13
  simp only [← c57924, ← c57925, k2, ← c57926] at e_933_13
  have hr := e_933_13
  simp only [e_933_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7495)) := by
  have c57927 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.1
  have c57928 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.1
  have c57929 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.1
  have c57930 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57931 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57932 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_14 := arithEq_of_rows h (row := 933) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_14
  simp only [← c57927, ← c57928, ← c57929] at e_933_14
  have e_933_15 := arithEq_of_rows h (row := 933) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_15
  simp only [← c57930, ← c57931, k2, ← c57932] at e_933_15
  have hr := e_933_15
  simp only [e_933_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7496)) := by
  have c57933 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57934 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57935 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57936 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57937 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57938 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_16 := arithEq_of_rows h (row := 933) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_16
  simp only [← c57933, ← c57934, ← c57935] at e_933_16
  have e_933_17 := arithEq_of_rows h (row := 933) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_17
  simp only [← c57936, ← c57937, k2, ← c57938] at e_933_17
  have hr := e_933_17
  simp only [e_933_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 933 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7497)) := by
  have c57939 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57940 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57941 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57942 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57943 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57944 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_933_18 := arithEq_of_rows h (row := 933) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_18
  simp only [← c57939, ← c57940, ← c57941] at e_933_18
  have e_933_19 := arithEq_of_rows h (row := 933) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_933_19
  simp only [← c57942, ← c57943, k2, ← c57944] at e_933_19
  have hr := e_933_19
  simp only [e_933_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7498)) := by
  have c57945 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57946 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57947 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57948 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57949 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57950 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_0 := arithEq_of_rows h (row := 934) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_0
  simp only [← c57945, ← c57946, ← c57947] at e_934_0
  have e_934_1 := arithEq_of_rows h (row := 934) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_1
  simp only [← c57948, ← c57949, k2, ← c57950] at e_934_1
  have hr := e_934_1
  simp only [e_934_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7499)) := by
  have c57951 := (publicBatchWrapper86_copies1810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57952 := (publicBatchWrapper86_copies1811 a h).1
  have c57953 := (publicBatchWrapper86_copies1811 a h).2.1
  have c57954 := (publicBatchWrapper86_copies1811 a h).2.2.1
  have c57955 := (publicBatchWrapper86_copies1811 a h).2.2.2.1
  have c57956 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_2 := arithEq_of_rows h (row := 934) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_2
  simp only [← c57951, ← c57952, ← c57953] at e_934_2
  have e_934_3 := arithEq_of_rows h (row := 934) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_3
  simp only [← c57954, ← c57955, k2, ← c57956] at e_934_3
  have hr := e_934_3
  simp only [e_934_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7500)) := by
  have c57957 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.1
  have c57958 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.1
  have c57959 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.1
  have c57960 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.1
  have c57961 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.1
  have c57962 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_4 := arithEq_of_rows h (row := 934) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_4
  simp only [← c57957, ← c57958, ← c57959] at e_934_4
  have e_934_5 := arithEq_of_rows h (row := 934) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_5
  simp only [← c57960, ← c57961, k2, ← c57962] at e_934_5
  have hr := e_934_5
  simp only [e_934_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7501)) := by
  have c57963 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57964 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57965 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57966 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57967 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57968 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_6 := arithEq_of_rows h (row := 934) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_6
  simp only [← c57963, ← c57964, ← c57965] at e_934_6
  have e_934_7 := arithEq_of_rows h (row := 934) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_7
  simp only [← c57966, ← c57967, k2, ← c57968] at e_934_7
  have hr := e_934_7
  simp only [e_934_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7502)) := by
  have c57969 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57970 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57971 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57972 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57973 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57974 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_8 := arithEq_of_rows h (row := 934) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_8
  simp only [← c57969, ← c57970, ← c57971] at e_934_8
  have e_934_9 := arithEq_of_rows h (row := 934) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_9
  simp only [← c57972, ← c57973, k2, ← c57974] at e_934_9
  have hr := e_934_9
  simp only [e_934_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7503)) := by
  have c57975 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57976 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57977 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57978 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57979 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57980 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_10 := arithEq_of_rows h (row := 934) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_10
  simp only [← c57975, ← c57976, ← c57977] at e_934_10
  have e_934_11 := arithEq_of_rows h (row := 934) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_11
  simp only [← c57978, ← c57979, k2, ← c57980] at e_934_11
  have hr := e_934_11
  simp only [e_934_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7504)) := by
  have c57981 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57982 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57983 := (publicBatchWrapper86_copies1811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c57984 := (publicBatchWrapper86_copies1812 a h).1
  have c57985 := (publicBatchWrapper86_copies1812 a h).2.1
  have c57986 := (publicBatchWrapper86_copies1812 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_12 := arithEq_of_rows h (row := 934) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_12
  simp only [← c57981, ← c57982, ← c57983] at e_934_12
  have e_934_13 := arithEq_of_rows h (row := 934) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_13
  simp only [← c57984, ← c57985, k2, ← c57986] at e_934_13
  have hr := e_934_13
  simp only [e_934_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7505)) := by
  have c57987 := (publicBatchWrapper86_copies1812 a h).2.2.2.1
  have c57988 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.1
  have c57989 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.1
  have c57990 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.1
  have c57991 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.1
  have c57992 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_14 := arithEq_of_rows h (row := 934) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_14
  simp only [← c57987, ← c57988, ← c57989] at e_934_14
  have e_934_15 := arithEq_of_rows h (row := 934) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_15
  simp only [← c57990, ← c57991, k2, ← c57992] at e_934_15
  have hr := e_934_15
  simp only [e_934_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7506)) := by
  have c57993 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.1
  have c57994 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c57995 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c57996 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57997 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c57998 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_16 := arithEq_of_rows h (row := 934) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_16
  simp only [← c57993, ← c57994, ← c57995] at e_934_16
  have e_934_17 := arithEq_of_rows h (row := 934) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_17
  simp only [← c57996, ← c57997, k2, ← c57998] at e_934_17
  have hr := e_934_17
  simp only [e_934_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 934 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7507)) := by
  have c57999 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58000 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58001 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58002 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58003 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58004 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_934_18 := arithEq_of_rows h (row := 934) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_18
  simp only [← c57999, ← c58000, ← c58001] at e_934_18
  have e_934_19 := arithEq_of_rows h (row := 934) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_934_19
  simp only [← c58002, ← c58003, k2, ← c58004] at e_934_19
  have hr := e_934_19
  simp only [e_934_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7508)) := by
  have c58005 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58006 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58007 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58008 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58009 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58010 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_0 := arithEq_of_rows h (row := 935) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_0
  simp only [← c58005, ← c58006, ← c58007] at e_935_0
  have e_935_1 := arithEq_of_rows h (row := 935) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_1
  simp only [← c58008, ← c58009, k2, ← c58010] at e_935_1
  have hr := e_935_1
  simp only [e_935_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7509)) := by
  have c58011 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58012 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58013 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58014 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58015 := (publicBatchWrapper86_copies1812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58016 := (publicBatchWrapper86_copies1813 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_2 := arithEq_of_rows h (row := 935) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_2
  simp only [← c58011, ← c58012, ← c58013] at e_935_2
  have e_935_3 := arithEq_of_rows h (row := 935) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_3
  simp only [← c58014, ← c58015, k2, ← c58016] at e_935_3
  have hr := e_935_3
  simp only [e_935_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7510)) := by
  have c58017 := (publicBatchWrapper86_copies1813 a h).2.1
  have c58018 := (publicBatchWrapper86_copies1813 a h).2.2.1
  have c58019 := (publicBatchWrapper86_copies1813 a h).2.2.2.1
  have c58020 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.1
  have c58021 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.1
  have c58022 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_4 := arithEq_of_rows h (row := 935) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_4
  simp only [← c58017, ← c58018, ← c58019] at e_935_4
  have e_935_5 := arithEq_of_rows h (row := 935) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_5
  simp only [← c58020, ← c58021, k2, ← c58022] at e_935_5
  have hr := e_935_5
  simp only [e_935_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7511)) := by
  have c58023 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.1
  have c58024 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.1
  have c58025 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.1
  have c58026 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58027 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58028 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_6 := arithEq_of_rows h (row := 935) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_6
  simp only [← c58023, ← c58024, ← c58025] at e_935_6
  have e_935_7 := arithEq_of_rows h (row := 935) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_7
  simp only [← c58026, ← c58027, k2, ← c58028] at e_935_7
  have hr := e_935_7
  simp only [e_935_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7512)) := by
  have c58029 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58030 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58031 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58032 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58033 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58034 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_8 := arithEq_of_rows h (row := 935) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_8
  simp only [← c58029, ← c58030, ← c58031] at e_935_8
  have e_935_9 := arithEq_of_rows h (row := 935) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_9
  simp only [← c58032, ← c58033, k2, ← c58034] at e_935_9
  have hr := e_935_9
  simp only [e_935_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7513)) := by
  have c58035 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58036 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58037 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58038 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58039 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58040 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_10 := arithEq_of_rows h (row := 935) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_10
  simp only [← c58035, ← c58036, ← c58037] at e_935_10
  have e_935_11 := arithEq_of_rows h (row := 935) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_11
  simp only [← c58038, ← c58039, k2, ← c58040] at e_935_11
  have hr := e_935_11
  simp only [e_935_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7514)) := by
  have c58041 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58042 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58043 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58044 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58045 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58046 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_12 := arithEq_of_rows h (row := 935) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_12
  simp only [← c58041, ← c58042, ← c58043] at e_935_12
  have e_935_13 := arithEq_of_rows h (row := 935) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_13
  simp only [← c58044, ← c58045, k2, ← c58046] at e_935_13
  have hr := e_935_13
  simp only [e_935_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7515)) := by
  have c58047 := (publicBatchWrapper86_copies1813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58048 := (publicBatchWrapper86_copies1814 a h).1
  have c58049 := (publicBatchWrapper86_copies1814 a h).2.1
  have c58050 := (publicBatchWrapper86_copies1814 a h).2.2.1
  have c58051 := (publicBatchWrapper86_copies1814 a h).2.2.2.1
  have c58052 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_14 := arithEq_of_rows h (row := 935) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_14
  simp only [← c58047, ← c58048, ← c58049] at e_935_14
  have e_935_15 := arithEq_of_rows h (row := 935) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_15
  simp only [← c58050, ← c58051, k2, ← c58052] at e_935_15
  have hr := e_935_15
  simp only [e_935_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7516)) := by
  have c58053 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.1
  have c58054 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.1
  have c58055 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.1
  have c58056 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.1
  have c58057 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.1
  have c58058 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_16 := arithEq_of_rows h (row := 935) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_16
  simp only [← c58053, ← c58054, ← c58055] at e_935_16
  have e_935_17 := arithEq_of_rows h (row := 935) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_17
  simp only [← c58056, ← c58057, k2, ← c58058] at e_935_17
  have hr := e_935_17
  simp only [e_935_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 935 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7517)) := by
  have c58059 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58060 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58061 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58062 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58063 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58064 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_935_18 := arithEq_of_rows h (row := 935) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_18
  simp only [← c58059, ← c58060, ← c58061] at e_935_18
  have e_935_19 := arithEq_of_rows h (row := 935) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_935_19
  simp only [← c58062, ← c58063, k2, ← c58064] at e_935_19
  have hr := e_935_19
  simp only [e_935_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7518)) := by
  have c58065 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58066 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58067 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58068 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58069 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58070 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_0 := arithEq_of_rows h (row := 936) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_0
  simp only [← c58065, ← c58066, ← c58067] at e_936_0
  have e_936_1 := arithEq_of_rows h (row := 936) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_1
  simp only [← c58068, ← c58069, k2, ← c58070] at e_936_1
  have hr := e_936_1
  simp only [e_936_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7519)) := by
  have c58071 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58072 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58073 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58074 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58075 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58076 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_2 := arithEq_of_rows h (row := 936) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_2
  simp only [← c58071, ← c58072, ← c58073] at e_936_2
  have e_936_3 := arithEq_of_rows h (row := 936) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_3
  simp only [← c58074, ← c58075, k2, ← c58076] at e_936_3
  have hr := e_936_3
  simp only [e_936_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7520)) := by
  have c58077 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58078 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58079 := (publicBatchWrapper86_copies1814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58080 := (publicBatchWrapper86_copies1815 a h).1
  have c58081 := (publicBatchWrapper86_copies1815 a h).2.1
  have c58082 := (publicBatchWrapper86_copies1815 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_4 := arithEq_of_rows h (row := 936) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_4
  simp only [← c58077, ← c58078, ← c58079] at e_936_4
  have e_936_5 := arithEq_of_rows h (row := 936) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_5
  simp only [← c58080, ← c58081, k2, ← c58082] at e_936_5
  have hr := e_936_5
  simp only [e_936_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7521)) := by
  have c58083 := (publicBatchWrapper86_copies1815 a h).2.2.2.1
  have c58084 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.1
  have c58085 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.1
  have c58086 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.1
  have c58087 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.1
  have c58088 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_6 := arithEq_of_rows h (row := 936) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_6
  simp only [← c58083, ← c58084, ← c58085] at e_936_6
  have e_936_7 := arithEq_of_rows h (row := 936) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_7
  simp only [← c58086, ← c58087, k2, ← c58088] at e_936_7
  have hr := e_936_7
  simp only [e_936_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7522)) := by
  have c58089 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.1
  have c58090 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58091 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58092 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58093 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58094 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_8 := arithEq_of_rows h (row := 936) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_8
  simp only [← c58089, ← c58090, ← c58091] at e_936_8
  have e_936_9 := arithEq_of_rows h (row := 936) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_9
  simp only [← c58092, ← c58093, k2, ← c58094] at e_936_9
  have hr := e_936_9
  simp only [e_936_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7523)) := by
  have c58095 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58096 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58097 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58098 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58099 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58100 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_10 := arithEq_of_rows h (row := 936) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_10
  simp only [← c58095, ← c58096, ← c58097] at e_936_10
  have e_936_11 := arithEq_of_rows h (row := 936) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_11
  simp only [← c58098, ← c58099, k2, ← c58100] at e_936_11
  have hr := e_936_11
  simp only [e_936_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7524)) := by
  have c58101 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58102 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58103 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58104 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58105 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58106 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_12 := arithEq_of_rows h (row := 936) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_12
  simp only [← c58101, ← c58102, ← c58103] at e_936_12
  have e_936_13 := arithEq_of_rows h (row := 936) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_13
  simp only [← c58104, ← c58105, k2, ← c58106] at e_936_13
  have hr := e_936_13
  simp only [e_936_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7525)) := by
  have c58107 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58108 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58109 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58110 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58111 := (publicBatchWrapper86_copies1815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58112 := (publicBatchWrapper86_copies1816 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_14 := arithEq_of_rows h (row := 936) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_14
  simp only [← c58107, ← c58108, ← c58109] at e_936_14
  have e_936_15 := arithEq_of_rows h (row := 936) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_15
  simp only [← c58110, ← c58111, k2, ← c58112] at e_936_15
  have hr := e_936_15
  simp only [e_936_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7526)) := by
  have c58113 := (publicBatchWrapper86_copies1816 a h).2.1
  have c58114 := (publicBatchWrapper86_copies1816 a h).2.2.1
  have c58115 := (publicBatchWrapper86_copies1816 a h).2.2.2.1
  have c58116 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.1
  have c58117 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.1
  have c58118 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_16 := arithEq_of_rows h (row := 936) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_16
  simp only [← c58113, ← c58114, ← c58115] at e_936_16
  have e_936_17 := arithEq_of_rows h (row := 936) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_17
  simp only [← c58116, ← c58117, k2, ← c58118] at e_936_17
  have hr := e_936_17
  simp only [e_936_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 936 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7527)) := by
  have c58119 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.1
  have c58120 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.1
  have c58121 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.1
  have c58122 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58123 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58124 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_936_18 := arithEq_of_rows h (row := 936) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_18
  simp only [← c58119, ← c58120, ← c58121] at e_936_18
  have e_936_19 := arithEq_of_rows h (row := 936) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_936_19
  simp only [← c58122, ← c58123, k2, ← c58124] at e_936_19
  have hr := e_936_19
  simp only [e_936_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7528)) := by
  have c58125 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58126 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58127 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58128 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58129 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58130 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_0 := arithEq_of_rows h (row := 937) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_0
  simp only [← c58125, ← c58126, ← c58127] at e_937_0
  have e_937_1 := arithEq_of_rows h (row := 937) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_1
  simp only [← c58128, ← c58129, k2, ← c58130] at e_937_1
  have hr := e_937_1
  simp only [e_937_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7529)) := by
  have c58131 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58132 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58133 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58134 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58135 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58136 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_2 := arithEq_of_rows h (row := 937) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_2
  simp only [← c58131, ← c58132, ← c58133] at e_937_2
  have e_937_3 := arithEq_of_rows h (row := 937) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_3
  simp only [← c58134, ← c58135, k2, ← c58136] at e_937_3
  have hr := e_937_3
  simp only [e_937_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7530)) := by
  have c58137 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58138 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58139 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58140 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58141 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58142 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_4 := arithEq_of_rows h (row := 937) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_4
  simp only [← c58137, ← c58138, ← c58139] at e_937_4
  have e_937_5 := arithEq_of_rows h (row := 937) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_5
  simp only [← c58140, ← c58141, k2, ← c58142] at e_937_5
  have hr := e_937_5
  simp only [e_937_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7531)) := by
  have c58143 := (publicBatchWrapper86_copies1816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58144 := (publicBatchWrapper86_copies1817 a h).1
  have c58145 := (publicBatchWrapper86_copies1817 a h).2.1
  have c58146 := (publicBatchWrapper86_copies1817 a h).2.2.1
  have c58147 := (publicBatchWrapper86_copies1817 a h).2.2.2.1
  have c58148 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_6 := arithEq_of_rows h (row := 937) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_6
  simp only [← c58143, ← c58144, ← c58145] at e_937_6
  have e_937_7 := arithEq_of_rows h (row := 937) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_7
  simp only [← c58146, ← c58147, k2, ← c58148] at e_937_7
  have hr := e_937_7
  simp only [e_937_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7532)) := by
  have c58149 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.1
  have c58150 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.1
  have c58151 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.1
  have c58152 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.1
  have c58153 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.1
  have c58154 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_8 := arithEq_of_rows h (row := 937) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_8
  simp only [← c58149, ← c58150, ← c58151] at e_937_8
  have e_937_9 := arithEq_of_rows h (row := 937) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_9
  simp only [← c58152, ← c58153, k2, ← c58154] at e_937_9
  have hr := e_937_9
  simp only [e_937_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7533)) := by
  have c58155 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58156 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58157 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58158 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58159 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58160 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_10 := arithEq_of_rows h (row := 937) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_10
  simp only [← c58155, ← c58156, ← c58157] at e_937_10
  have e_937_11 := arithEq_of_rows h (row := 937) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_11
  simp only [← c58158, ← c58159, k2, ← c58160] at e_937_11
  have hr := e_937_11
  simp only [e_937_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7534)) := by
  have c58161 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58162 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58163 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58164 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58165 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58166 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_12 := arithEq_of_rows h (row := 937) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_12
  simp only [← c58161, ← c58162, ← c58163] at e_937_12
  have e_937_13 := arithEq_of_rows h (row := 937) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_13
  simp only [← c58164, ← c58165, k2, ← c58166] at e_937_13
  have hr := e_937_13
  simp only [e_937_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7535)) := by
  have c58167 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58168 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58169 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58170 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58171 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58172 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_14 := arithEq_of_rows h (row := 937) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_14
  simp only [← c58167, ← c58168, ← c58169] at e_937_14
  have e_937_15 := arithEq_of_rows h (row := 937) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_15
  simp only [← c58170, ← c58171, k2, ← c58172] at e_937_15
  have hr := e_937_15
  simp only [e_937_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7536)) := by
  have c58173 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58174 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58175 := (publicBatchWrapper86_copies1817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58176 := (publicBatchWrapper86_copies1818 a h).1
  have c58177 := (publicBatchWrapper86_copies1818 a h).2.1
  have c58178 := (publicBatchWrapper86_copies1818 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_16 := arithEq_of_rows h (row := 937) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_16
  simp only [← c58173, ← c58174, ← c58175] at e_937_16
  have e_937_17 := arithEq_of_rows h (row := 937) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_17
  simp only [← c58176, ← c58177, k2, ← c58178] at e_937_17
  have hr := e_937_17
  simp only [e_937_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 937 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7537)) := by
  have c58179 := (publicBatchWrapper86_copies1818 a h).2.2.2.1
  have c58180 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.1
  have c58181 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.1
  have c58182 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.1
  have c58183 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.1
  have c58184 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_937_18 := arithEq_of_rows h (row := 937) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_18
  simp only [← c58179, ← c58180, ← c58181] at e_937_18
  have e_937_19 := arithEq_of_rows h (row := 937) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_937_19
  simp only [← c58182, ← c58183, k2, ← c58184] at e_937_19
  have hr := e_937_19
  simp only [e_937_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7538)) := by
  have c58185 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.1
  have c58186 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58187 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58188 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58189 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58190 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_0 := arithEq_of_rows h (row := 938) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_0
  simp only [← c58185, ← c58186, ← c58187] at e_938_0
  have e_938_1 := arithEq_of_rows h (row := 938) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_1
  simp only [← c58188, ← c58189, k2, ← c58190] at e_938_1
  have hr := e_938_1
  simp only [e_938_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7539)) := by
  have c58191 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58192 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58193 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58194 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58195 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58196 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_2 := arithEq_of_rows h (row := 938) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_2
  simp only [← c58191, ← c58192, ← c58193] at e_938_2
  have e_938_3 := arithEq_of_rows h (row := 938) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_3
  simp only [← c58194, ← c58195, k2, ← c58196] at e_938_3
  have hr := e_938_3
  simp only [e_938_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7540)) := by
  have c58197 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58198 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58199 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58200 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58201 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58202 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_4 := arithEq_of_rows h (row := 938) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_4
  simp only [← c58197, ← c58198, ← c58199] at e_938_4
  have e_938_5 := arithEq_of_rows h (row := 938) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_5
  simp only [← c58200, ← c58201, k2, ← c58202] at e_938_5
  have hr := e_938_5
  simp only [e_938_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7541)) := by
  have c58203 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58204 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58205 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58206 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58207 := (publicBatchWrapper86_copies1818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58208 := (publicBatchWrapper86_copies1819 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_6 := arithEq_of_rows h (row := 938) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_6
  simp only [← c58203, ← c58204, ← c58205] at e_938_6
  have e_938_7 := arithEq_of_rows h (row := 938) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_7
  simp only [← c58206, ← c58207, k2, ← c58208] at e_938_7
  have hr := e_938_7
  simp only [e_938_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7542)) := by
  have c58209 := (publicBatchWrapper86_copies1819 a h).2.1
  have c58210 := (publicBatchWrapper86_copies1819 a h).2.2.1
  have c58211 := (publicBatchWrapper86_copies1819 a h).2.2.2.1
  have c58212 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.1
  have c58213 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.1
  have c58214 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_8 := arithEq_of_rows h (row := 938) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_8
  simp only [← c58209, ← c58210, ← c58211] at e_938_8
  have e_938_9 := arithEq_of_rows h (row := 938) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_9
  simp only [← c58212, ← c58213, k2, ← c58214] at e_938_9
  have hr := e_938_9
  simp only [e_938_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7543)) := by
  have c58215 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.1
  have c58216 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.1
  have c58217 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.1
  have c58218 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58219 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58220 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_10 := arithEq_of_rows h (row := 938) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_10
  simp only [← c58215, ← c58216, ← c58217] at e_938_10
  have e_938_11 := arithEq_of_rows h (row := 938) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_11
  simp only [← c58218, ← c58219, k2, ← c58220] at e_938_11
  have hr := e_938_11
  simp only [e_938_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7544)) := by
  have c58221 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58222 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58223 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58224 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58225 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58226 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_12 := arithEq_of_rows h (row := 938) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_12
  simp only [← c58221, ← c58222, ← c58223] at e_938_12
  have e_938_13 := arithEq_of_rows h (row := 938) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_13
  simp only [← c58224, ← c58225, k2, ← c58226] at e_938_13
  have hr := e_938_13
  simp only [e_938_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7545)) := by
  have c58227 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58228 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58229 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58230 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58231 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58232 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_14 := arithEq_of_rows h (row := 938) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_14
  simp only [← c58227, ← c58228, ← c58229] at e_938_14
  have e_938_15 := arithEq_of_rows h (row := 938) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_15
  simp only [← c58230, ← c58231, k2, ← c58232] at e_938_15
  have hr := e_938_15
  simp only [e_938_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7546)) := by
  have c58233 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58234 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58235 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58236 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58237 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58238 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_16 := arithEq_of_rows h (row := 938) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_16
  simp only [← c58233, ← c58234, ← c58235] at e_938_16
  have e_938_17 := arithEq_of_rows h (row := 938) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_17
  simp only [← c58236, ← c58237, k2, ← c58238] at e_938_17
  have hr := e_938_17
  simp only [e_938_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 938 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7547)) := by
  have c58239 := (publicBatchWrapper86_copies1819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58240 := (publicBatchWrapper86_copies1820 a h).1
  have c58241 := (publicBatchWrapper86_copies1820 a h).2.1
  have c58242 := (publicBatchWrapper86_copies1820 a h).2.2.1
  have c58243 := (publicBatchWrapper86_copies1820 a h).2.2.2.1
  have c58244 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_938_18 := arithEq_of_rows h (row := 938) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_18
  simp only [← c58239, ← c58240, ← c58241] at e_938_18
  have e_938_19 := arithEq_of_rows h (row := 938) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_938_19
  simp only [← c58242, ← c58243, k2, ← c58244] at e_938_19
  have hr := e_938_19
  simp only [e_938_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7548)) := by
  have c58245 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.1
  have c58246 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.1
  have c58247 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.1
  have c58248 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.1
  have c58249 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.1
  have c58250 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_0 := arithEq_of_rows h (row := 939) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_0
  simp only [← c58245, ← c58246, ← c58247] at e_939_0
  have e_939_1 := arithEq_of_rows h (row := 939) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_1
  simp only [← c58248, ← c58249, k2, ← c58250] at e_939_1
  have hr := e_939_1
  simp only [e_939_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7549)) := by
  have c58251 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58252 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58253 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58254 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58255 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58256 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_2 := arithEq_of_rows h (row := 939) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_2
  simp only [← c58251, ← c58252, ← c58253] at e_939_2
  have e_939_3 := arithEq_of_rows h (row := 939) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_3
  simp only [← c58254, ← c58255, k2, ← c58256] at e_939_3
  have hr := e_939_3
  simp only [e_939_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7550)) := by
  have c58257 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58258 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58259 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58260 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58261 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58262 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_4 := arithEq_of_rows h (row := 939) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_4
  simp only [← c58257, ← c58258, ← c58259] at e_939_4
  have e_939_5 := arithEq_of_rows h (row := 939) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_5
  simp only [← c58260, ← c58261, k2, ← c58262] at e_939_5
  have hr := e_939_5
  simp only [e_939_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7551)) := by
  have c58263 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58264 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58265 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58266 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58267 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58268 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_6 := arithEq_of_rows h (row := 939) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_6
  simp only [← c58263, ← c58264, ← c58265] at e_939_6
  have e_939_7 := arithEq_of_rows h (row := 939) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_7
  simp only [← c58266, ← c58267, k2, ← c58268] at e_939_7
  have hr := e_939_7
  simp only [e_939_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7552)) := by
  have c58269 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58270 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58271 := (publicBatchWrapper86_copies1820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58272 := (publicBatchWrapper86_copies1821 a h).1
  have c58273 := (publicBatchWrapper86_copies1821 a h).2.1
  have c58274 := (publicBatchWrapper86_copies1821 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_8 := arithEq_of_rows h (row := 939) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_8
  simp only [← c58269, ← c58270, ← c58271] at e_939_8
  have e_939_9 := arithEq_of_rows h (row := 939) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_9
  simp only [← c58272, ← c58273, k2, ← c58274] at e_939_9
  have hr := e_939_9
  simp only [e_939_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7553)) := by
  have c58275 := (publicBatchWrapper86_copies1821 a h).2.2.2.1
  have c58276 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.1
  have c58277 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.1
  have c58278 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.1
  have c58279 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.1
  have c58280 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_10 := arithEq_of_rows h (row := 939) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_10
  simp only [← c58275, ← c58276, ← c58277] at e_939_10
  have e_939_11 := arithEq_of_rows h (row := 939) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_11
  simp only [← c58278, ← c58279, k2, ← c58280] at e_939_11
  have hr := e_939_11
  simp only [e_939_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7554)) := by
  have c58281 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.1
  have c58282 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58283 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58284 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58285 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58286 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_12 := arithEq_of_rows h (row := 939) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_12
  simp only [← c58281, ← c58282, ← c58283] at e_939_12
  have e_939_13 := arithEq_of_rows h (row := 939) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_13
  simp only [← c58284, ← c58285, k2, ← c58286] at e_939_13
  have hr := e_939_13
  simp only [e_939_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7555)) := by
  have c58287 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58288 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58289 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58290 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58291 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58292 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_14 := arithEq_of_rows h (row := 939) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_14
  simp only [← c58287, ← c58288, ← c58289] at e_939_14
  have e_939_15 := arithEq_of_rows h (row := 939) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_15
  simp only [← c58290, ← c58291, k2, ← c58292] at e_939_15
  have hr := e_939_15
  simp only [e_939_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7556)) := by
  have c58293 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58294 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58295 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58296 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58297 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58298 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_16 := arithEq_of_rows h (row := 939) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_16
  simp only [← c58293, ← c58294, ← c58295] at e_939_16
  have e_939_17 := arithEq_of_rows h (row := 939) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_17
  simp only [← c58296, ← c58297, k2, ← c58298] at e_939_17
  have hr := e_939_17
  simp only [e_939_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 939 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7557)) := by
  have c58299 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58300 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58301 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58302 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58303 := (publicBatchWrapper86_copies1821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58304 := (publicBatchWrapper86_copies1822 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_939_18 := arithEq_of_rows h (row := 939) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_18
  simp only [← c58299, ← c58300, ← c58301] at e_939_18
  have e_939_19 := arithEq_of_rows h (row := 939) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_939_19
  simp only [← c58302, ← c58303, k2, ← c58304] at e_939_19
  have hr := e_939_19
  simp only [e_939_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7558)) := by
  have c58305 := (publicBatchWrapper86_copies1822 a h).2.1
  have c58306 := (publicBatchWrapper86_copies1822 a h).2.2.1
  have c58307 := (publicBatchWrapper86_copies1822 a h).2.2.2.1
  have c58308 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.1
  have c58309 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.1
  have c58310 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_0 := arithEq_of_rows h (row := 940) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_0
  simp only [← c58305, ← c58306, ← c58307] at e_940_0
  have e_940_1 := arithEq_of_rows h (row := 940) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_1
  simp only [← c58308, ← c58309, k2, ← c58310] at e_940_1
  have hr := e_940_1
  simp only [e_940_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7559)) := by
  have c58311 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.1
  have c58312 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.1
  have c58313 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.1
  have c58314 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58315 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58316 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_2 := arithEq_of_rows h (row := 940) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_2
  simp only [← c58311, ← c58312, ← c58313] at e_940_2
  have e_940_3 := arithEq_of_rows h (row := 940) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_3
  simp only [← c58314, ← c58315, k2, ← c58316] at e_940_3
  have hr := e_940_3
  simp only [e_940_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7560)) := by
  have c58317 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58318 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58319 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58320 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58321 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58322 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_4 := arithEq_of_rows h (row := 940) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_4
  simp only [← c58317, ← c58318, ← c58319] at e_940_4
  have e_940_5 := arithEq_of_rows h (row := 940) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_5
  simp only [← c58320, ← c58321, k2, ← c58322] at e_940_5
  have hr := e_940_5
  simp only [e_940_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7561)) := by
  have c58323 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58324 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58325 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58326 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58327 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58328 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_6 := arithEq_of_rows h (row := 940) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_6
  simp only [← c58323, ← c58324, ← c58325] at e_940_6
  have e_940_7 := arithEq_of_rows h (row := 940) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_7
  simp only [← c58326, ← c58327, k2, ← c58328] at e_940_7
  have hr := e_940_7
  simp only [e_940_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7562)) := by
  have c58329 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58330 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58331 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58332 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58333 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58334 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_8 := arithEq_of_rows h (row := 940) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_8
  simp only [← c58329, ← c58330, ← c58331] at e_940_8
  have e_940_9 := arithEq_of_rows h (row := 940) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_9
  simp only [← c58332, ← c58333, k2, ← c58334] at e_940_9
  have hr := e_940_9
  simp only [e_940_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7563)) := by
  have c58335 := (publicBatchWrapper86_copies1822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58336 := (publicBatchWrapper86_copies1823 a h).1
  have c58337 := (publicBatchWrapper86_copies1823 a h).2.1
  have c58338 := (publicBatchWrapper86_copies1823 a h).2.2.1
  have c58339 := (publicBatchWrapper86_copies1823 a h).2.2.2.1
  have c58340 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_10 := arithEq_of_rows h (row := 940) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_10
  simp only [← c58335, ← c58336, ← c58337] at e_940_10
  have e_940_11 := arithEq_of_rows h (row := 940) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_11
  simp only [← c58338, ← c58339, k2, ← c58340] at e_940_11
  have hr := e_940_11
  simp only [e_940_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7564)) := by
  have c58341 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.1
  have c58342 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.1
  have c58343 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.1
  have c58344 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.1
  have c58345 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.1
  have c58346 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_12 := arithEq_of_rows h (row := 940) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_12
  simp only [← c58341, ← c58342, ← c58343] at e_940_12
  have e_940_13 := arithEq_of_rows h (row := 940) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_13
  simp only [← c58344, ← c58345, k2, ← c58346] at e_940_13
  have hr := e_940_13
  simp only [e_940_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f8999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7565)) := by
  have c58347 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58348 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58349 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58350 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58351 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58352 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_14 := arithEq_of_rows h (row := 940) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_14
  simp only [← c58347, ← c58348, ← c58349] at e_940_14
  have e_940_15 := arithEq_of_rows h (row := 940) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_15
  simp only [← c58350, ← c58351, k2, ← c58352] at e_940_15
  have hr := e_940_15
  simp only [e_940_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7566)) := by
  have c58353 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58354 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58355 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58356 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58357 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58358 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_16 := arithEq_of_rows h (row := 940) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_16
  simp only [← c58353, ← c58354, ← c58355] at e_940_16
  have e_940_17 := arithEq_of_rows h (row := 940) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_17
  simp only [← c58356, ← c58357, k2, ← c58358] at e_940_17
  have hr := e_940_17
  simp only [e_940_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 940 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7567)) := by
  have c58359 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58360 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58361 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58362 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58363 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58364 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_940_18 := arithEq_of_rows h (row := 940) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_18
  simp only [← c58359, ← c58360, ← c58361] at e_940_18
  have e_940_19 := arithEq_of_rows h (row := 940) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_940_19
  simp only [← c58362, ← c58363, k2, ← c58364] at e_940_19
  have hr := e_940_19
  simp only [e_940_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7568)) := by
  have c58365 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58366 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58367 := (publicBatchWrapper86_copies1823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58368 := (publicBatchWrapper86_copies1824 a h).1
  have c58369 := (publicBatchWrapper86_copies1824 a h).2.1
  have c58370 := (publicBatchWrapper86_copies1824 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_0 := arithEq_of_rows h (row := 941) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_0
  simp only [← c58365, ← c58366, ← c58367] at e_941_0
  have e_941_1 := arithEq_of_rows h (row := 941) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_1
  simp only [← c58368, ← c58369, k2, ← c58370] at e_941_1
  have hr := e_941_1
  simp only [e_941_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7569)) := by
  have c58371 := (publicBatchWrapper86_copies1824 a h).2.2.2.1
  have c58372 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.1
  have c58373 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.1
  have c58374 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.1
  have c58375 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.1
  have c58376 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_2 := arithEq_of_rows h (row := 941) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_2
  simp only [← c58371, ← c58372, ← c58373] at e_941_2
  have e_941_3 := arithEq_of_rows h (row := 941) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_3
  simp only [← c58374, ← c58375, k2, ← c58376] at e_941_3
  have hr := e_941_3
  simp only [e_941_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7610)) := by
  have c58377 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.1
  have c58378 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58379 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58380 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58381 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58382 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_4 := arithEq_of_rows h (row := 941) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_4
  simp only [← c58377, ← c58378, ← c58379] at e_941_4
  have e_941_5 := arithEq_of_rows h (row := 941) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_5
  simp only [← c58380, ← c58381, k2, ← c58382] at e_941_5
  have hr := e_941_5
  simp only [e_941_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7611)) := by
  have c58383 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58384 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58385 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58386 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58387 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58388 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_6 := arithEq_of_rows h (row := 941) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_6
  simp only [← c58383, ← c58384, ← c58385] at e_941_6
  have e_941_7 := arithEq_of_rows h (row := 941) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_7
  simp only [← c58386, ← c58387, k2, ← c58388] at e_941_7
  have hr := e_941_7
  simp only [e_941_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7612)) := by
  have c58389 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58390 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58391 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58392 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58393 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58394 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_8 := arithEq_of_rows h (row := 941) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_8
  simp only [← c58389, ← c58390, ← c58391] at e_941_8
  have e_941_9 := arithEq_of_rows h (row := 941) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_9
  simp only [← c58392, ← c58393, k2, ← c58394] at e_941_9
  have hr := e_941_9
  simp only [e_941_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7613)) := by
  have c58395 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58396 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58397 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58398 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58399 := (publicBatchWrapper86_copies1824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58400 := (publicBatchWrapper86_copies1825 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_10 := arithEq_of_rows h (row := 941) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_10
  simp only [← c58395, ← c58396, ← c58397] at e_941_10
  have e_941_11 := arithEq_of_rows h (row := 941) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_11
  simp only [← c58398, ← c58399, k2, ← c58400] at e_941_11
  have hr := e_941_11
  simp only [e_941_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7614)) := by
  have c58401 := (publicBatchWrapper86_copies1825 a h).2.1
  have c58402 := (publicBatchWrapper86_copies1825 a h).2.2.1
  have c58403 := (publicBatchWrapper86_copies1825 a h).2.2.2.1
  have c58404 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.1
  have c58405 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.1
  have c58406 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_12 := arithEq_of_rows h (row := 941) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_12
  simp only [← c58401, ← c58402, ← c58403] at e_941_12
  have e_941_13 := arithEq_of_rows h (row := 941) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_13
  simp only [← c58404, ← c58405, k2, ← c58406] at e_941_13
  have hr := e_941_13
  simp only [e_941_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7615)) := by
  have c58407 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.1
  have c58408 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.1
  have c58409 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.1
  have c58410 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58411 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58412 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_14 := arithEq_of_rows h (row := 941) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_14
  simp only [← c58407, ← c58408, ← c58409] at e_941_14
  have e_941_15 := arithEq_of_rows h (row := 941) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_15
  simp only [← c58410, ← c58411, k2, ← c58412] at e_941_15
  have hr := e_941_15
  simp only [e_941_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7616)) := by
  have c58413 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58414 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58415 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58416 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58417 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58418 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_16 := arithEq_of_rows h (row := 941) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_16
  simp only [← c58413, ← c58414, ← c58415] at e_941_16
  have e_941_17 := arithEq_of_rows h (row := 941) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_17
  simp only [← c58416, ← c58417, k2, ← c58418] at e_941_17
  have hr := e_941_17
  simp only [e_941_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 941 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7617)) := by
  have c58419 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58420 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58421 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58422 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58423 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58424 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_941_18 := arithEq_of_rows h (row := 941) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_18
  simp only [← c58419, ← c58420, ← c58421] at e_941_18
  have e_941_19 := arithEq_of_rows h (row := 941) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_941_19
  simp only [← c58422, ← c58423, k2, ← c58424] at e_941_19
  have hr := e_941_19
  simp only [e_941_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7618)) := by
  have c58425 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58426 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58427 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58428 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58429 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58430 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_0 := arithEq_of_rows h (row := 942) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_0
  simp only [← c58425, ← c58426, ← c58427] at e_942_0
  have e_942_1 := arithEq_of_rows h (row := 942) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_1
  simp only [← c58428, ← c58429, k2, ← c58430] at e_942_1
  have hr := e_942_1
  simp only [e_942_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7619)) := by
  have c58431 := (publicBatchWrapper86_copies1825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58432 := (publicBatchWrapper86_copies1826 a h).1
  have c58433 := (publicBatchWrapper86_copies1826 a h).2.1
  have c58434 := (publicBatchWrapper86_copies1826 a h).2.2.1
  have c58435 := (publicBatchWrapper86_copies1826 a h).2.2.2.1
  have c58436 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_2 := arithEq_of_rows h (row := 942) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_2
  simp only [← c58431, ← c58432, ← c58433] at e_942_2
  have e_942_3 := arithEq_of_rows h (row := 942) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_3
  simp only [← c58434, ← c58435, k2, ← c58436] at e_942_3
  have hr := e_942_3
  simp only [e_942_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7620)) := by
  have c58437 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.1
  have c58438 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.1
  have c58439 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.1
  have c58440 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.1
  have c58441 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.1
  have c58442 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_4 := arithEq_of_rows h (row := 942) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_4
  simp only [← c58437, ← c58438, ← c58439] at e_942_4
  have e_942_5 := arithEq_of_rows h (row := 942) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_5
  simp only [← c58440, ← c58441, k2, ← c58442] at e_942_5
  have hr := e_942_5
  simp only [e_942_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7621)) := by
  have c58443 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58444 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58445 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58446 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58447 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58448 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_6 := arithEq_of_rows h (row := 942) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_6
  simp only [← c58443, ← c58444, ← c58445] at e_942_6
  have e_942_7 := arithEq_of_rows h (row := 942) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_7
  simp only [← c58446, ← c58447, k2, ← c58448] at e_942_7
  have hr := e_942_7
  simp only [e_942_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7622)) := by
  have c58449 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58450 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58451 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58452 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58453 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58454 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_8 := arithEq_of_rows h (row := 942) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_8
  simp only [← c58449, ← c58450, ← c58451] at e_942_8
  have e_942_9 := arithEq_of_rows h (row := 942) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_9
  simp only [← c58452, ← c58453, k2, ← c58454] at e_942_9
  have hr := e_942_9
  simp only [e_942_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7623)) := by
  have c58455 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58456 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58457 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58458 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58459 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58460 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_10 := arithEq_of_rows h (row := 942) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_10
  simp only [← c58455, ← c58456, ← c58457] at e_942_10
  have e_942_11 := arithEq_of_rows h (row := 942) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_11
  simp only [← c58458, ← c58459, k2, ← c58460] at e_942_11
  have hr := e_942_11
  simp only [e_942_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7624)) := by
  have c58461 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58462 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58463 := (publicBatchWrapper86_copies1826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58464 := (publicBatchWrapper86_copies1827 a h).1
  have c58465 := (publicBatchWrapper86_copies1827 a h).2.1
  have c58466 := (publicBatchWrapper86_copies1827 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_12 := arithEq_of_rows h (row := 942) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_12
  simp only [← c58461, ← c58462, ← c58463] at e_942_12
  have e_942_13 := arithEq_of_rows h (row := 942) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_13
  simp only [← c58464, ← c58465, k2, ← c58466] at e_942_13
  have hr := e_942_13
  simp only [e_942_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7625)) := by
  have c58467 := (publicBatchWrapper86_copies1827 a h).2.2.2.1
  have c58468 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.1
  have c58469 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.1
  have c58470 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.1
  have c58471 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.1
  have c58472 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_14 := arithEq_of_rows h (row := 942) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_14
  simp only [← c58467, ← c58468, ← c58469] at e_942_14
  have e_942_15 := arithEq_of_rows h (row := 942) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_15
  simp only [← c58470, ← c58471, k2, ← c58472] at e_942_15
  have hr := e_942_15
  simp only [e_942_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7626)) := by
  have c58473 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.1
  have c58474 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58475 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58476 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58477 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58478 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_16 := arithEq_of_rows h (row := 942) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_16
  simp only [← c58473, ← c58474, ← c58475] at e_942_16
  have e_942_17 := arithEq_of_rows h (row := 942) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_17
  simp only [← c58476, ← c58477, k2, ← c58478] at e_942_17
  have hr := e_942_17
  simp only [e_942_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 942 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7627)) := by
  have c58479 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58480 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58481 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58482 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58483 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58484 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_942_18 := arithEq_of_rows h (row := 942) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_18
  simp only [← c58479, ← c58480, ← c58481] at e_942_18
  have e_942_19 := arithEq_of_rows h (row := 942) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_942_19
  simp only [← c58482, ← c58483, k2, ← c58484] at e_942_19
  have hr := e_942_19
  simp only [e_942_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7628)) := by
  have c58485 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58486 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58487 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58488 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58489 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58490 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_0 := arithEq_of_rows h (row := 943) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_0
  simp only [← c58485, ← c58486, ← c58487] at e_943_0
  have e_943_1 := arithEq_of_rows h (row := 943) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_1
  simp only [← c58488, ← c58489, k2, ← c58490] at e_943_1
  have hr := e_943_1
  simp only [e_943_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7629)) := by
  have c58491 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58492 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58493 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58494 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58495 := (publicBatchWrapper86_copies1827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58496 := (publicBatchWrapper86_copies1828 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_2 := arithEq_of_rows h (row := 943) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_2
  simp only [← c58491, ← c58492, ← c58493] at e_943_2
  have e_943_3 := arithEq_of_rows h (row := 943) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_3
  simp only [← c58494, ← c58495, k2, ← c58496] at e_943_3
  have hr := e_943_3
  simp only [e_943_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7630)) := by
  have c58497 := (publicBatchWrapper86_copies1828 a h).2.1
  have c58498 := (publicBatchWrapper86_copies1828 a h).2.2.1
  have c58499 := (publicBatchWrapper86_copies1828 a h).2.2.2.1
  have c58500 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.1
  have c58501 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.1
  have c58502 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_4 := arithEq_of_rows h (row := 943) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_4
  simp only [← c58497, ← c58498, ← c58499] at e_943_4
  have e_943_5 := arithEq_of_rows h (row := 943) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_5
  simp only [← c58500, ← c58501, k2, ← c58502] at e_943_5
  have hr := e_943_5
  simp only [e_943_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7631)) := by
  have c58503 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.1
  have c58504 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.1
  have c58505 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.1
  have c58506 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58507 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58508 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_6 := arithEq_of_rows h (row := 943) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_6
  simp only [← c58503, ← c58504, ← c58505] at e_943_6
  have e_943_7 := arithEq_of_rows h (row := 943) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_7
  simp only [← c58506, ← c58507, k2, ← c58508] at e_943_7
  have hr := e_943_7
  simp only [e_943_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7632)) := by
  have c58509 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58510 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58511 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58512 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58513 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58514 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_8 := arithEq_of_rows h (row := 943) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_8
  simp only [← c58509, ← c58510, ← c58511] at e_943_8
  have e_943_9 := arithEq_of_rows h (row := 943) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_9
  simp only [← c58512, ← c58513, k2, ← c58514] at e_943_9
  have hr := e_943_9
  simp only [e_943_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7633)) := by
  have c58515 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58516 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58517 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58518 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58519 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58520 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_10 := arithEq_of_rows h (row := 943) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_10
  simp only [← c58515, ← c58516, ← c58517] at e_943_10
  have e_943_11 := arithEq_of_rows h (row := 943) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_11
  simp only [← c58518, ← c58519, k2, ← c58520] at e_943_11
  have hr := e_943_11
  simp only [e_943_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7634)) := by
  have c58521 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58522 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58523 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58524 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58525 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58526 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_12 := arithEq_of_rows h (row := 943) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_12
  simp only [← c58521, ← c58522, ← c58523] at e_943_12
  have e_943_13 := arithEq_of_rows h (row := 943) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_13
  simp only [← c58524, ← c58525, k2, ← c58526] at e_943_13
  have hr := e_943_13
  simp only [e_943_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7635)) := by
  have c58527 := (publicBatchWrapper86_copies1828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58528 := (publicBatchWrapper86_copies1829 a h).1
  have c58529 := (publicBatchWrapper86_copies1829 a h).2.1
  have c58530 := (publicBatchWrapper86_copies1829 a h).2.2.1
  have c58531 := (publicBatchWrapper86_copies1829 a h).2.2.2.1
  have c58532 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_14 := arithEq_of_rows h (row := 943) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_14
  simp only [← c58527, ← c58528, ← c58529] at e_943_14
  have e_943_15 := arithEq_of_rows h (row := 943) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_15
  simp only [← c58530, ← c58531, k2, ← c58532] at e_943_15
  have hr := e_943_15
  simp only [e_943_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7636)) := by
  have c58533 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.1
  have c58534 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.1
  have c58535 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.1
  have c58536 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.1
  have c58537 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.1
  have c58538 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_16 := arithEq_of_rows h (row := 943) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_16
  simp only [← c58533, ← c58534, ← c58535] at e_943_16
  have e_943_17 := arithEq_of_rows h (row := 943) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_17
  simp only [← c58536, ← c58537, k2, ← c58538] at e_943_17
  have hr := e_943_17
  simp only [e_943_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 943 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7637)) := by
  have c58539 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58540 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58541 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58542 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58543 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58544 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_943_18 := arithEq_of_rows h (row := 943) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_18
  simp only [← c58539, ← c58540, ← c58541] at e_943_18
  have e_943_19 := arithEq_of_rows h (row := 943) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_943_19
  simp only [← c58542, ← c58543, k2, ← c58544] at e_943_19
  have hr := e_943_19
  simp only [e_943_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7638)) := by
  have c58545 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58546 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58547 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58548 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58549 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58550 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_0 := arithEq_of_rows h (row := 944) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_0
  simp only [← c58545, ← c58546, ← c58547] at e_944_0
  have e_944_1 := arithEq_of_rows h (row := 944) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_1
  simp only [← c58548, ← c58549, k2, ← c58550] at e_944_1
  have hr := e_944_1
  simp only [e_944_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7639)) := by
  have c58551 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58552 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58553 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58554 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58555 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58556 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_2 := arithEq_of_rows h (row := 944) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_2
  simp only [← c58551, ← c58552, ← c58553] at e_944_2
  have e_944_3 := arithEq_of_rows h (row := 944) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_3
  simp only [← c58554, ← c58555, k2, ← c58556] at e_944_3
  have hr := e_944_3
  simp only [e_944_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7640)) := by
  have c58557 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58558 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58559 := (publicBatchWrapper86_copies1829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58560 := (publicBatchWrapper86_copies1830 a h).1
  have c58561 := (publicBatchWrapper86_copies1830 a h).2.1
  have c58562 := (publicBatchWrapper86_copies1830 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_4 := arithEq_of_rows h (row := 944) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_4
  simp only [← c58557, ← c58558, ← c58559] at e_944_4
  have e_944_5 := arithEq_of_rows h (row := 944) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_5
  simp only [← c58560, ← c58561, k2, ← c58562] at e_944_5
  have hr := e_944_5
  simp only [e_944_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7641)) := by
  have c58563 := (publicBatchWrapper86_copies1830 a h).2.2.2.1
  have c58564 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.1
  have c58565 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.1
  have c58566 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.1
  have c58567 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.1
  have c58568 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_6 := arithEq_of_rows h (row := 944) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_6
  simp only [← c58563, ← c58564, ← c58565] at e_944_6
  have e_944_7 := arithEq_of_rows h (row := 944) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_7
  simp only [← c58566, ← c58567, k2, ← c58568] at e_944_7
  have hr := e_944_7
  simp only [e_944_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7642)) := by
  have c58569 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.1
  have c58570 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58571 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58572 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58573 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58574 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_8 := arithEq_of_rows h (row := 944) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_8
  simp only [← c58569, ← c58570, ← c58571] at e_944_8
  have e_944_9 := arithEq_of_rows h (row := 944) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_9
  simp only [← c58572, ← c58573, k2, ← c58574] at e_944_9
  have hr := e_944_9
  simp only [e_944_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7643)) := by
  have c58575 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58576 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58577 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58578 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58579 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58580 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_10 := arithEq_of_rows h (row := 944) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_10
  simp only [← c58575, ← c58576, ← c58577] at e_944_10
  have e_944_11 := arithEq_of_rows h (row := 944) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_11
  simp only [← c58578, ← c58579, k2, ← c58580] at e_944_11
  have hr := e_944_11
  simp only [e_944_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7644)) := by
  have c58581 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58582 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58583 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58584 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58585 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58586 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_12 := arithEq_of_rows h (row := 944) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_12
  simp only [← c58581, ← c58582, ← c58583] at e_944_12
  have e_944_13 := arithEq_of_rows h (row := 944) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_13
  simp only [← c58584, ← c58585, k2, ← c58586] at e_944_13
  have hr := e_944_13
  simp only [e_944_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7645)) := by
  have c58587 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58588 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58589 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58590 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58591 := (publicBatchWrapper86_copies1830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58592 := (publicBatchWrapper86_copies1831 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_14 := arithEq_of_rows h (row := 944) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_14
  simp only [← c58587, ← c58588, ← c58589] at e_944_14
  have e_944_15 := arithEq_of_rows h (row := 944) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_15
  simp only [← c58590, ← c58591, k2, ← c58592] at e_944_15
  have hr := e_944_15
  simp only [e_944_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7646)) := by
  have c58593 := (publicBatchWrapper86_copies1831 a h).2.1
  have c58594 := (publicBatchWrapper86_copies1831 a h).2.2.1
  have c58595 := (publicBatchWrapper86_copies1831 a h).2.2.2.1
  have c58596 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.1
  have c58597 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.1
  have c58598 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_16 := arithEq_of_rows h (row := 944) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_16
  simp only [← c58593, ← c58594, ← c58595] at e_944_16
  have e_944_17 := arithEq_of_rows h (row := 944) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_17
  simp only [← c58596, ← c58597, k2, ← c58598] at e_944_17
  have hr := e_944_17
  simp only [e_944_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 944 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7647)) := by
  have c58599 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.1
  have c58600 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.1
  have c58601 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.1
  have c58602 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58603 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58604 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_944_18 := arithEq_of_rows h (row := 944) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_18
  simp only [← c58599, ← c58600, ← c58601] at e_944_18
  have e_944_19 := arithEq_of_rows h (row := 944) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_944_19
  simp only [← c58602, ← c58603, k2, ← c58604] at e_944_19
  have hr := e_944_19
  simp only [e_944_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7648)) := by
  have c58605 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58606 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58607 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58608 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58609 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58610 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_0 := arithEq_of_rows h (row := 945) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_0
  simp only [← c58605, ← c58606, ← c58607] at e_945_0
  have e_945_1 := arithEq_of_rows h (row := 945) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_1
  simp only [← c58608, ← c58609, k2, ← c58610] at e_945_1
  have hr := e_945_1
  simp only [e_945_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7649)) := by
  have c58611 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58612 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58613 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58614 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58615 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58616 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_2 := arithEq_of_rows h (row := 945) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_2
  simp only [← c58611, ← c58612, ← c58613] at e_945_2
  have e_945_3 := arithEq_of_rows h (row := 945) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_3
  simp only [← c58614, ← c58615, k2, ← c58616] at e_945_3
  have hr := e_945_3
  simp only [e_945_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7650)) := by
  have c58617 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58618 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58619 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58620 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58621 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58622 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_4 := arithEq_of_rows h (row := 945) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_4
  simp only [← c58617, ← c58618, ← c58619] at e_945_4
  have e_945_5 := arithEq_of_rows h (row := 945) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_5
  simp only [← c58620, ← c58621, k2, ← c58622] at e_945_5
  have hr := e_945_5
  simp only [e_945_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7651)) := by
  have c58623 := (publicBatchWrapper86_copies1831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58624 := (publicBatchWrapper86_copies1832 a h).1
  have c58625 := (publicBatchWrapper86_copies1832 a h).2.1
  have c58626 := (publicBatchWrapper86_copies1832 a h).2.2.1
  have c58627 := (publicBatchWrapper86_copies1832 a h).2.2.2.1
  have c58628 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_6 := arithEq_of_rows h (row := 945) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_6
  simp only [← c58623, ← c58624, ← c58625] at e_945_6
  have e_945_7 := arithEq_of_rows h (row := 945) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_7
  simp only [← c58626, ← c58627, k2, ← c58628] at e_945_7
  have hr := e_945_7
  simp only [e_945_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7652)) := by
  have c58629 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.1
  have c58630 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.1
  have c58631 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.1
  have c58632 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.1
  have c58633 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.1
  have c58634 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_8 := arithEq_of_rows h (row := 945) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_8
  simp only [← c58629, ← c58630, ← c58631] at e_945_8
  have e_945_9 := arithEq_of_rows h (row := 945) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_9
  simp only [← c58632, ← c58633, k2, ← c58634] at e_945_9
  have hr := e_945_9
  simp only [e_945_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7653)) := by
  have c58635 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58636 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58637 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58638 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58639 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58640 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_10 := arithEq_of_rows h (row := 945) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_10
  simp only [← c58635, ← c58636, ← c58637] at e_945_10
  have e_945_11 := arithEq_of_rows h (row := 945) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_11
  simp only [← c58638, ← c58639, k2, ← c58640] at e_945_11
  have hr := e_945_11
  simp only [e_945_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7654)) := by
  have c58641 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58642 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58643 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58644 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58645 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58646 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_12 := arithEq_of_rows h (row := 945) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_12
  simp only [← c58641, ← c58642, ← c58643] at e_945_12
  have e_945_13 := arithEq_of_rows h (row := 945) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_13
  simp only [← c58644, ← c58645, k2, ← c58646] at e_945_13
  have hr := e_945_13
  simp only [e_945_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7655)) := by
  have c58647 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58648 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58649 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58650 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58651 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58652 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_14 := arithEq_of_rows h (row := 945) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_14
  simp only [← c58647, ← c58648, ← c58649] at e_945_14
  have e_945_15 := arithEq_of_rows h (row := 945) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_15
  simp only [← c58650, ← c58651, k2, ← c58652] at e_945_15
  have hr := e_945_15
  simp only [e_945_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7656)) := by
  have c58653 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58654 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58655 := (publicBatchWrapper86_copies1832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58656 := (publicBatchWrapper86_copies1833 a h).1
  have c58657 := (publicBatchWrapper86_copies1833 a h).2.1
  have c58658 := (publicBatchWrapper86_copies1833 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_16 := arithEq_of_rows h (row := 945) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_16
  simp only [← c58653, ← c58654, ← c58655] at e_945_16
  have e_945_17 := arithEq_of_rows h (row := 945) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_17
  simp only [← c58656, ← c58657, k2, ← c58658] at e_945_17
  have hr := e_945_17
  simp only [e_945_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 945 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7657)) := by
  have c58659 := (publicBatchWrapper86_copies1833 a h).2.2.2.1
  have c58660 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.1
  have c58661 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.1
  have c58662 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.1
  have c58663 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.1
  have c58664 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_945_18 := arithEq_of_rows h (row := 945) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_18
  simp only [← c58659, ← c58660, ← c58661] at e_945_18
  have e_945_19 := arithEq_of_rows h (row := 945) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_945_19
  simp only [← c58662, ← c58663, k2, ← c58664] at e_945_19
  have hr := e_945_19
  simp only [e_945_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7658)) := by
  have c58665 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.1
  have c58666 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58667 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58668 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58669 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58670 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_0 := arithEq_of_rows h (row := 946) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_0
  simp only [← c58665, ← c58666, ← c58667] at e_946_0
  have e_946_1 := arithEq_of_rows h (row := 946) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_1
  simp only [← c58668, ← c58669, k2, ← c58670] at e_946_1
  have hr := e_946_1
  simp only [e_946_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7659)) := by
  have c58671 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58672 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58673 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58674 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58675 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58676 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_2 := arithEq_of_rows h (row := 946) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_2
  simp only [← c58671, ← c58672, ← c58673] at e_946_2
  have e_946_3 := arithEq_of_rows h (row := 946) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_3
  simp only [← c58674, ← c58675, k2, ← c58676] at e_946_3
  have hr := e_946_3
  simp only [e_946_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7660)) := by
  have c58677 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58678 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58679 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58680 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58681 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58682 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_4 := arithEq_of_rows h (row := 946) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_4
  simp only [← c58677, ← c58678, ← c58679] at e_946_4
  have e_946_5 := arithEq_of_rows h (row := 946) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_5
  simp only [← c58680, ← c58681, k2, ← c58682] at e_946_5
  have hr := e_946_5
  simp only [e_946_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7661)) := by
  have c58683 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58684 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58685 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58686 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58687 := (publicBatchWrapper86_copies1833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58688 := (publicBatchWrapper86_copies1834 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_6 := arithEq_of_rows h (row := 946) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_6
  simp only [← c58683, ← c58684, ← c58685] at e_946_6
  have e_946_7 := arithEq_of_rows h (row := 946) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_7
  simp only [← c58686, ← c58687, k2, ← c58688] at e_946_7
  have hr := e_946_7
  simp only [e_946_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7662)) := by
  have c58689 := (publicBatchWrapper86_copies1834 a h).2.1
  have c58690 := (publicBatchWrapper86_copies1834 a h).2.2.1
  have c58691 := (publicBatchWrapper86_copies1834 a h).2.2.2.1
  have c58692 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.1
  have c58693 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.1
  have c58694 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_8 := arithEq_of_rows h (row := 946) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_8
  simp only [← c58689, ← c58690, ← c58691] at e_946_8
  have e_946_9 := arithEq_of_rows h (row := 946) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_9
  simp only [← c58692, ← c58693, k2, ← c58694] at e_946_9
  have hr := e_946_9
  simp only [e_946_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7663)) := by
  have c58695 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.1
  have c58696 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.1
  have c58697 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.1
  have c58698 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58699 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58700 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_10 := arithEq_of_rows h (row := 946) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_10
  simp only [← c58695, ← c58696, ← c58697] at e_946_10
  have e_946_11 := arithEq_of_rows h (row := 946) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_11
  simp only [← c58698, ← c58699, k2, ← c58700] at e_946_11
  have hr := e_946_11
  simp only [e_946_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7664)) := by
  have c58701 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58702 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58703 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58704 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58705 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58706 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_12 := arithEq_of_rows h (row := 946) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_12
  simp only [← c58701, ← c58702, ← c58703] at e_946_12
  have e_946_13 := arithEq_of_rows h (row := 946) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_13
  simp only [← c58704, ← c58705, k2, ← c58706] at e_946_13
  have hr := e_946_13
  simp only [e_946_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7665)) := by
  have c58707 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58708 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58709 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58710 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58711 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58712 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_14 := arithEq_of_rows h (row := 946) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_14
  simp only [← c58707, ← c58708, ← c58709] at e_946_14
  have e_946_15 := arithEq_of_rows h (row := 946) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_15
  simp only [← c58710, ← c58711, k2, ← c58712] at e_946_15
  have hr := e_946_15
  simp only [e_946_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7666)) := by
  have c58713 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58714 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58715 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58716 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58717 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58718 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_16 := arithEq_of_rows h (row := 946) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_16
  simp only [← c58713, ← c58714, ← c58715] at e_946_16
  have e_946_17 := arithEq_of_rows h (row := 946) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_17
  simp only [← c58716, ← c58717, k2, ← c58718] at e_946_17
  have hr := e_946_17
  simp only [e_946_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 946 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7667)) := by
  have c58719 := (publicBatchWrapper86_copies1834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58720 := (publicBatchWrapper86_copies1835 a h).1
  have c58721 := (publicBatchWrapper86_copies1835 a h).2.1
  have c58722 := (publicBatchWrapper86_copies1835 a h).2.2.1
  have c58723 := (publicBatchWrapper86_copies1835 a h).2.2.2.1
  have c58724 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_946_18 := arithEq_of_rows h (row := 946) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_18
  simp only [← c58719, ← c58720, ← c58721] at e_946_18
  have e_946_19 := arithEq_of_rows h (row := 946) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_946_19
  simp only [← c58722, ← c58723, k2, ← c58724] at e_946_19
  have hr := e_946_19
  simp only [e_946_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7668)) := by
  have c58725 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.1
  have c58726 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.1
  have c58727 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.1
  have c58728 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.1
  have c58729 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.1
  have c58730 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_0 := arithEq_of_rows h (row := 947) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_0
  simp only [← c58725, ← c58726, ← c58727] at e_947_0
  have e_947_1 := arithEq_of_rows h (row := 947) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_1
  simp only [← c58728, ← c58729, k2, ← c58730] at e_947_1
  have hr := e_947_1
  simp only [e_947_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7669)) := by
  have c58731 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58732 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58733 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58734 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58735 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58736 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_2 := arithEq_of_rows h (row := 947) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_2
  simp only [← c58731, ← c58732, ← c58733] at e_947_2
  have e_947_3 := arithEq_of_rows h (row := 947) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_3
  simp only [← c58734, ← c58735, k2, ← c58736] at e_947_3
  have hr := e_947_3
  simp only [e_947_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7670)) := by
  have c58737 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58738 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58739 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58740 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58741 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58742 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_4 := arithEq_of_rows h (row := 947) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_4
  simp only [← c58737, ← c58738, ← c58739] at e_947_4
  have e_947_5 := arithEq_of_rows h (row := 947) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_5
  simp only [← c58740, ← c58741, k2, ← c58742] at e_947_5
  have hr := e_947_5
  simp only [e_947_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7671)) := by
  have c58743 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58744 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58745 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58746 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58747 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58748 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_6 := arithEq_of_rows h (row := 947) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_6
  simp only [← c58743, ← c58744, ← c58745] at e_947_6
  have e_947_7 := arithEq_of_rows h (row := 947) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_7
  simp only [← c58746, ← c58747, k2, ← c58748] at e_947_7
  have hr := e_947_7
  simp only [e_947_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7672)) := by
  have c58749 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58750 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58751 := (publicBatchWrapper86_copies1835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58752 := (publicBatchWrapper86_copies1836 a h).1
  have c58753 := (publicBatchWrapper86_copies1836 a h).2.1
  have c58754 := (publicBatchWrapper86_copies1836 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_8 := arithEq_of_rows h (row := 947) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_8
  simp only [← c58749, ← c58750, ← c58751] at e_947_8
  have e_947_9 := arithEq_of_rows h (row := 947) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_9
  simp only [← c58752, ← c58753, k2, ← c58754] at e_947_9
  have hr := e_947_9
  simp only [e_947_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7673)) := by
  have c58755 := (publicBatchWrapper86_copies1836 a h).2.2.2.1
  have c58756 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.1
  have c58757 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.1
  have c58758 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.1
  have c58759 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.1
  have c58760 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_10 := arithEq_of_rows h (row := 947) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_10
  simp only [← c58755, ← c58756, ← c58757] at e_947_10
  have e_947_11 := arithEq_of_rows h (row := 947) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_11
  simp only [← c58758, ← c58759, k2, ← c58760] at e_947_11
  have hr := e_947_11
  simp only [e_947_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7674)) := by
  have c58761 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.1
  have c58762 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58763 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58764 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58765 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58766 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_12 := arithEq_of_rows h (row := 947) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_12
  simp only [← c58761, ← c58762, ← c58763] at e_947_12
  have e_947_13 := arithEq_of_rows h (row := 947) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_13
  simp only [← c58764, ← c58765, k2, ← c58766] at e_947_13
  have hr := e_947_13
  simp only [e_947_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7675)) := by
  have c58767 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58768 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58769 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58770 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58771 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58772 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_14 := arithEq_of_rows h (row := 947) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_14
  simp only [← c58767, ← c58768, ← c58769] at e_947_14
  have e_947_15 := arithEq_of_rows h (row := 947) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_15
  simp only [← c58770, ← c58771, k2, ← c58772] at e_947_15
  have hr := e_947_15
  simp only [e_947_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7676)) := by
  have c58773 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58774 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58775 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58776 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58777 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58778 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_16 := arithEq_of_rows h (row := 947) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_16
  simp only [← c58773, ← c58774, ← c58775] at e_947_16
  have e_947_17 := arithEq_of_rows h (row := 947) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_17
  simp only [← c58776, ← c58777, k2, ← c58778] at e_947_17
  have hr := e_947_17
  simp only [e_947_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 947 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7677)) := by
  have c58779 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58780 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58781 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58782 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58783 := (publicBatchWrapper86_copies1836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58784 := (publicBatchWrapper86_copies1837 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_947_18 := arithEq_of_rows h (row := 947) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_18
  simp only [← c58779, ← c58780, ← c58781] at e_947_18
  have e_947_19 := arithEq_of_rows h (row := 947) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_947_19
  simp only [← c58782, ← c58783, k2, ← c58784] at e_947_19
  have hr := e_947_19
  simp only [e_947_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7678)) := by
  have c58785 := (publicBatchWrapper86_copies1837 a h).2.1
  have c58786 := (publicBatchWrapper86_copies1837 a h).2.2.1
  have c58787 := (publicBatchWrapper86_copies1837 a h).2.2.2.1
  have c58788 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.1
  have c58789 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.1
  have c58790 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_0 := arithEq_of_rows h (row := 948) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_0
  simp only [← c58785, ← c58786, ← c58787] at e_948_0
  have e_948_1 := arithEq_of_rows h (row := 948) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_1
  simp only [← c58788, ← c58789, k2, ← c58790] at e_948_1
  have hr := e_948_1
  simp only [e_948_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7679)) := by
  have c58791 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.1
  have c58792 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.1
  have c58793 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.1
  have c58794 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58795 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58796 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_2 := arithEq_of_rows h (row := 948) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_2
  simp only [← c58791, ← c58792, ← c58793] at e_948_2
  have e_948_3 := arithEq_of_rows h (row := 948) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_3
  simp only [← c58794, ← c58795, k2, ← c58796] at e_948_3
  have hr := e_948_3
  simp only [e_948_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7680)) := by
  have c58797 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58798 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58799 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58800 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58801 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58802 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_4 := arithEq_of_rows h (row := 948) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_4
  simp only [← c58797, ← c58798, ← c58799] at e_948_4
  have e_948_5 := arithEq_of_rows h (row := 948) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_5
  simp only [← c58800, ← c58801, k2, ← c58802] at e_948_5
  have hr := e_948_5
  simp only [e_948_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7681)) := by
  have c58803 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58804 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58805 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58806 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58807 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58808 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_6 := arithEq_of_rows h (row := 948) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_6
  simp only [← c58803, ← c58804, ← c58805] at e_948_6
  have e_948_7 := arithEq_of_rows h (row := 948) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_7
  simp only [← c58806, ← c58807, k2, ← c58808] at e_948_7
  have hr := e_948_7
  simp only [e_948_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7682)) := by
  have c58809 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58810 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58811 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58812 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58813 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58814 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_8 := arithEq_of_rows h (row := 948) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_8
  simp only [← c58809, ← c58810, ← c58811] at e_948_8
  have e_948_9 := arithEq_of_rows h (row := 948) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_9
  simp only [← c58812, ← c58813, k2, ← c58814] at e_948_9
  have hr := e_948_9
  simp only [e_948_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7683)) := by
  have c58815 := (publicBatchWrapper86_copies1837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58816 := (publicBatchWrapper86_copies1838 a h).1
  have c58817 := (publicBatchWrapper86_copies1838 a h).2.1
  have c58818 := (publicBatchWrapper86_copies1838 a h).2.2.1
  have c58819 := (publicBatchWrapper86_copies1838 a h).2.2.2.1
  have c58820 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_10 := arithEq_of_rows h (row := 948) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_10
  simp only [← c58815, ← c58816, ← c58817] at e_948_10
  have e_948_11 := arithEq_of_rows h (row := 948) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_11
  simp only [← c58818, ← c58819, k2, ← c58820] at e_948_11
  have hr := e_948_11
  simp only [e_948_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7684)) := by
  have c58821 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.1
  have c58822 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.1
  have c58823 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.1
  have c58824 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.1
  have c58825 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.1
  have c58826 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_12 := arithEq_of_rows h (row := 948) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_12
  simp only [← c58821, ← c58822, ← c58823] at e_948_12
  have e_948_13 := arithEq_of_rows h (row := 948) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_13
  simp only [← c58824, ← c58825, k2, ← c58826] at e_948_13
  have hr := e_948_13
  simp only [e_948_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7685)) := by
  have c58827 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58828 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58829 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58830 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58831 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58832 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_14 := arithEq_of_rows h (row := 948) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_14
  simp only [← c58827, ← c58828, ← c58829] at e_948_14
  have e_948_15 := arithEq_of_rows h (row := 948) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_15
  simp only [← c58830, ← c58831, k2, ← c58832] at e_948_15
  have hr := e_948_15
  simp only [e_948_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7686)) := by
  have c58833 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58834 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58835 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58836 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58837 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58838 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_16 := arithEq_of_rows h (row := 948) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_16
  simp only [← c58833, ← c58834, ← c58835] at e_948_16
  have e_948_17 := arithEq_of_rows h (row := 948) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_17
  simp only [← c58836, ← c58837, k2, ← c58838] at e_948_17
  have hr := e_948_17
  simp only [e_948_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 948 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7687)) := by
  have c58839 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58840 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58841 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58842 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58843 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58844 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_948_18 := arithEq_of_rows h (row := 948) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_18
  simp only [← c58839, ← c58840, ← c58841] at e_948_18
  have e_948_19 := arithEq_of_rows h (row := 948) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_948_19
  simp only [← c58842, ← c58843, k2, ← c58844] at e_948_19
  have hr := e_948_19
  simp only [e_948_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7688)) := by
  have c58845 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58846 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58847 := (publicBatchWrapper86_copies1838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58848 := (publicBatchWrapper86_copies1839 a h).1
  have c58849 := (publicBatchWrapper86_copies1839 a h).2.1
  have c58850 := (publicBatchWrapper86_copies1839 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_0 := arithEq_of_rows h (row := 949) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_0
  simp only [← c58845, ← c58846, ← c58847] at e_949_0
  have e_949_1 := arithEq_of_rows h (row := 949) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_1
  simp only [← c58848, ← c58849, k2, ← c58850] at e_949_1
  have hr := e_949_1
  simp only [e_949_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7689)) := by
  have c58851 := (publicBatchWrapper86_copies1839 a h).2.2.2.1
  have c58852 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.1
  have c58853 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.1
  have c58854 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.1
  have c58855 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.1
  have c58856 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_2 := arithEq_of_rows h (row := 949) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_2
  simp only [← c58851, ← c58852, ← c58853] at e_949_2
  have e_949_3 := arithEq_of_rows h (row := 949) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_3
  simp only [← c58854, ← c58855, k2, ← c58856] at e_949_3
  have hr := e_949_3
  simp only [e_949_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7690)) := by
  have c58857 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.1
  have c58858 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58859 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58860 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58861 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58862 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_4 := arithEq_of_rows h (row := 949) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_4
  simp only [← c58857, ← c58858, ← c58859] at e_949_4
  have e_949_5 := arithEq_of_rows h (row := 949) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_5
  simp only [← c58860, ← c58861, k2, ← c58862] at e_949_5
  have hr := e_949_5
  simp only [e_949_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7691)) := by
  have c58863 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58864 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58865 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58866 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58867 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58868 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_6 := arithEq_of_rows h (row := 949) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_6
  simp only [← c58863, ← c58864, ← c58865] at e_949_6
  have e_949_7 := arithEq_of_rows h (row := 949) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_7
  simp only [← c58866, ← c58867, k2, ← c58868] at e_949_7
  have hr := e_949_7
  simp only [e_949_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7692)) := by
  have c58869 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58870 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58871 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58872 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58873 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58874 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_8 := arithEq_of_rows h (row := 949) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_8
  simp only [← c58869, ← c58870, ← c58871] at e_949_8
  have e_949_9 := arithEq_of_rows h (row := 949) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_9
  simp only [← c58872, ← c58873, k2, ← c58874] at e_949_9
  have hr := e_949_9
  simp only [e_949_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7693)) := by
  have c58875 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58876 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58877 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58878 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58879 := (publicBatchWrapper86_copies1839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58880 := (publicBatchWrapper86_copies1840 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_10 := arithEq_of_rows h (row := 949) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_10
  simp only [← c58875, ← c58876, ← c58877] at e_949_10
  have e_949_11 := arithEq_of_rows h (row := 949) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_11
  simp only [← c58878, ← c58879, k2, ← c58880] at e_949_11
  have hr := e_949_11
  simp only [e_949_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7694)) := by
  have c58881 := (publicBatchWrapper86_copies1840 a h).2.1
  have c58882 := (publicBatchWrapper86_copies1840 a h).2.2.1
  have c58883 := (publicBatchWrapper86_copies1840 a h).2.2.2.1
  have c58884 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.1
  have c58885 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.1
  have c58886 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_12 := arithEq_of_rows h (row := 949) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_12
  simp only [← c58881, ← c58882, ← c58883] at e_949_12
  have e_949_13 := arithEq_of_rows h (row := 949) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_13
  simp only [← c58884, ← c58885, k2, ← c58886] at e_949_13
  have hr := e_949_13
  simp only [e_949_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7695)) := by
  have c58887 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.1
  have c58888 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.1
  have c58889 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.1
  have c58890 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58891 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58892 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_14 := arithEq_of_rows h (row := 949) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_14
  simp only [← c58887, ← c58888, ← c58889] at e_949_14
  have e_949_15 := arithEq_of_rows h (row := 949) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_15
  simp only [← c58890, ← c58891, k2, ← c58892] at e_949_15
  have hr := e_949_15
  simp only [e_949_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7696)) := by
  have c58893 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58894 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58895 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58896 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58897 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58898 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_16 := arithEq_of_rows h (row := 949) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_16
  simp only [← c58893, ← c58894, ← c58895] at e_949_16
  have e_949_17 := arithEq_of_rows h (row := 949) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_17
  simp only [← c58896, ← c58897, k2, ← c58898] at e_949_17
  have hr := e_949_17
  simp only [e_949_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 949 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7697)) := by
  have c58899 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58900 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58901 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58902 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58903 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58904 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_949_18 := arithEq_of_rows h (row := 949) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_18
  simp only [← c58899, ← c58900, ← c58901] at e_949_18
  have e_949_19 := arithEq_of_rows h (row := 949) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_949_19
  simp only [← c58902, ← c58903, k2, ← c58904] at e_949_19
  have hr := e_949_19
  simp only [e_949_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7698)) := by
  have c58905 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58906 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58907 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58908 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58909 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58910 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_0 := arithEq_of_rows h (row := 950) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_0
  simp only [← c58905, ← c58906, ← c58907] at e_950_0
  have e_950_1 := arithEq_of_rows h (row := 950) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_1
  simp only [← c58908, ← c58909, k2, ← c58910] at e_950_1
  have hr := e_950_1
  simp only [e_950_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7699)) := by
  have c58911 := (publicBatchWrapper86_copies1840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58912 := (publicBatchWrapper86_copies1841 a h).1
  have c58913 := (publicBatchWrapper86_copies1841 a h).2.1
  have c58914 := (publicBatchWrapper86_copies1841 a h).2.2.1
  have c58915 := (publicBatchWrapper86_copies1841 a h).2.2.2.1
  have c58916 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_2 := arithEq_of_rows h (row := 950) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_2
  simp only [← c58911, ← c58912, ← c58913] at e_950_2
  have e_950_3 := arithEq_of_rows h (row := 950) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_3
  simp only [← c58914, ← c58915, k2, ← c58916] at e_950_3
  have hr := e_950_3
  simp only [e_950_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7700)) := by
  have c58917 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.1
  have c58918 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.1
  have c58919 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.1
  have c58920 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.1
  have c58921 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.1
  have c58922 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_4 := arithEq_of_rows h (row := 950) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_4
  simp only [← c58917, ← c58918, ← c58919] at e_950_4
  have e_950_5 := arithEq_of_rows h (row := 950) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_5
  simp only [← c58920, ← c58921, k2, ← c58922] at e_950_5
  have hr := e_950_5
  simp only [e_950_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7701)) := by
  have c58923 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58924 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58925 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58926 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58927 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58928 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_6 := arithEq_of_rows h (row := 950) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_6
  simp only [← c58923, ← c58924, ← c58925] at e_950_6
  have e_950_7 := arithEq_of_rows h (row := 950) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_7
  simp only [← c58926, ← c58927, k2, ← c58928] at e_950_7
  have hr := e_950_7
  simp only [e_950_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7702)) := by
  have c58929 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58930 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58931 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58932 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58933 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58934 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_8 := arithEq_of_rows h (row := 950) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_8
  simp only [← c58929, ← c58930, ← c58931] at e_950_8
  have e_950_9 := arithEq_of_rows h (row := 950) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_9
  simp only [← c58932, ← c58933, k2, ← c58934] at e_950_9
  have hr := e_950_9
  simp only [e_950_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7703)) := by
  have c58935 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58936 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58937 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58938 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58939 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58940 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_10 := arithEq_of_rows h (row := 950) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_10
  simp only [← c58935, ← c58936, ← c58937] at e_950_10
  have e_950_11 := arithEq_of_rows h (row := 950) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_11
  simp only [← c58938, ← c58939, k2, ← c58940] at e_950_11
  have hr := e_950_11
  simp only [e_950_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7704)) := by
  have c58941 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58942 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58943 := (publicBatchWrapper86_copies1841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58944 := (publicBatchWrapper86_copies1842 a h).1
  have c58945 := (publicBatchWrapper86_copies1842 a h).2.1
  have c58946 := (publicBatchWrapper86_copies1842 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_12 := arithEq_of_rows h (row := 950) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_12
  simp only [← c58941, ← c58942, ← c58943] at e_950_12
  have e_950_13 := arithEq_of_rows h (row := 950) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_13
  simp only [← c58944, ← c58945, k2, ← c58946] at e_950_13
  have hr := e_950_13
  simp only [e_950_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7705)) := by
  have c58947 := (publicBatchWrapper86_copies1842 a h).2.2.2.1
  have c58948 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.1
  have c58949 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.1
  have c58950 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.1
  have c58951 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.1
  have c58952 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_14 := arithEq_of_rows h (row := 950) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_14
  simp only [← c58947, ← c58948, ← c58949] at e_950_14
  have e_950_15 := arithEq_of_rows h (row := 950) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_15
  simp only [← c58950, ← c58951, k2, ← c58952] at e_950_15
  have hr := e_950_15
  simp only [e_950_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7706)) := by
  have c58953 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.1
  have c58954 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58955 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58956 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58957 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58958 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_16 := arithEq_of_rows h (row := 950) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_16
  simp only [← c58953, ← c58954, ← c58955] at e_950_16
  have e_950_17 := arithEq_of_rows h (row := 950) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_17
  simp only [← c58956, ← c58957, k2, ← c58958] at e_950_17
  have hr := e_950_17
  simp only [e_950_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 950 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7707)) := by
  have c58959 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58960 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58961 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58962 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58963 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58964 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_950_18 := arithEq_of_rows h (row := 950) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_18
  simp only [← c58959, ← c58960, ← c58961] at e_950_18
  have e_950_19 := arithEq_of_rows h (row := 950) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_950_19
  simp only [← c58962, ← c58963, k2, ← c58964] at e_950_19
  have hr := e_950_19
  simp only [e_950_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7708)) := by
  have c58965 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58966 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58967 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58968 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58969 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58970 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_0 := arithEq_of_rows h (row := 951) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_0
  simp only [← c58965, ← c58966, ← c58967] at e_951_0
  have e_951_1 := arithEq_of_rows h (row := 951) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_1
  simp only [← c58968, ← c58969, k2, ← c58970] at e_951_1
  have hr := e_951_1
  simp only [e_951_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7709)) := by
  have c58971 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58972 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58973 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58974 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58975 := (publicBatchWrapper86_copies1842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c58976 := (publicBatchWrapper86_copies1843 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_2 := arithEq_of_rows h (row := 951) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_2
  simp only [← c58971, ← c58972, ← c58973] at e_951_2
  have e_951_3 := arithEq_of_rows h (row := 951) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_3
  simp only [← c58974, ← c58975, k2, ← c58976] at e_951_3
  have hr := e_951_3
  simp only [e_951_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7710)) := by
  have c58977 := (publicBatchWrapper86_copies1843 a h).2.1
  have c58978 := (publicBatchWrapper86_copies1843 a h).2.2.1
  have c58979 := (publicBatchWrapper86_copies1843 a h).2.2.2.1
  have c58980 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.1
  have c58981 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.1
  have c58982 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_4 := arithEq_of_rows h (row := 951) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_4
  simp only [← c58977, ← c58978, ← c58979] at e_951_4
  have e_951_5 := arithEq_of_rows h (row := 951) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_5
  simp only [← c58980, ← c58981, k2, ← c58982] at e_951_5
  have hr := e_951_5
  simp only [e_951_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7711)) := by
  have c58983 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.1
  have c58984 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.1
  have c58985 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.1
  have c58986 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.1
  have c58987 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c58988 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_6 := arithEq_of_rows h (row := 951) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_6
  simp only [← c58983, ← c58984, ← c58985] at e_951_6
  have e_951_7 := arithEq_of_rows h (row := 951) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_7
  simp only [← c58986, ← c58987, k2, ← c58988] at e_951_7
  have hr := e_951_7
  simp only [e_951_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7712)) := by
  have c58989 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58990 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58991 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58992 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58993 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58994 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_8 := arithEq_of_rows h (row := 951) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_8
  simp only [← c58989, ← c58990, ← c58991] at e_951_8
  have e_951_9 := arithEq_of_rows h (row := 951) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_9
  simp only [← c58992, ← c58993, k2, ← c58994] at e_951_9
  have hr := e_951_9
  simp only [e_951_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7713)) := by
  have c58995 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58996 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58997 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58998 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c58999 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59000 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_10 := arithEq_of_rows h (row := 951) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_10
  simp only [← c58995, ← c58996, ← c58997] at e_951_10
  have e_951_11 := arithEq_of_rows h (row := 951) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_11
  simp only [← c58998, ← c58999, k2, ← c59000] at e_951_11
  have hr := e_951_11
  simp only [e_951_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7714)) := by
  have c59001 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59002 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59003 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59004 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59005 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59006 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_12 := arithEq_of_rows h (row := 951) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_12
  simp only [← c59001, ← c59002, ← c59003] at e_951_12
  have e_951_13 := arithEq_of_rows h (row := 951) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_13
  simp only [← c59004, ← c59005, k2, ← c59006] at e_951_13
  have hr := e_951_13
  simp only [e_951_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7715)) := by
  have c59007 := (publicBatchWrapper86_copies1843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59008 := (publicBatchWrapper86_copies1844 a h).1
  have c59009 := (publicBatchWrapper86_copies1844 a h).2.1
  have c59010 := (publicBatchWrapper86_copies1844 a h).2.2.1
  have c59011 := (publicBatchWrapper86_copies1844 a h).2.2.2.1
  have c59012 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_14 := arithEq_of_rows h (row := 951) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_14
  simp only [← c59007, ← c59008, ← c59009] at e_951_14
  have e_951_15 := arithEq_of_rows h (row := 951) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_15
  simp only [← c59010, ← c59011, k2, ← c59012] at e_951_15
  have hr := e_951_15
  simp only [e_951_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7716)) := by
  have c59013 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.1
  have c59014 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.1
  have c59015 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.1
  have c59016 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.1
  have c59017 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.1
  have c59018 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_16 := arithEq_of_rows h (row := 951) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_16
  simp only [← c59013, ← c59014, ← c59015] at e_951_16
  have e_951_17 := arithEq_of_rows h (row := 951) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_17
  simp only [← c59016, ← c59017, k2, ← c59018] at e_951_17
  have hr := e_951_17
  simp only [e_951_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 951 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7717)) := by
  have c59019 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59020 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59021 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59022 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59023 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59024 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_951_18 := arithEq_of_rows h (row := 951) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_18
  simp only [← c59019, ← c59020, ← c59021] at e_951_18
  have e_951_19 := arithEq_of_rows h (row := 951) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_951_19
  simp only [← c59022, ← c59023, k2, ← c59024] at e_951_19
  have hr := e_951_19
  simp only [e_951_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7718)) := by
  have c59025 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59026 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59027 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59028 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59029 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59030 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_0 := arithEq_of_rows h (row := 952) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_0
  simp only [← c59025, ← c59026, ← c59027] at e_952_0
  have e_952_1 := arithEq_of_rows h (row := 952) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_1
  simp only [← c59028, ← c59029, k2, ← c59030] at e_952_1
  have hr := e_952_1
  simp only [e_952_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7719)) := by
  have c59031 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59032 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59033 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59034 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59035 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59036 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_2 := arithEq_of_rows h (row := 952) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_2
  simp only [← c59031, ← c59032, ← c59033] at e_952_2
  have e_952_3 := arithEq_of_rows h (row := 952) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_3
  simp only [← c59034, ← c59035, k2, ← c59036] at e_952_3
  have hr := e_952_3
  simp only [e_952_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7720)) := by
  have c59037 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59038 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59039 := (publicBatchWrapper86_copies1844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59040 := (publicBatchWrapper86_copies1845 a h).1
  have c59041 := (publicBatchWrapper86_copies1845 a h).2.1
  have c59042 := (publicBatchWrapper86_copies1845 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_4 := arithEq_of_rows h (row := 952) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_4
  simp only [← c59037, ← c59038, ← c59039] at e_952_4
  have e_952_5 := arithEq_of_rows h (row := 952) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_5
  simp only [← c59040, ← c59041, k2, ← c59042] at e_952_5
  have hr := e_952_5
  simp only [e_952_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7721)) := by
  have c59043 := (publicBatchWrapper86_copies1845 a h).2.2.2.1
  have c59044 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.1
  have c59045 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.1
  have c59046 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.1
  have c59047 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.1
  have c59048 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_6 := arithEq_of_rows h (row := 952) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_6
  simp only [← c59043, ← c59044, ← c59045] at e_952_6
  have e_952_7 := arithEq_of_rows h (row := 952) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_7
  simp only [← c59046, ← c59047, k2, ← c59048] at e_952_7
  have hr := e_952_7
  simp only [e_952_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7722)) := by
  have c59049 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.1
  have c59050 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59051 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59052 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59053 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59054 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_8 := arithEq_of_rows h (row := 952) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_8
  simp only [← c59049, ← c59050, ← c59051] at e_952_8
  have e_952_9 := arithEq_of_rows h (row := 952) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_9
  simp only [← c59052, ← c59053, k2, ← c59054] at e_952_9
  have hr := e_952_9
  simp only [e_952_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7723)) := by
  have c59055 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59056 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59057 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59058 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59059 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59060 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_10 := arithEq_of_rows h (row := 952) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_10
  simp only [← c59055, ← c59056, ← c59057] at e_952_10
  have e_952_11 := arithEq_of_rows h (row := 952) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_11
  simp only [← c59058, ← c59059, k2, ← c59060] at e_952_11
  have hr := e_952_11
  simp only [e_952_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7724)) := by
  have c59061 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59062 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59063 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59064 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59065 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59066 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_12 := arithEq_of_rows h (row := 952) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_12
  simp only [← c59061, ← c59062, ← c59063] at e_952_12
  have e_952_13 := arithEq_of_rows h (row := 952) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_13
  simp only [← c59064, ← c59065, k2, ← c59066] at e_952_13
  have hr := e_952_13
  simp only [e_952_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7725)) := by
  have c59067 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59068 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59069 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59070 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59071 := (publicBatchWrapper86_copies1845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59072 := (publicBatchWrapper86_copies1846 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_14 := arithEq_of_rows h (row := 952) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_14
  simp only [← c59067, ← c59068, ← c59069] at e_952_14
  have e_952_15 := arithEq_of_rows h (row := 952) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_15
  simp only [← c59070, ← c59071, k2, ← c59072] at e_952_15
  have hr := e_952_15
  simp only [e_952_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7726)) := by
  have c59073 := (publicBatchWrapper86_copies1846 a h).2.1
  have c59074 := (publicBatchWrapper86_copies1846 a h).2.2.1
  have c59075 := (publicBatchWrapper86_copies1846 a h).2.2.2.1
  have c59076 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.1
  have c59077 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.1
  have c59078 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_16 := arithEq_of_rows h (row := 952) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_16
  simp only [← c59073, ← c59074, ← c59075] at e_952_16
  have e_952_17 := arithEq_of_rows h (row := 952) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_17
  simp only [← c59076, ← c59077, k2, ← c59078] at e_952_17
  have hr := e_952_17
  simp only [e_952_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 952 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7727)) := by
  have c59079 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.1
  have c59080 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.1
  have c59081 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.1
  have c59082 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59083 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59084 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_952_18 := arithEq_of_rows h (row := 952) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_18
  simp only [← c59079, ← c59080, ← c59081] at e_952_18
  have e_952_19 := arithEq_of_rows h (row := 952) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_952_19
  simp only [← c59082, ← c59083, k2, ← c59084] at e_952_19
  have hr := e_952_19
  simp only [e_952_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7728)) := by
  have c59085 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59086 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59087 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59088 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59089 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59090 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_0 := arithEq_of_rows h (row := 953) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_0
  simp only [← c59085, ← c59086, ← c59087] at e_953_0
  have e_953_1 := arithEq_of_rows h (row := 953) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_1
  simp only [← c59088, ← c59089, k2, ← c59090] at e_953_1
  have hr := e_953_1
  simp only [e_953_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7729)) := by
  have c59091 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59092 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59093 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59094 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59095 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59096 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_2 := arithEq_of_rows h (row := 953) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_2
  simp only [← c59091, ← c59092, ← c59093] at e_953_2
  have e_953_3 := arithEq_of_rows h (row := 953) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_3
  simp only [← c59094, ← c59095, k2, ← c59096] at e_953_3
  have hr := e_953_3
  simp only [e_953_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7730)) := by
  have c59097 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59098 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59099 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59100 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59101 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59102 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_4 := arithEq_of_rows h (row := 953) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_4
  simp only [← c59097, ← c59098, ← c59099] at e_953_4
  have e_953_5 := arithEq_of_rows h (row := 953) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_5
  simp only [← c59100, ← c59101, k2, ← c59102] at e_953_5
  have hr := e_953_5
  simp only [e_953_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7731)) := by
  have c59103 := (publicBatchWrapper86_copies1846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59104 := (publicBatchWrapper86_copies1847 a h).1
  have c59105 := (publicBatchWrapper86_copies1847 a h).2.1
  have c59106 := (publicBatchWrapper86_copies1847 a h).2.2.1
  have c59107 := (publicBatchWrapper86_copies1847 a h).2.2.2.1
  have c59108 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_6 := arithEq_of_rows h (row := 953) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_6
  simp only [← c59103, ← c59104, ← c59105] at e_953_6
  have e_953_7 := arithEq_of_rows h (row := 953) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_7
  simp only [← c59106, ← c59107, k2, ← c59108] at e_953_7
  have hr := e_953_7
  simp only [e_953_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7732)) := by
  have c59109 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.1
  have c59110 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.1
  have c59111 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.1
  have c59112 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.1
  have c59113 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.1
  have c59114 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_8 := arithEq_of_rows h (row := 953) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_8
  simp only [← c59109, ← c59110, ← c59111] at e_953_8
  have e_953_9 := arithEq_of_rows h (row := 953) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_9
  simp only [← c59112, ← c59113, k2, ← c59114] at e_953_9
  have hr := e_953_9
  simp only [e_953_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7733)) := by
  have c59115 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59116 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59117 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59118 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59119 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59120 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_10 := arithEq_of_rows h (row := 953) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_10
  simp only [← c59115, ← c59116, ← c59117] at e_953_10
  have e_953_11 := arithEq_of_rows h (row := 953) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_11
  simp only [← c59118, ← c59119, k2, ← c59120] at e_953_11
  have hr := e_953_11
  simp only [e_953_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7734)) := by
  have c59121 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59122 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59123 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59124 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59125 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59126 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_12 := arithEq_of_rows h (row := 953) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_12
  simp only [← c59121, ← c59122, ← c59123] at e_953_12
  have e_953_13 := arithEq_of_rows h (row := 953) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_13
  simp only [← c59124, ← c59125, k2, ← c59126] at e_953_13
  have hr := e_953_13
  simp only [e_953_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7735)) := by
  have c59127 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59128 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59129 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59130 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59131 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59132 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_14 := arithEq_of_rows h (row := 953) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_14
  simp only [← c59127, ← c59128, ← c59129] at e_953_14
  have e_953_15 := arithEq_of_rows h (row := 953) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_15
  simp only [← c59130, ← c59131, k2, ← c59132] at e_953_15
  have hr := e_953_15
  simp only [e_953_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7736)) := by
  have c59133 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59134 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59135 := (publicBatchWrapper86_copies1847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59136 := (publicBatchWrapper86_copies1848 a h).1
  have c59137 := (publicBatchWrapper86_copies1848 a h).2.1
  have c59138 := (publicBatchWrapper86_copies1848 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_16 := arithEq_of_rows h (row := 953) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_16
  simp only [← c59133, ← c59134, ← c59135] at e_953_16
  have e_953_17 := arithEq_of_rows h (row := 953) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_17
  simp only [← c59136, ← c59137, k2, ← c59138] at e_953_17
  have hr := e_953_17
  simp only [e_953_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 953 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7737)) := by
  have c59139 := (publicBatchWrapper86_copies1848 a h).2.2.2.1
  have c59140 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.1
  have c59141 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.1
  have c59142 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.1
  have c59143 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.1
  have c59144 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_953_18 := arithEq_of_rows h (row := 953) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_18
  simp only [← c59139, ← c59140, ← c59141] at e_953_18
  have e_953_19 := arithEq_of_rows h (row := 953) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_953_19
  simp only [← c59142, ← c59143, k2, ← c59144] at e_953_19
  have hr := e_953_19
  simp only [e_953_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7738)) := by
  have c59145 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.1
  have c59146 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59147 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59148 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59149 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59150 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_0 := arithEq_of_rows h (row := 954) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_0
  simp only [← c59145, ← c59146, ← c59147] at e_954_0
  have e_954_1 := arithEq_of_rows h (row := 954) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_1
  simp only [← c59148, ← c59149, k2, ← c59150] at e_954_1
  have hr := e_954_1
  simp only [e_954_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7739)) := by
  have c59151 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59152 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59153 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59154 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59155 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59156 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_2 := arithEq_of_rows h (row := 954) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_2
  simp only [← c59151, ← c59152, ← c59153] at e_954_2
  have e_954_3 := arithEq_of_rows h (row := 954) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_3
  simp only [← c59154, ← c59155, k2, ← c59156] at e_954_3
  have hr := e_954_3
  simp only [e_954_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7740)) := by
  have c59157 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59158 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59159 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59160 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59161 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59162 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_4 := arithEq_of_rows h (row := 954) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_4
  simp only [← c59157, ← c59158, ← c59159] at e_954_4
  have e_954_5 := arithEq_of_rows h (row := 954) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_5
  simp only [← c59160, ← c59161, k2, ← c59162] at e_954_5
  have hr := e_954_5
  simp only [e_954_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7741)) := by
  have c59163 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59164 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59165 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59166 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59167 := (publicBatchWrapper86_copies1848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59168 := (publicBatchWrapper86_copies1849 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_6 := arithEq_of_rows h (row := 954) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_6
  simp only [← c59163, ← c59164, ← c59165] at e_954_6
  have e_954_7 := arithEq_of_rows h (row := 954) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_7
  simp only [← c59166, ← c59167, k2, ← c59168] at e_954_7
  have hr := e_954_7
  simp only [e_954_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7742)) := by
  have c59169 := (publicBatchWrapper86_copies1849 a h).2.1
  have c59170 := (publicBatchWrapper86_copies1849 a h).2.2.1
  have c59171 := (publicBatchWrapper86_copies1849 a h).2.2.2.1
  have c59172 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.1
  have c59173 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.1
  have c59174 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_8 := arithEq_of_rows h (row := 954) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_8
  simp only [← c59169, ← c59170, ← c59171] at e_954_8
  have e_954_9 := arithEq_of_rows h (row := 954) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_9
  simp only [← c59172, ← c59173, k2, ← c59174] at e_954_9
  have hr := e_954_9
  simp only [e_954_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7743)) := by
  have c59175 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.1
  have c59176 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.1
  have c59177 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.1
  have c59178 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59179 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59180 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_10 := arithEq_of_rows h (row := 954) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_10
  simp only [← c59175, ← c59176, ← c59177] at e_954_10
  have e_954_11 := arithEq_of_rows h (row := 954) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_11
  simp only [← c59178, ← c59179, k2, ← c59180] at e_954_11
  have hr := e_954_11
  simp only [e_954_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7744)) := by
  have c59181 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59182 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59183 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59184 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59185 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59186 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_12 := arithEq_of_rows h (row := 954) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_12
  simp only [← c59181, ← c59182, ← c59183] at e_954_12
  have e_954_13 := arithEq_of_rows h (row := 954) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_13
  simp only [← c59184, ← c59185, k2, ← c59186] at e_954_13
  have hr := e_954_13
  simp only [e_954_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7745)) := by
  have c59187 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59188 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59189 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59190 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59191 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59192 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_14 := arithEq_of_rows h (row := 954) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_14
  simp only [← c59187, ← c59188, ← c59189] at e_954_14
  have e_954_15 := arithEq_of_rows h (row := 954) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_15
  simp only [← c59190, ← c59191, k2, ← c59192] at e_954_15
  have hr := e_954_15
  simp only [e_954_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7746)) := by
  have c59193 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59194 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59195 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59196 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59197 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59198 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_16 := arithEq_of_rows h (row := 954) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_16
  simp only [← c59193, ← c59194, ← c59195] at e_954_16
  have e_954_17 := arithEq_of_rows h (row := 954) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_17
  simp only [← c59196, ← c59197, k2, ← c59198] at e_954_17
  have hr := e_954_17
  simp only [e_954_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 954 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7747)) := by
  have c59199 := (publicBatchWrapper86_copies1849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59200 := (publicBatchWrapper86_copies1850 a h).1
  have c59201 := (publicBatchWrapper86_copies1850 a h).2.1
  have c59202 := (publicBatchWrapper86_copies1850 a h).2.2.1
  have c59203 := (publicBatchWrapper86_copies1850 a h).2.2.2.1
  have c59204 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_954_18 := arithEq_of_rows h (row := 954) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_18
  simp only [← c59199, ← c59200, ← c59201] at e_954_18
  have e_954_19 := arithEq_of_rows h (row := 954) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_954_19
  simp only [← c59202, ← c59203, k2, ← c59204] at e_954_19
  have hr := e_954_19
  simp only [e_954_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7748)) := by
  have c59205 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.1
  have c59206 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.1
  have c59207 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.1
  have c59208 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.1
  have c59209 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.1
  have c59210 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_0 := arithEq_of_rows h (row := 955) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_0
  simp only [← c59205, ← c59206, ← c59207] at e_955_0
  have e_955_1 := arithEq_of_rows h (row := 955) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_1
  simp only [← c59208, ← c59209, k2, ← c59210] at e_955_1
  have hr := e_955_1
  simp only [e_955_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7749)) := by
  have c59211 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59212 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59213 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59214 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59215 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59216 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_2 := arithEq_of_rows h (row := 955) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_2
  simp only [← c59211, ← c59212, ← c59213] at e_955_2
  have e_955_3 := arithEq_of_rows h (row := 955) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_3
  simp only [← c59214, ← c59215, k2, ← c59216] at e_955_3
  have hr := e_955_3
  simp only [e_955_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7750)) := by
  have c59217 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59218 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59219 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59220 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59221 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59222 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_4 := arithEq_of_rows h (row := 955) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_4
  simp only [← c59217, ← c59218, ← c59219] at e_955_4
  have e_955_5 := arithEq_of_rows h (row := 955) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_5
  simp only [← c59220, ← c59221, k2, ← c59222] at e_955_5
  have hr := e_955_5
  simp only [e_955_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7751)) := by
  have c59223 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59224 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59225 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59226 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59227 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59228 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_6 := arithEq_of_rows h (row := 955) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_6
  simp only [← c59223, ← c59224, ← c59225] at e_955_6
  have e_955_7 := arithEq_of_rows h (row := 955) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_7
  simp only [← c59226, ← c59227, k2, ← c59228] at e_955_7
  have hr := e_955_7
  simp only [e_955_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7752)) := by
  have c59229 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59230 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59231 := (publicBatchWrapper86_copies1850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59232 := (publicBatchWrapper86_copies1851 a h).1
  have c59233 := (publicBatchWrapper86_copies1851 a h).2.1
  have c59234 := (publicBatchWrapper86_copies1851 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_8 := arithEq_of_rows h (row := 955) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_8
  simp only [← c59229, ← c59230, ← c59231] at e_955_8
  have e_955_9 := arithEq_of_rows h (row := 955) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_9
  simp only [← c59232, ← c59233, k2, ← c59234] at e_955_9
  have hr := e_955_9
  simp only [e_955_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7753)) := by
  have c59235 := (publicBatchWrapper86_copies1851 a h).2.2.2.1
  have c59236 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.1
  have c59237 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.1
  have c59238 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.1
  have c59239 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.1
  have c59240 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_10 := arithEq_of_rows h (row := 955) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_10
  simp only [← c59235, ← c59236, ← c59237] at e_955_10
  have e_955_11 := arithEq_of_rows h (row := 955) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_11
  simp only [← c59238, ← c59239, k2, ← c59240] at e_955_11
  have hr := e_955_11
  simp only [e_955_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7754)) := by
  have c59241 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.1
  have c59242 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59243 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59244 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59245 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59246 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_12 := arithEq_of_rows h (row := 955) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_12
  simp only [← c59241, ← c59242, ← c59243] at e_955_12
  have e_955_13 := arithEq_of_rows h (row := 955) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_13
  simp only [← c59244, ← c59245, k2, ← c59246] at e_955_13
  have hr := e_955_13
  simp only [e_955_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7755)) := by
  have c59247 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59248 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59249 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59250 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59251 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59252 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_14 := arithEq_of_rows h (row := 955) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_14
  simp only [← c59247, ← c59248, ← c59249] at e_955_14
  have e_955_15 := arithEq_of_rows h (row := 955) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_15
  simp only [← c59250, ← c59251, k2, ← c59252] at e_955_15
  have hr := e_955_15
  simp only [e_955_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7756)) := by
  have c59253 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59254 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59255 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59256 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59257 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59258 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_16 := arithEq_of_rows h (row := 955) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_16
  simp only [← c59253, ← c59254, ← c59255] at e_955_16
  have e_955_17 := arithEq_of_rows h (row := 955) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_17
  simp only [← c59256, ← c59257, k2, ← c59258] at e_955_17
  have hr := e_955_17
  simp only [e_955_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 955 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7757)) := by
  have c59259 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59260 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59261 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59262 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59263 := (publicBatchWrapper86_copies1851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59264 := (publicBatchWrapper86_copies1852 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_955_18 := arithEq_of_rows h (row := 955) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_18
  simp only [← c59259, ← c59260, ← c59261] at e_955_18
  have e_955_19 := arithEq_of_rows h (row := 955) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_955_19
  simp only [← c59262, ← c59263, k2, ← c59264] at e_955_19
  have hr := e_955_19
  simp only [e_955_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7758)) := by
  have c59265 := (publicBatchWrapper86_copies1852 a h).2.1
  have c59266 := (publicBatchWrapper86_copies1852 a h).2.2.1
  have c59267 := (publicBatchWrapper86_copies1852 a h).2.2.2.1
  have c59268 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.1
  have c59269 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.1
  have c59270 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_0 := arithEq_of_rows h (row := 956) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_0
  simp only [← c59265, ← c59266, ← c59267] at e_956_0
  have e_956_1 := arithEq_of_rows h (row := 956) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_1
  simp only [← c59268, ← c59269, k2, ← c59270] at e_956_1
  have hr := e_956_1
  simp only [e_956_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7759)) := by
  have c59271 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.1
  have c59272 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.1
  have c59273 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.1
  have c59274 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59275 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59276 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_2 := arithEq_of_rows h (row := 956) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_2
  simp only [← c59271, ← c59272, ← c59273] at e_956_2
  have e_956_3 := arithEq_of_rows h (row := 956) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_3
  simp only [← c59274, ← c59275, k2, ← c59276] at e_956_3
  have hr := e_956_3
  simp only [e_956_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7760)) := by
  have c59277 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59278 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59279 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59280 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59281 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59282 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_4 := arithEq_of_rows h (row := 956) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_4
  simp only [← c59277, ← c59278, ← c59279] at e_956_4
  have e_956_5 := arithEq_of_rows h (row := 956) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_5
  simp only [← c59280, ← c59281, k2, ← c59282] at e_956_5
  have hr := e_956_5
  simp only [e_956_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7761)) := by
  have c59283 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59284 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59285 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59286 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59287 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59288 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_6 := arithEq_of_rows h (row := 956) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_6
  simp only [← c59283, ← c59284, ← c59285] at e_956_6
  have e_956_7 := arithEq_of_rows h (row := 956) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_7
  simp only [← c59286, ← c59287, k2, ← c59288] at e_956_7
  have hr := e_956_7
  simp only [e_956_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7762)) := by
  have c59289 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59290 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59291 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59292 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59293 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59294 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_8 := arithEq_of_rows h (row := 956) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_8
  simp only [← c59289, ← c59290, ← c59291] at e_956_8
  have e_956_9 := arithEq_of_rows h (row := 956) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_9
  simp only [← c59292, ← c59293, k2, ← c59294] at e_956_9
  have hr := e_956_9
  simp only [e_956_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7763)) := by
  have c59295 := (publicBatchWrapper86_copies1852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59296 := (publicBatchWrapper86_copies1853 a h).1
  have c59297 := (publicBatchWrapper86_copies1853 a h).2.1
  have c59298 := (publicBatchWrapper86_copies1853 a h).2.2.1
  have c59299 := (publicBatchWrapper86_copies1853 a h).2.2.2.1
  have c59300 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_10 := arithEq_of_rows h (row := 956) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_10
  simp only [← c59295, ← c59296, ← c59297] at e_956_10
  have e_956_11 := arithEq_of_rows h (row := 956) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_11
  simp only [← c59298, ← c59299, k2, ← c59300] at e_956_11
  have hr := e_956_11
  simp only [e_956_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7764)) := by
  have c59301 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.1
  have c59302 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.1
  have c59303 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.1
  have c59304 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.1
  have c59305 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.1
  have c59306 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_12 := arithEq_of_rows h (row := 956) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_12
  simp only [← c59301, ← c59302, ← c59303] at e_956_12
  have e_956_13 := arithEq_of_rows h (row := 956) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_13
  simp only [← c59304, ← c59305, k2, ← c59306] at e_956_13
  have hr := e_956_13
  simp only [e_956_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7765)) := by
  have c59307 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59308 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59309 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59310 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59311 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59312 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_14 := arithEq_of_rows h (row := 956) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_14
  simp only [← c59307, ← c59308, ← c59309] at e_956_14
  have e_956_15 := arithEq_of_rows h (row := 956) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_15
  simp only [← c59310, ← c59311, k2, ← c59312] at e_956_15
  have hr := e_956_15
  simp only [e_956_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7766)) := by
  have c59313 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59314 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59315 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59316 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59317 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59318 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_16 := arithEq_of_rows h (row := 956) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_16
  simp only [← c59313, ← c59314, ← c59315] at e_956_16
  have e_956_17 := arithEq_of_rows h (row := 956) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_17
  simp only [← c59316, ← c59317, k2, ← c59318] at e_956_17
  have hr := e_956_17
  simp only [e_956_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 956 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7767)) := by
  have c59319 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59320 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59321 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59322 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59323 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59324 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_956_18 := arithEq_of_rows h (row := 956) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_18
  simp only [← c59319, ← c59320, ← c59321] at e_956_18
  have e_956_19 := arithEq_of_rows h (row := 956) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_956_19
  simp only [← c59322, ← c59323, k2, ← c59324] at e_956_19
  have hr := e_956_19
  simp only [e_956_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7768)) := by
  have c59325 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59326 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59327 := (publicBatchWrapper86_copies1853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59328 := (publicBatchWrapper86_copies1854 a h).1
  have c59329 := (publicBatchWrapper86_copies1854 a h).2.1
  have c59330 := (publicBatchWrapper86_copies1854 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_0 := arithEq_of_rows h (row := 957) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_0
  simp only [← c59325, ← c59326, ← c59327] at e_957_0
  have e_957_1 := arithEq_of_rows h (row := 957) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_1
  simp only [← c59328, ← c59329, k2, ← c59330] at e_957_1
  have hr := e_957_1
  simp only [e_957_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7769)) := by
  have c59331 := (publicBatchWrapper86_copies1854 a h).2.2.2.1
  have c59332 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.1
  have c59333 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.1
  have c59334 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.1
  have c59335 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.1
  have c59336 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_2 := arithEq_of_rows h (row := 957) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_2
  simp only [← c59331, ← c59332, ← c59333] at e_957_2
  have e_957_3 := arithEq_of_rows h (row := 957) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_3
  simp only [← c59334, ← c59335, k2, ← c59336] at e_957_3
  have hr := e_957_3
  simp only [e_957_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7810)) := by
  have c59337 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.1
  have c59338 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59339 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59340 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59341 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59342 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_4 := arithEq_of_rows h (row := 957) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_4
  simp only [← c59337, ← c59338, ← c59339] at e_957_4
  have e_957_5 := arithEq_of_rows h (row := 957) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_5
  simp only [← c59340, ← c59341, k2, ← c59342] at e_957_5
  have hr := e_957_5
  simp only [e_957_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7811)) := by
  have c59343 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59344 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59345 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59346 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59347 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59348 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_6 := arithEq_of_rows h (row := 957) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_6
  simp only [← c59343, ← c59344, ← c59345] at e_957_6
  have e_957_7 := arithEq_of_rows h (row := 957) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_7
  simp only [← c59346, ← c59347, k2, ← c59348] at e_957_7
  have hr := e_957_7
  simp only [e_957_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7812)) := by
  have c59349 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59350 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59351 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59352 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59353 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59354 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_8 := arithEq_of_rows h (row := 957) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_8
  simp only [← c59349, ← c59350, ← c59351] at e_957_8
  have e_957_9 := arithEq_of_rows h (row := 957) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_9
  simp only [← c59352, ← c59353, k2, ← c59354] at e_957_9
  have hr := e_957_9
  simp only [e_957_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 47) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7813)) := by
  have c59355 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59356 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59357 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59358 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59359 := (publicBatchWrapper86_copies1854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59360 := (publicBatchWrapper86_copies1855 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_10 := arithEq_of_rows h (row := 957) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_10
  simp only [← c59355, ← c59356, ← c59357] at e_957_10
  have e_957_11 := arithEq_of_rows h (row := 957) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_11
  simp only [← c59358, ← c59359, k2, ← c59360] at e_957_11
  have hr := e_957_11
  simp only [e_957_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 55) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7814)) := by
  have c59361 := (publicBatchWrapper86_copies1855 a h).2.1
  have c59362 := (publicBatchWrapper86_copies1855 a h).2.2.1
  have c59363 := (publicBatchWrapper86_copies1855 a h).2.2.2.1
  have c59364 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.1
  have c59365 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.1
  have c59366 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_12 := arithEq_of_rows h (row := 957) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_12
  simp only [← c59361, ← c59362, ← c59363] at e_957_12
  have e_957_13 := arithEq_of_rows h (row := 957) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_13
  simp only [← c59364, ← c59365, k2, ← c59366] at e_957_13
  have hr := e_957_13
  simp only [e_957_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 63) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7815)) := by
  have c59367 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.1
  have c59368 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.1
  have c59369 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.1
  have c59370 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59371 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59372 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_14 := arithEq_of_rows h (row := 957) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_14
  simp only [← c59367, ← c59368, ← c59369] at e_957_14
  have e_957_15 := arithEq_of_rows h (row := 957) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_15
  simp only [← c59370, ← c59371, k2, ← c59372] at e_957_15
  have hr := e_957_15
  simp only [e_957_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 71) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7816)) := by
  have c59373 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59374 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59375 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59376 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59377 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59378 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_16 := arithEq_of_rows h (row := 957) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_16
  simp only [← c59373, ← c59374, ← c59375] at e_957_16
  have e_957_17 := arithEq_of_rows h (row := 957) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_17
  simp only [← c59376, ← c59377, k2, ← c59378] at e_957_17
  have hr := e_957_17
  simp only [e_957_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 957 79) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7817)) := by
  have c59379 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59380 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59381 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59382 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59383 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59384 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_957_18 := arithEq_of_rows h (row := 957) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_18
  simp only [← c59379, ← c59380, ← c59381] at e_957_18
  have e_957_19 := arithEq_of_rows h (row := 957) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_957_19
  simp only [← c59382, ← c59383, k2, ← c59384] at e_957_19
  have hr := e_957_19
  simp only [e_957_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7818)) := by
  have c59385 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59386 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59387 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59388 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59389 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59390 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_0 := arithEq_of_rows h (row := 958) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_0
  simp only [← c59385, ← c59386, ← c59387] at e_958_0
  have e_958_1 := arithEq_of_rows h (row := 958) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_1
  simp only [← c59388, ← c59389, k2, ← c59390] at e_958_1
  have hr := e_958_1
  simp only [e_958_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7819)) := by
  have c59391 := (publicBatchWrapper86_copies1855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59392 := (publicBatchWrapper86_copies1856 a h).1
  have c59393 := (publicBatchWrapper86_copies1856 a h).2.1
  have c59394 := (publicBatchWrapper86_copies1856 a h).2.2.1
  have c59395 := (publicBatchWrapper86_copies1856 a h).2.2.2.1
  have c59396 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_2 := arithEq_of_rows h (row := 958) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_2
  simp only [← c59391, ← c59392, ← c59393] at e_958_2
  have e_958_3 := arithEq_of_rows h (row := 958) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_3
  simp only [← c59394, ← c59395, k2, ← c59396] at e_958_3
  have hr := e_958_3
  simp only [e_958_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7820)) := by
  have c59397 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.1
  have c59398 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.1
  have c59399 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.1
  have c59400 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.1
  have c59401 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.1
  have c59402 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_4 := arithEq_of_rows h (row := 958) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_4
  simp only [← c59397, ← c59398, ← c59399] at e_958_4
  have e_958_5 := arithEq_of_rows h (row := 958) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_5
  simp only [← c59400, ← c59401, k2, ← c59402] at e_958_5
  have hr := e_958_5
  simp only [e_958_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7821)) := by
  have c59403 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59404 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59405 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59406 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59407 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59408 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_6 := arithEq_of_rows h (row := 958) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_6
  simp only [← c59403, ← c59404, ← c59405] at e_958_6
  have e_958_7 := arithEq_of_rows h (row := 958) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_7
  simp only [← c59406, ← c59407, k2, ← c59408] at e_958_7
  have hr := e_958_7
  simp only [e_958_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7822)) := by
  have c59409 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59410 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59411 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59412 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59413 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59414 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_8 := arithEq_of_rows h (row := 958) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_8
  simp only [← c59409, ← c59410, ← c59411] at e_958_8
  have e_958_9 := arithEq_of_rows h (row := 958) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_9
  simp only [← c59412, ← c59413, k2, ← c59414] at e_958_9
  have hr := e_958_9
  simp only [e_958_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 47) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7823)) := by
  have c59415 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59416 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59417 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59418 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59419 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59420 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_10 := arithEq_of_rows h (row := 958) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_10
  simp only [← c59415, ← c59416, ← c59417] at e_958_10
  have e_958_11 := arithEq_of_rows h (row := 958) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_11
  simp only [← c59418, ← c59419, k2, ← c59420] at e_958_11
  have hr := e_958_11
  simp only [e_958_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 55) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7824)) := by
  have c59421 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59422 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59423 := (publicBatchWrapper86_copies1856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59424 := (publicBatchWrapper86_copies1857 a h).1
  have c59425 := (publicBatchWrapper86_copies1857 a h).2.1
  have c59426 := (publicBatchWrapper86_copies1857 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_12 := arithEq_of_rows h (row := 958) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_12
  simp only [← c59421, ← c59422, ← c59423] at e_958_12
  have e_958_13 := arithEq_of_rows h (row := 958) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_13
  simp only [← c59424, ← c59425, k2, ← c59426] at e_958_13
  have hr := e_958_13
  simp only [e_958_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 63) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7825)) := by
  have c59427 := (publicBatchWrapper86_copies1857 a h).2.2.2.1
  have c59428 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.1
  have c59429 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.1
  have c59430 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.1
  have c59431 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.1
  have c59432 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_14 := arithEq_of_rows h (row := 958) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_14
  simp only [← c59427, ← c59428, ← c59429] at e_958_14
  have e_958_15 := arithEq_of_rows h (row := 958) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_15
  simp only [← c59430, ← c59431, k2, ← c59432] at e_958_15
  have hr := e_958_15
  simp only [e_958_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 71) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7826)) := by
  have c59433 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.1
  have c59434 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59435 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59436 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59437 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59438 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_16 := arithEq_of_rows h (row := 958) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_16
  simp only [← c59433, ← c59434, ← c59435] at e_958_16
  have e_958_17 := arithEq_of_rows h (row := 958) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_17
  simp only [← c59436, ← c59437, k2, ← c59438] at e_958_17
  have hr := e_958_17
  simp only [e_958_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 958 79) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7827)) := by
  have c59439 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59440 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59441 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59442 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59443 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59444 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_958_18 := arithEq_of_rows h (row := 958) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_18
  simp only [← c59439, ← c59440, ← c59441] at e_958_18
  have e_958_19 := arithEq_of_rows h (row := 958) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_958_19
  simp only [← c59442, ← c59443, k2, ← c59444] at e_958_19
  have hr := e_958_19
  simp only [e_958_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 959 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7828)) := by
  have c59445 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59446 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59447 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59448 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59449 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59450 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_959_0 := arithEq_of_rows h (row := 959) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_0
  simp only [← c59445, ← c59446, ← c59447] at e_959_0
  have e_959_1 := arithEq_of_rows h (row := 959) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_1
  simp only [← c59448, ← c59449, k2, ← c59450] at e_959_1
  have hr := e_959_1
  simp only [e_959_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 959 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7829)) := by
  have c59451 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59452 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59453 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59454 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59455 := (publicBatchWrapper86_copies1857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c59456 := (publicBatchWrapper86_copies1858 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_959_2 := arithEq_of_rows h (row := 959) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_2
  simp only [← c59451, ← c59452, ← c59453] at e_959_2
  have e_959_3 := arithEq_of_rows h (row := 959) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_3
  simp only [← c59454, ← c59455, k2, ← c59456] at e_959_3
  have hr := e_959_3
  simp only [e_959_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 959 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7830)) := by
  have c59457 := (publicBatchWrapper86_copies1858 a h).2.1
  have c59458 := (publicBatchWrapper86_copies1858 a h).2.2.1
  have c59459 := (publicBatchWrapper86_copies1858 a h).2.2.2.1
  have c59460 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.1
  have c59461 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.1
  have c59462 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_959_4 := arithEq_of_rows h (row := 959) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_4
  simp only [← c59457, ← c59458, ← c59459] at e_959_4
  have e_959_5 := arithEq_of_rows h (row := 959) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_5
  simp only [← c59460, ← c59461, k2, ← c59462] at e_959_5
  have hr := e_959_5
  simp only [e_959_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 959 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7831)) := by
  have c59463 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.1
  have c59464 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.1
  have c59465 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.1
  have c59466 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.1
  have c59467 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c59468 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_959_6 := arithEq_of_rows h (row := 959) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_6
  simp only [← c59463, ← c59464, ← c59465] at e_959_6
  have e_959_7 := arithEq_of_rows h (row := 959) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_7
  simp only [← c59466, ← c59467, k2, ← c59468] at e_959_7
  have hr := e_959_7
  simp only [e_959_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f9186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 959 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7832)) := by
  have c59469 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59470 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59471 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59472 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59473 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c59474 := (publicBatchWrapper86_copies1858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_959_8 := arithEq_of_rows h (row := 959) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_8
  simp only [← c59469, ← c59470, ← c59471] at e_959_8
  have e_959_9 := arithEq_of_rows h (row := 959) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_959_9
  simp only [← c59472, ← c59473, k2, ← c59474] at e_959_9
  have hr := e_959_9
  simp only [e_959_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
