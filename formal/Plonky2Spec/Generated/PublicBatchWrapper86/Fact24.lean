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

theorem publicBatchWrapper86_f18241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9791)) := by
  have c113799 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.1
  have c113800 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.1
  have c113801 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.1
  have c113802 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113803 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113804 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_18
  simp only [← c113799, ← c113800, ← c113801] at e_1864_18
  have e_1864_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_19
  simp only [← c113802, ← c113803, k2, ← c113804] at e_1864_19
  have hr := e_1864_19
  simp only [e_1864_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9792)) := by
  have c113805 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113806 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113807 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113808 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113809 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113810 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_0
  simp only [← c113805, ← c113806, ← c113807] at e_1865_0
  have e_1865_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_1
  simp only [← c113808, ← c113809, k2, ← c113810] at e_1865_1
  have hr := e_1865_1
  simp only [e_1865_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9793)) := by
  have c113811 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113812 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113813 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113814 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113815 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113816 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_2
  simp only [← c113811, ← c113812, ← c113813] at e_1865_2
  have e_1865_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_3
  simp only [← c113814, ← c113815, k2, ← c113816] at e_1865_3
  have hr := e_1865_3
  simp only [e_1865_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9794)) := by
  have c113817 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113818 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113819 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113820 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113821 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113822 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_4
  simp only [← c113817, ← c113818, ← c113819] at e_1865_4
  have e_1865_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_5
  simp only [← c113820, ← c113821, k2, ← c113822] at e_1865_5
  have hr := e_1865_5
  simp only [e_1865_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9795)) := by
  have c113823 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113824 := (publicBatchWrapper86_copies3557 a h).1
  have c113825 := (publicBatchWrapper86_copies3557 a h).2.1
  have c113826 := (publicBatchWrapper86_copies3557 a h).2.2.1
  have c113827 := (publicBatchWrapper86_copies3557 a h).2.2.2.1
  have c113828 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_6
  simp only [← c113823, ← c113824, ← c113825] at e_1865_6
  have e_1865_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_7
  simp only [← c113826, ← c113827, k2, ← c113828] at e_1865_7
  have hr := e_1865_7
  simp only [e_1865_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9796)) := by
  have c113829 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.1
  have c113830 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.1
  have c113831 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.1
  have c113832 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.1
  have c113833 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.1
  have c113834 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_8
  simp only [← c113829, ← c113830, ← c113831] at e_1865_8
  have e_1865_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_9
  simp only [← c113832, ← c113833, k2, ← c113834] at e_1865_9
  have hr := e_1865_9
  simp only [e_1865_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9797)) := by
  have c113835 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113836 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113837 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113838 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113839 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113840 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_10
  simp only [← c113835, ← c113836, ← c113837] at e_1865_10
  have e_1865_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_11
  simp only [← c113838, ← c113839, k2, ← c113840] at e_1865_11
  have hr := e_1865_11
  simp only [e_1865_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9798)) := by
  have c113841 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113842 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113843 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113844 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113845 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113846 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_12
  simp only [← c113841, ← c113842, ← c113843] at e_1865_12
  have e_1865_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_13
  simp only [← c113844, ← c113845, k2, ← c113846] at e_1865_13
  have hr := e_1865_13
  simp only [e_1865_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9799)) := by
  have c113847 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113848 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113849 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113850 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113851 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113852 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_14
  simp only [← c113847, ← c113848, ← c113849] at e_1865_14
  have e_1865_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_15
  simp only [← c113850, ← c113851, k2, ← c113852] at e_1865_15
  have hr := e_1865_15
  simp only [e_1865_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9800)) := by
  have c113853 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113854 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113855 := (publicBatchWrapper86_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113856 := (publicBatchWrapper86_copies3558 a h).1
  have c113857 := (publicBatchWrapper86_copies3558 a h).2.1
  have c113858 := (publicBatchWrapper86_copies3558 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_16
  simp only [← c113853, ← c113854, ← c113855] at e_1865_16
  have e_1865_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_17
  simp only [← c113856, ← c113857, k2, ← c113858] at e_1865_17
  have hr := e_1865_17
  simp only [e_1865_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1865 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9801)) := by
  have c113859 := (publicBatchWrapper86_copies3558 a h).2.2.2.1
  have c113860 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.1
  have c113861 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.1
  have c113862 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.1
  have c113863 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.1
  have c113864 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1865_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_18
  simp only [← c113859, ← c113860, ← c113861] at e_1865_18
  have e_1865_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_19
  simp only [← c113862, ← c113863, k2, ← c113864] at e_1865_19
  have hr := e_1865_19
  simp only [e_1865_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 7) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9970)) := by
  have c113865 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.1
  have c113866 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113867 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113868 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113869 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113870 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_0
  simp only [← c113865, ← c113866, ← c113867] at e_1866_0
  have e_1866_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_1
  simp only [← c113868, ← c113869, k2, ← c113870] at e_1866_1
  have hr := e_1866_1
  simp only [e_1866_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 15) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9971)) := by
  have c113871 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113872 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113873 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113874 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113875 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113876 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_2
  simp only [← c113871, ← c113872, ← c113873] at e_1866_2
  have e_1866_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_3
  simp only [← c113874, ← c113875, k2, ← c113876] at e_1866_3
  have hr := e_1866_3
  simp only [e_1866_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 23) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9972)) := by
  have c113877 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113878 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113879 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113880 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113881 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113882 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_4
  simp only [← c113877, ← c113878, ← c113879] at e_1866_4
  have e_1866_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_5
  simp only [← c113880, ← c113881, k2, ← c113882] at e_1866_5
  have hr := e_1866_5
  simp only [e_1866_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 31) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9973)) := by
  have c113883 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113884 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113885 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113886 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113887 := (publicBatchWrapper86_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113888 := (publicBatchWrapper86_copies3559 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_6
  simp only [← c113883, ← c113884, ← c113885] at e_1866_6
  have e_1866_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_7
  simp only [← c113886, ← c113887, k2, ← c113888] at e_1866_7
  have hr := e_1866_7
  simp only [e_1866_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 39) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9974)) := by
  have c113889 := (publicBatchWrapper86_copies3559 a h).2.1
  have c113890 := (publicBatchWrapper86_copies3559 a h).2.2.1
  have c113891 := (publicBatchWrapper86_copies3559 a h).2.2.2.1
  have c113892 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.1
  have c113893 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.1
  have c113894 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_8
  simp only [← c113889, ← c113890, ← c113891] at e_1866_8
  have e_1866_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_9
  simp only [← c113892, ← c113893, k2, ← c113894] at e_1866_9
  have hr := e_1866_9
  simp only [e_1866_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 47) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9975)) := by
  have c113895 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.1
  have c113896 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.1
  have c113897 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.1
  have c113898 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113899 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113900 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_10
  simp only [← c113895, ← c113896, ← c113897] at e_1866_10
  have e_1866_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_11
  simp only [← c113898, ← c113899, k2, ← c113900] at e_1866_11
  have hr := e_1866_11
  simp only [e_1866_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 55) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9976)) := by
  have c113901 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113902 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113903 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113904 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113905 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113906 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_12
  simp only [← c113901, ← c113902, ← c113903] at e_1866_12
  have e_1866_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_13
  simp only [← c113904, ← c113905, k2, ← c113906] at e_1866_13
  have hr := e_1866_13
  simp only [e_1866_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 63) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9977)) := by
  have c113907 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113908 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113909 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113910 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113911 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113912 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_14
  simp only [← c113907, ← c113908, ← c113909] at e_1866_14
  have e_1866_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_15
  simp only [← c113910, ← c113911, k2, ← c113912] at e_1866_15
  have hr := e_1866_15
  simp only [e_1866_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 71) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9978)) := by
  have c113913 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113914 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113915 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113916 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113917 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113918 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_16
  simp only [← c113913, ← c113914, ← c113915] at e_1866_16
  have e_1866_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_17
  simp only [← c113916, ← c113917, k2, ← c113918] at e_1866_17
  have hr := e_1866_17
  simp only [e_1866_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1866 79) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9979)) := by
  have c113919 := (publicBatchWrapper86_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113920 := (publicBatchWrapper86_copies3560 a h).1
  have c113921 := (publicBatchWrapper86_copies3560 a h).2.1
  have c113922 := (publicBatchWrapper86_copies3560 a h).2.2.1
  have c113923 := (publicBatchWrapper86_copies3560 a h).2.2.2.1
  have c113924 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1866_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_18
  simp only [← c113919, ← c113920, ← c113921] at e_1866_18
  have e_1866_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_19
  simp only [← c113922, ← c113923, k2, ← c113924] at e_1866_19
  have hr := e_1866_19
  simp only [e_1866_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 7) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9980)) := by
  have c113925 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.1
  have c113926 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.1
  have c113927 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.1
  have c113928 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.1
  have c113929 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.1
  have c113930 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_0
  simp only [← c113925, ← c113926, ← c113927] at e_1867_0
  have e_1867_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_1
  simp only [← c113928, ← c113929, k2, ← c113930] at e_1867_1
  have hr := e_1867_1
  simp only [e_1867_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 15) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9981)) := by
  have c113931 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113932 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113933 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113934 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113935 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113936 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_2
  simp only [← c113931, ← c113932, ← c113933] at e_1867_2
  have e_1867_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_3
  simp only [← c113934, ← c113935, k2, ← c113936] at e_1867_3
  have hr := e_1867_3
  simp only [e_1867_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 23) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9982)) := by
  have c113937 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113938 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113939 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113940 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113941 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113942 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_4
  simp only [← c113937, ← c113938, ← c113939] at e_1867_4
  have e_1867_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_5
  simp only [← c113940, ← c113941, k2, ← c113942] at e_1867_5
  have hr := e_1867_5
  simp only [e_1867_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 31) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9983)) := by
  have c113943 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113944 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113945 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113946 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113947 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113948 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_6
  simp only [← c113943, ← c113944, ← c113945] at e_1867_6
  have e_1867_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_7
  simp only [← c113946, ← c113947, k2, ← c113948] at e_1867_7
  have hr := e_1867_7
  simp only [e_1867_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 39) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9984)) := by
  have c113949 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113950 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113951 := (publicBatchWrapper86_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113952 := (publicBatchWrapper86_copies3561 a h).1
  have c113953 := (publicBatchWrapper86_copies3561 a h).2.1
  have c113954 := (publicBatchWrapper86_copies3561 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_8
  simp only [← c113949, ← c113950, ← c113951] at e_1867_8
  have e_1867_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_9
  simp only [← c113952, ← c113953, k2, ← c113954] at e_1867_9
  have hr := e_1867_9
  simp only [e_1867_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 47) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9985)) := by
  have c113955 := (publicBatchWrapper86_copies3561 a h).2.2.2.1
  have c113956 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.1
  have c113957 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.1
  have c113958 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.1
  have c113959 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.1
  have c113960 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_10
  simp only [← c113955, ← c113956, ← c113957] at e_1867_10
  have e_1867_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_11
  simp only [← c113958, ← c113959, k2, ← c113960] at e_1867_11
  have hr := e_1867_11
  simp only [e_1867_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 55) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9986)) := by
  have c113961 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.1
  have c113962 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113963 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113964 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113965 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113966 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_12
  simp only [← c113961, ← c113962, ← c113963] at e_1867_12
  have e_1867_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_13
  simp only [← c113964, ← c113965, k2, ← c113966] at e_1867_13
  have hr := e_1867_13
  simp only [e_1867_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 63) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9987)) := by
  have c113967 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113968 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113969 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113970 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113971 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113972 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_14
  simp only [← c113967, ← c113968, ← c113969] at e_1867_14
  have e_1867_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_15
  simp only [← c113970, ← c113971, k2, ← c113972] at e_1867_15
  have hr := e_1867_15
  simp only [e_1867_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 71) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9988)) := by
  have c113973 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113974 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113975 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113976 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113977 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113978 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_16
  simp only [← c113973, ← c113974, ← c113975] at e_1867_16
  have e_1867_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_17
  simp only [← c113976, ← c113977, k2, ← c113978] at e_1867_17
  have hr := e_1867_17
  simp only [e_1867_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1867 79) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9989)) := by
  have c113979 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113980 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113981 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113982 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113983 := (publicBatchWrapper86_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113984 := (publicBatchWrapper86_copies3562 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1867_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_18
  simp only [← c113979, ← c113980, ← c113981] at e_1867_18
  have e_1867_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_19
  simp only [← c113982, ← c113983, k2, ← c113984] at e_1867_19
  have hr := e_1867_19
  simp only [e_1867_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 7) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9990)) := by
  have c113985 := (publicBatchWrapper86_copies3562 a h).2.1
  have c113986 := (publicBatchWrapper86_copies3562 a h).2.2.1
  have c113987 := (publicBatchWrapper86_copies3562 a h).2.2.2.1
  have c113988 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.1
  have c113989 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.1
  have c113990 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_0
  simp only [← c113985, ← c113986, ← c113987] at e_1868_0
  have e_1868_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_1
  simp only [← c113988, ← c113989, k2, ← c113990] at e_1868_1
  have hr := e_1868_1
  simp only [e_1868_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 15) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9991)) := by
  have c113991 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.1
  have c113992 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.1
  have c113993 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.1
  have c113994 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113995 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113996 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_2
  simp only [← c113991, ← c113992, ← c113993] at e_1868_2
  have e_1868_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_3
  simp only [← c113994, ← c113995, k2, ← c113996] at e_1868_3
  have hr := e_1868_3
  simp only [e_1868_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 23) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9992)) := by
  have c113997 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113998 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113999 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114000 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114001 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114002 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_4
  simp only [← c113997, ← c113998, ← c113999] at e_1868_4
  have e_1868_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_5
  simp only [← c114000, ← c114001, k2, ← c114002] at e_1868_5
  have hr := e_1868_5
  simp only [e_1868_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 31) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9993)) := by
  have c114003 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114004 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114005 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114006 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114007 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114008 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_6
  simp only [← c114003, ← c114004, ← c114005] at e_1868_6
  have e_1868_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_7
  simp only [← c114006, ← c114007, k2, ← c114008] at e_1868_7
  have hr := e_1868_7
  simp only [e_1868_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 39) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9994)) := by
  have c114009 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114010 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114011 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114012 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114013 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114014 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_8
  simp only [← c114009, ← c114010, ← c114011] at e_1868_8
  have e_1868_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_9
  simp only [← c114012, ← c114013, k2, ← c114014] at e_1868_9
  have hr := e_1868_9
  simp only [e_1868_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 47) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9995)) := by
  have c114015 := (publicBatchWrapper86_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114016 := (publicBatchWrapper86_copies3563 a h).1
  have c114017 := (publicBatchWrapper86_copies3563 a h).2.1
  have c114018 := (publicBatchWrapper86_copies3563 a h).2.2.1
  have c114019 := (publicBatchWrapper86_copies3563 a h).2.2.2.1
  have c114020 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_10
  simp only [← c114015, ← c114016, ← c114017] at e_1868_10
  have e_1868_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_11
  simp only [← c114018, ← c114019, k2, ← c114020] at e_1868_11
  have hr := e_1868_11
  simp only [e_1868_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 55) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9996)) := by
  have c114021 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.1
  have c114022 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.1
  have c114023 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.1
  have c114024 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.1
  have c114025 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.1
  have c114026 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_12
  simp only [← c114021, ← c114022, ← c114023] at e_1868_12
  have e_1868_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_13
  simp only [← c114024, ← c114025, k2, ← c114026] at e_1868_13
  have hr := e_1868_13
  simp only [e_1868_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 63) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9997)) := by
  have c114027 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114028 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114029 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114030 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114031 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114032 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_14
  simp only [← c114027, ← c114028, ← c114029] at e_1868_14
  have e_1868_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_15
  simp only [← c114030, ← c114031, k2, ← c114032] at e_1868_15
  have hr := e_1868_15
  simp only [e_1868_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 71) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9998)) := by
  have c114033 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114034 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114035 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114036 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114037 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114038 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_16
  simp only [← c114033, ← c114034, ← c114035] at e_1868_16
  have e_1868_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_17
  simp only [← c114036, ← c114037, k2, ← c114038] at e_1868_17
  have hr := e_1868_17
  simp only [e_1868_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1868 79) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 9999)) := by
  have c114039 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114040 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114041 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114042 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114043 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114044 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1868_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_18
  simp only [← c114039, ← c114040, ← c114041] at e_1868_18
  have e_1868_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_19
  simp only [← c114042, ← c114043, k2, ← c114044] at e_1868_19
  have hr := e_1868_19
  simp only [e_1868_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 7) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 10000)) := by
  have c114045 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114046 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114047 := (publicBatchWrapper86_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114048 := (publicBatchWrapper86_copies3564 a h).1
  have c114049 := (publicBatchWrapper86_copies3564 a h).2.1
  have c114050 := (publicBatchWrapper86_copies3564 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_0
  simp only [← c114045, ← c114046, ← c114047] at e_1869_0
  have e_1869_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_1
  simp only [← c114048, ← c114049, k2, ← c114050] at e_1869_1
  have hr := e_1869_1
  simp only [e_1869_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 15) = bselect (a (.wire 49 23)) (a (.virt 200)) (a (.virt 10001)) := by
  have c114051 := (publicBatchWrapper86_copies3564 a h).2.2.2.1
  have c114052 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.1
  have c114053 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.1
  have c114054 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.1
  have c114055 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.1
  have c114056 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_2
  simp only [← c114051, ← c114052, ← c114053] at e_1869_2
  have e_1869_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_3
  simp only [← c114054, ← c114055, k2, ← c114056] at e_1869_3
  have hr := e_1869_3
  simp only [e_1869_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 23) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10170)) := by
  have c114057 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.1
  have c114058 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114059 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114060 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114061 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114062 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_4
  simp only [← c114057, ← c114058, ← c114059] at e_1869_4
  have e_1869_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_5
  simp only [← c114060, ← c114061, k2, ← c114062] at e_1869_5
  have hr := e_1869_5
  simp only [e_1869_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 31) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10171)) := by
  have c114063 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114064 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114065 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114066 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114067 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114068 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_6
  simp only [← c114063, ← c114064, ← c114065] at e_1869_6
  have e_1869_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_7
  simp only [← c114066, ← c114067, k2, ← c114068] at e_1869_7
  have hr := e_1869_7
  simp only [e_1869_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 39) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10172)) := by
  have c114069 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114070 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114071 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114072 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114073 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114074 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_8
  simp only [← c114069, ← c114070, ← c114071] at e_1869_8
  have e_1869_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_9
  simp only [← c114072, ← c114073, k2, ← c114074] at e_1869_9
  have hr := e_1869_9
  simp only [e_1869_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 47) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10173)) := by
  have c114075 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114076 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114077 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114078 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114079 := (publicBatchWrapper86_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114080 := (publicBatchWrapper86_copies3565 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_10
  simp only [← c114075, ← c114076, ← c114077] at e_1869_10
  have e_1869_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_11
  simp only [← c114078, ← c114079, k2, ← c114080] at e_1869_11
  have hr := e_1869_11
  simp only [e_1869_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 55) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10174)) := by
  have c114081 := (publicBatchWrapper86_copies3565 a h).2.1
  have c114082 := (publicBatchWrapper86_copies3565 a h).2.2.1
  have c114083 := (publicBatchWrapper86_copies3565 a h).2.2.2.1
  have c114084 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.1
  have c114085 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.1
  have c114086 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_12
  simp only [← c114081, ← c114082, ← c114083] at e_1869_12
  have e_1869_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_13
  simp only [← c114084, ← c114085, k2, ← c114086] at e_1869_13
  have hr := e_1869_13
  simp only [e_1869_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 63) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10175)) := by
  have c114087 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.1
  have c114088 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.1
  have c114089 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.1
  have c114090 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114091 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114092 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_14
  simp only [← c114087, ← c114088, ← c114089] at e_1869_14
  have e_1869_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_15
  simp only [← c114090, ← c114091, k2, ← c114092] at e_1869_15
  have hr := e_1869_15
  simp only [e_1869_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 71) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10176)) := by
  have c114093 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114094 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114095 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114096 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114097 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114098 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_16
  simp only [← c114093, ← c114094, ← c114095] at e_1869_16
  have e_1869_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_17
  simp only [← c114096, ← c114097, k2, ← c114098] at e_1869_17
  have hr := e_1869_17
  simp only [e_1869_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1869 79) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10177)) := by
  have c114099 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114100 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114101 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114102 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114103 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114104 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1869_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_18
  simp only [← c114099, ← c114100, ← c114101] at e_1869_18
  have e_1869_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_19
  simp only [← c114102, ← c114103, k2, ← c114104] at e_1869_19
  have hr := e_1869_19
  simp only [e_1869_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 7) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10178)) := by
  have c114105 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114106 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114107 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114108 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114109 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114110 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_0
  simp only [← c114105, ← c114106, ← c114107] at e_1870_0
  have e_1870_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_1
  simp only [← c114108, ← c114109, k2, ← c114110] at e_1870_1
  have hr := e_1870_1
  simp only [e_1870_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 15) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10179)) := by
  have c114111 := (publicBatchWrapper86_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114112 := (publicBatchWrapper86_copies3566 a h).1
  have c114113 := (publicBatchWrapper86_copies3566 a h).2.1
  have c114114 := (publicBatchWrapper86_copies3566 a h).2.2.1
  have c114115 := (publicBatchWrapper86_copies3566 a h).2.2.2.1
  have c114116 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_2
  simp only [← c114111, ← c114112, ← c114113] at e_1870_2
  have e_1870_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_3
  simp only [← c114114, ← c114115, k2, ← c114116] at e_1870_3
  have hr := e_1870_3
  simp only [e_1870_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 23) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10180)) := by
  have c114117 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.1
  have c114118 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.1
  have c114119 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.1
  have c114120 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.1
  have c114121 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.1
  have c114122 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_4
  simp only [← c114117, ← c114118, ← c114119] at e_1870_4
  have e_1870_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_5
  simp only [← c114120, ← c114121, k2, ← c114122] at e_1870_5
  have hr := e_1870_5
  simp only [e_1870_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 31) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10181)) := by
  have c114123 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114124 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114125 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114126 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114127 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114128 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_6
  simp only [← c114123, ← c114124, ← c114125] at e_1870_6
  have e_1870_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_7
  simp only [← c114126, ← c114127, k2, ← c114128] at e_1870_7
  have hr := e_1870_7
  simp only [e_1870_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 39) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10182)) := by
  have c114129 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114130 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114131 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114132 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114133 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114134 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_8
  simp only [← c114129, ← c114130, ← c114131] at e_1870_8
  have e_1870_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_9
  simp only [← c114132, ← c114133, k2, ← c114134] at e_1870_9
  have hr := e_1870_9
  simp only [e_1870_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 47) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10183)) := by
  have c114135 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114136 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114137 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114138 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114139 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114140 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_10
  simp only [← c114135, ← c114136, ← c114137] at e_1870_10
  have e_1870_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_11
  simp only [← c114138, ← c114139, k2, ← c114140] at e_1870_11
  have hr := e_1870_11
  simp only [e_1870_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 55) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10184)) := by
  have c114141 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114142 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114143 := (publicBatchWrapper86_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114144 := (publicBatchWrapper86_copies3567 a h).1
  have c114145 := (publicBatchWrapper86_copies3567 a h).2.1
  have c114146 := (publicBatchWrapper86_copies3567 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_12
  simp only [← c114141, ← c114142, ← c114143] at e_1870_12
  have e_1870_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_13
  simp only [← c114144, ← c114145, k2, ← c114146] at e_1870_13
  have hr := e_1870_13
  simp only [e_1870_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 63) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10185)) := by
  have c114147 := (publicBatchWrapper86_copies3567 a h).2.2.2.1
  have c114148 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.1
  have c114149 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.1
  have c114150 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.1
  have c114151 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.1
  have c114152 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_14
  simp only [← c114147, ← c114148, ← c114149] at e_1870_14
  have e_1870_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_15
  simp only [← c114150, ← c114151, k2, ← c114152] at e_1870_15
  have hr := e_1870_15
  simp only [e_1870_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 71) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10186)) := by
  have c114153 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.1
  have c114154 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114155 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114156 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114157 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114158 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_16
  simp only [← c114153, ← c114154, ← c114155] at e_1870_16
  have e_1870_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_17
  simp only [← c114156, ← c114157, k2, ← c114158] at e_1870_17
  have hr := e_1870_17
  simp only [e_1870_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1870 79) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10187)) := by
  have c114159 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114160 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114161 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114162 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114163 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114164 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1870_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_18
  simp only [← c114159, ← c114160, ← c114161] at e_1870_18
  have e_1870_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_19
  simp only [← c114162, ← c114163, k2, ← c114164] at e_1870_19
  have hr := e_1870_19
  simp only [e_1870_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 7) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10188)) := by
  have c114165 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114166 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114167 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114168 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114169 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114170 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_0
  simp only [← c114165, ← c114166, ← c114167] at e_1871_0
  have e_1871_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_1
  simp only [← c114168, ← c114169, k2, ← c114170] at e_1871_1
  have hr := e_1871_1
  simp only [e_1871_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 15) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10189)) := by
  have c114171 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114172 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114173 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114174 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114175 := (publicBatchWrapper86_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114176 := (publicBatchWrapper86_copies3568 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_2
  simp only [← c114171, ← c114172, ← c114173] at e_1871_2
  have e_1871_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_3
  simp only [← c114174, ← c114175, k2, ← c114176] at e_1871_3
  have hr := e_1871_3
  simp only [e_1871_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 23) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10190)) := by
  have c114177 := (publicBatchWrapper86_copies3568 a h).2.1
  have c114178 := (publicBatchWrapper86_copies3568 a h).2.2.1
  have c114179 := (publicBatchWrapper86_copies3568 a h).2.2.2.1
  have c114180 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.1
  have c114181 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.1
  have c114182 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_4
  simp only [← c114177, ← c114178, ← c114179] at e_1871_4
  have e_1871_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_5
  simp only [← c114180, ← c114181, k2, ← c114182] at e_1871_5
  have hr := e_1871_5
  simp only [e_1871_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 31) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10191)) := by
  have c114183 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.1
  have c114184 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.1
  have c114185 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.1
  have c114186 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114187 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114188 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_6
  simp only [← c114183, ← c114184, ← c114185] at e_1871_6
  have e_1871_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_7
  simp only [← c114186, ← c114187, k2, ← c114188] at e_1871_7
  have hr := e_1871_7
  simp only [e_1871_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 39) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10192)) := by
  have c114189 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114190 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114191 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114192 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114193 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114194 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_8
  simp only [← c114189, ← c114190, ← c114191] at e_1871_8
  have e_1871_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_9
  simp only [← c114192, ← c114193, k2, ← c114194] at e_1871_9
  have hr := e_1871_9
  simp only [e_1871_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 47) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10193)) := by
  have c114195 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114196 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114197 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114198 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114199 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114200 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_10
  simp only [← c114195, ← c114196, ← c114197] at e_1871_10
  have e_1871_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_11
  simp only [← c114198, ← c114199, k2, ← c114200] at e_1871_11
  have hr := e_1871_11
  simp only [e_1871_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 55) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10194)) := by
  have c114201 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114202 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114203 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114204 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114205 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114206 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_12
  simp only [← c114201, ← c114202, ← c114203] at e_1871_12
  have e_1871_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_13
  simp only [← c114204, ← c114205, k2, ← c114206] at e_1871_13
  have hr := e_1871_13
  simp only [e_1871_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 63) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10195)) := by
  have c114207 := (publicBatchWrapper86_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114208 := (publicBatchWrapper86_copies3569 a h).1
  have c114209 := (publicBatchWrapper86_copies3569 a h).2.1
  have c114210 := (publicBatchWrapper86_copies3569 a h).2.2.1
  have c114211 := (publicBatchWrapper86_copies3569 a h).2.2.2.1
  have c114212 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_14
  simp only [← c114207, ← c114208, ← c114209] at e_1871_14
  have e_1871_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_15
  simp only [← c114210, ← c114211, k2, ← c114212] at e_1871_15
  have hr := e_1871_15
  simp only [e_1871_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 71) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10196)) := by
  have c114213 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.1
  have c114214 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.1
  have c114215 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.1
  have c114216 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.1
  have c114217 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.1
  have c114218 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_16
  simp only [← c114213, ← c114214, ← c114215] at e_1871_16
  have e_1871_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_17
  simp only [← c114216, ← c114217, k2, ← c114218] at e_1871_17
  have hr := e_1871_17
  simp only [e_1871_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1871 79) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10197)) := by
  have c114219 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114220 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114221 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114222 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114223 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114224 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1871_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_18
  simp only [← c114219, ← c114220, ← c114221] at e_1871_18
  have e_1871_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_19
  simp only [← c114222, ← c114223, k2, ← c114224] at e_1871_19
  have hr := e_1871_19
  simp only [e_1871_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 7) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10198)) := by
  have c114225 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114226 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114227 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114228 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114229 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114230 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_0
  simp only [← c114225, ← c114226, ← c114227] at e_1872_0
  have e_1872_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_1
  simp only [← c114228, ← c114229, k2, ← c114230] at e_1872_1
  have hr := e_1872_1
  simp only [e_1872_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 15) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10199)) := by
  have c114231 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114232 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114233 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114234 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114235 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114236 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_2
  simp only [← c114231, ← c114232, ← c114233] at e_1872_2
  have e_1872_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_3
  simp only [← c114234, ← c114235, k2, ← c114236] at e_1872_3
  have hr := e_1872_3
  simp only [e_1872_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 23) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10200)) := by
  have c114237 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114238 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114239 := (publicBatchWrapper86_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114240 := (publicBatchWrapper86_copies3570 a h).1
  have c114241 := (publicBatchWrapper86_copies3570 a h).2.1
  have c114242 := (publicBatchWrapper86_copies3570 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_4
  simp only [← c114237, ← c114238, ← c114239] at e_1872_4
  have e_1872_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_5
  simp only [← c114240, ← c114241, k2, ← c114242] at e_1872_5
  have hr := e_1872_5
  simp only [e_1872_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 31) = bselect (a (.wire 49 67)) (a (.virt 200)) (a (.virt 10201)) := by
  have c114243 := (publicBatchWrapper86_copies3570 a h).2.2.2.1
  have c114244 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.1
  have c114245 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.1
  have c114246 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.1
  have c114247 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.1
  have c114248 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_6
  simp only [← c114243, ← c114244, ← c114245] at e_1872_6
  have e_1872_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_7
  simp only [← c114246, ← c114247, k2, ← c114248] at e_1872_7
  have hr := e_1872_7
  simp only [e_1872_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 39) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10370)) := by
  have c114249 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.1
  have c114250 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114251 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114252 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114253 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114254 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_8
  simp only [← c114249, ← c114250, ← c114251] at e_1872_8
  have e_1872_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_9
  simp only [← c114252, ← c114253, k2, ← c114254] at e_1872_9
  have hr := e_1872_9
  simp only [e_1872_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 47) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10371)) := by
  have c114255 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114256 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114257 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114258 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114259 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114260 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_10
  simp only [← c114255, ← c114256, ← c114257] at e_1872_10
  have e_1872_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_11
  simp only [← c114258, ← c114259, k2, ← c114260] at e_1872_11
  have hr := e_1872_11
  simp only [e_1872_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 55) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10372)) := by
  have c114261 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114262 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114263 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114264 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114265 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114266 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_12
  simp only [← c114261, ← c114262, ← c114263] at e_1872_12
  have e_1872_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_13
  simp only [← c114264, ← c114265, k2, ← c114266] at e_1872_13
  have hr := e_1872_13
  simp only [e_1872_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 63) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10373)) := by
  have c114267 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114268 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114269 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114270 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114271 := (publicBatchWrapper86_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114272 := (publicBatchWrapper86_copies3571 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_14
  simp only [← c114267, ← c114268, ← c114269] at e_1872_14
  have e_1872_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_15
  simp only [← c114270, ← c114271, k2, ← c114272] at e_1872_15
  have hr := e_1872_15
  simp only [e_1872_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 71) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10374)) := by
  have c114273 := (publicBatchWrapper86_copies3571 a h).2.1
  have c114274 := (publicBatchWrapper86_copies3571 a h).2.2.1
  have c114275 := (publicBatchWrapper86_copies3571 a h).2.2.2.1
  have c114276 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.1
  have c114277 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.1
  have c114278 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_16
  simp only [← c114273, ← c114274, ← c114275] at e_1872_16
  have e_1872_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_17
  simp only [← c114276, ← c114277, k2, ← c114278] at e_1872_17
  have hr := e_1872_17
  simp only [e_1872_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1872 79) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10375)) := by
  have c114279 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.1
  have c114280 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.1
  have c114281 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.1
  have c114282 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114283 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114284 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1872_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_18
  simp only [← c114279, ← c114280, ← c114281] at e_1872_18
  have e_1872_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_19
  simp only [← c114282, ← c114283, k2, ← c114284] at e_1872_19
  have hr := e_1872_19
  simp only [e_1872_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 7) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10376)) := by
  have c114285 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114286 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114287 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114288 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114289 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114290 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_0
  simp only [← c114285, ← c114286, ← c114287] at e_1873_0
  have e_1873_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_1
  simp only [← c114288, ← c114289, k2, ← c114290] at e_1873_1
  have hr := e_1873_1
  simp only [e_1873_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 15) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10377)) := by
  have c114291 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114292 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114293 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114294 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114295 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114296 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_2
  simp only [← c114291, ← c114292, ← c114293] at e_1873_2
  have e_1873_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_3
  simp only [← c114294, ← c114295, k2, ← c114296] at e_1873_3
  have hr := e_1873_3
  simp only [e_1873_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 23) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10378)) := by
  have c114297 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114298 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114299 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114300 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114301 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114302 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_4
  simp only [← c114297, ← c114298, ← c114299] at e_1873_4
  have e_1873_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_5
  simp only [← c114300, ← c114301, k2, ← c114302] at e_1873_5
  have hr := e_1873_5
  simp only [e_1873_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 31) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10379)) := by
  have c114303 := (publicBatchWrapper86_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114304 := (publicBatchWrapper86_copies3572 a h).1
  have c114305 := (publicBatchWrapper86_copies3572 a h).2.1
  have c114306 := (publicBatchWrapper86_copies3572 a h).2.2.1
  have c114307 := (publicBatchWrapper86_copies3572 a h).2.2.2.1
  have c114308 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_6
  simp only [← c114303, ← c114304, ← c114305] at e_1873_6
  have e_1873_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_7
  simp only [← c114306, ← c114307, k2, ← c114308] at e_1873_7
  have hr := e_1873_7
  simp only [e_1873_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 39) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10380)) := by
  have c114309 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.1
  have c114310 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.1
  have c114311 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.1
  have c114312 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.1
  have c114313 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.1
  have c114314 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_8
  simp only [← c114309, ← c114310, ← c114311] at e_1873_8
  have e_1873_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_9
  simp only [← c114312, ← c114313, k2, ← c114314] at e_1873_9
  have hr := e_1873_9
  simp only [e_1873_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 47) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10381)) := by
  have c114315 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114316 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114317 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114318 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114319 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114320 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_10
  simp only [← c114315, ← c114316, ← c114317] at e_1873_10
  have e_1873_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_11
  simp only [← c114318, ← c114319, k2, ← c114320] at e_1873_11
  have hr := e_1873_11
  simp only [e_1873_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 55) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10382)) := by
  have c114321 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114322 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114323 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114324 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114325 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114326 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_12
  simp only [← c114321, ← c114322, ← c114323] at e_1873_12
  have e_1873_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_13
  simp only [← c114324, ← c114325, k2, ← c114326] at e_1873_13
  have hr := e_1873_13
  simp only [e_1873_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 63) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10383)) := by
  have c114327 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114328 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114329 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114330 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114331 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114332 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_14
  simp only [← c114327, ← c114328, ← c114329] at e_1873_14
  have e_1873_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_15
  simp only [← c114330, ← c114331, k2, ← c114332] at e_1873_15
  have hr := e_1873_15
  simp only [e_1873_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 71) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10384)) := by
  have c114333 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114334 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114335 := (publicBatchWrapper86_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114336 := (publicBatchWrapper86_copies3573 a h).1
  have c114337 := (publicBatchWrapper86_copies3573 a h).2.1
  have c114338 := (publicBatchWrapper86_copies3573 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_16
  simp only [← c114333, ← c114334, ← c114335] at e_1873_16
  have e_1873_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_17
  simp only [← c114336, ← c114337, k2, ← c114338] at e_1873_17
  have hr := e_1873_17
  simp only [e_1873_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1873 79) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10385)) := by
  have c114339 := (publicBatchWrapper86_copies3573 a h).2.2.2.1
  have c114340 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.1
  have c114341 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.1
  have c114342 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.1
  have c114343 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.1
  have c114344 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1873_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_18
  simp only [← c114339, ← c114340, ← c114341] at e_1873_18
  have e_1873_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_19
  simp only [← c114342, ← c114343, k2, ← c114344] at e_1873_19
  have hr := e_1873_19
  simp only [e_1873_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 7) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10386)) := by
  have c114345 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.1
  have c114346 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114347 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114348 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114349 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114350 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_0
  simp only [← c114345, ← c114346, ← c114347] at e_1874_0
  have e_1874_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_1
  simp only [← c114348, ← c114349, k2, ← c114350] at e_1874_1
  have hr := e_1874_1
  simp only [e_1874_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 15) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10387)) := by
  have c114351 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114352 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114353 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114354 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114355 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114356 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_2
  simp only [← c114351, ← c114352, ← c114353] at e_1874_2
  have e_1874_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_3
  simp only [← c114354, ← c114355, k2, ← c114356] at e_1874_3
  have hr := e_1874_3
  simp only [e_1874_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 23) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10388)) := by
  have c114357 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114358 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114359 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114360 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114361 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114362 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_4
  simp only [← c114357, ← c114358, ← c114359] at e_1874_4
  have e_1874_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_5
  simp only [← c114360, ← c114361, k2, ← c114362] at e_1874_5
  have hr := e_1874_5
  simp only [e_1874_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 31) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10389)) := by
  have c114363 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114364 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114365 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114366 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114367 := (publicBatchWrapper86_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114368 := (publicBatchWrapper86_copies3574 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_6
  simp only [← c114363, ← c114364, ← c114365] at e_1874_6
  have e_1874_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_7
  simp only [← c114366, ← c114367, k2, ← c114368] at e_1874_7
  have hr := e_1874_7
  simp only [e_1874_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 39) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10390)) := by
  have c114369 := (publicBatchWrapper86_copies3574 a h).2.1
  have c114370 := (publicBatchWrapper86_copies3574 a h).2.2.1
  have c114371 := (publicBatchWrapper86_copies3574 a h).2.2.2.1
  have c114372 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.1
  have c114373 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.1
  have c114374 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_8
  simp only [← c114369, ← c114370, ← c114371] at e_1874_8
  have e_1874_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_9
  simp only [← c114372, ← c114373, k2, ← c114374] at e_1874_9
  have hr := e_1874_9
  simp only [e_1874_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 47) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10391)) := by
  have c114375 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.1
  have c114376 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.1
  have c114377 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.1
  have c114378 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114379 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114380 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_10
  simp only [← c114375, ← c114376, ← c114377] at e_1874_10
  have e_1874_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_11
  simp only [← c114378, ← c114379, k2, ← c114380] at e_1874_11
  have hr := e_1874_11
  simp only [e_1874_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 55) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10392)) := by
  have c114381 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114382 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114383 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114384 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114385 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114386 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_12
  simp only [← c114381, ← c114382, ← c114383] at e_1874_12
  have e_1874_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_13
  simp only [← c114384, ← c114385, k2, ← c114386] at e_1874_13
  have hr := e_1874_13
  simp only [e_1874_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 63) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10393)) := by
  have c114387 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114388 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114389 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114390 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114391 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114392 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_14
  simp only [← c114387, ← c114388, ← c114389] at e_1874_14
  have e_1874_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_15
  simp only [← c114390, ← c114391, k2, ← c114392] at e_1874_15
  have hr := e_1874_15
  simp only [e_1874_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 71) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10394)) := by
  have c114393 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114394 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114395 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114396 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114397 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114398 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_16
  simp only [← c114393, ← c114394, ← c114395] at e_1874_16
  have e_1874_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_17
  simp only [← c114396, ← c114397, k2, ← c114398] at e_1874_17
  have hr := e_1874_17
  simp only [e_1874_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1874 79) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10395)) := by
  have c114399 := (publicBatchWrapper86_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114400 := (publicBatchWrapper86_copies3575 a h).1
  have c114401 := (publicBatchWrapper86_copies3575 a h).2.1
  have c114402 := (publicBatchWrapper86_copies3575 a h).2.2.1
  have c114403 := (publicBatchWrapper86_copies3575 a h).2.2.2.1
  have c114404 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1874_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_18
  simp only [← c114399, ← c114400, ← c114401] at e_1874_18
  have e_1874_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_19
  simp only [← c114402, ← c114403, k2, ← c114404] at e_1874_19
  have hr := e_1874_19
  simp only [e_1874_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 7) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10396)) := by
  have c114405 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.1
  have c114406 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.1
  have c114407 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.1
  have c114408 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.1
  have c114409 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.1
  have c114410 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_0
  simp only [← c114405, ← c114406, ← c114407] at e_1875_0
  have e_1875_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_1
  simp only [← c114408, ← c114409, k2, ← c114410] at e_1875_1
  have hr := e_1875_1
  simp only [e_1875_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 15) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10397)) := by
  have c114411 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114412 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114413 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114414 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114415 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114416 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_2
  simp only [← c114411, ← c114412, ← c114413] at e_1875_2
  have e_1875_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_3
  simp only [← c114414, ← c114415, k2, ← c114416] at e_1875_3
  have hr := e_1875_3
  simp only [e_1875_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 23) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10398)) := by
  have c114417 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114418 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114419 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114420 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114421 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114422 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_4
  simp only [← c114417, ← c114418, ← c114419] at e_1875_4
  have e_1875_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_5
  simp only [← c114420, ← c114421, k2, ← c114422] at e_1875_5
  have hr := e_1875_5
  simp only [e_1875_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 31) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10399)) := by
  have c114423 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114424 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114425 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114426 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114427 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114428 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_6
  simp only [← c114423, ← c114424, ← c114425] at e_1875_6
  have e_1875_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_7
  simp only [← c114426, ← c114427, k2, ← c114428] at e_1875_7
  have hr := e_1875_7
  simp only [e_1875_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 39) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10400)) := by
  have c114429 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114430 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114431 := (publicBatchWrapper86_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114432 := (publicBatchWrapper86_copies3576 a h).1
  have c114433 := (publicBatchWrapper86_copies3576 a h).2.1
  have c114434 := (publicBatchWrapper86_copies3576 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_8
  simp only [← c114429, ← c114430, ← c114431] at e_1875_8
  have e_1875_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_9
  simp only [← c114432, ← c114433, k2, ← c114434] at e_1875_9
  have hr := e_1875_9
  simp only [e_1875_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 47) = bselect (a (.wire 51 31)) (a (.virt 200)) (a (.virt 10401)) := by
  have c114435 := (publicBatchWrapper86_copies3576 a h).2.2.2.1
  have c114436 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.1
  have c114437 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.1
  have c114438 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.1
  have c114439 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.1
  have c114440 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_10
  simp only [← c114435, ← c114436, ← c114437] at e_1875_10
  have e_1875_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_11
  simp only [← c114438, ← c114439, k2, ← c114440] at e_1875_11
  have hr := e_1875_11
  simp only [e_1875_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10570)) := by
  have c114441 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.1
  have c114442 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114443 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114444 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114445 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114446 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_12
  simp only [← c114441, ← c114442, ← c114443] at e_1875_12
  have e_1875_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_13
  simp only [← c114444, ← c114445, k2, ← c114446] at e_1875_13
  have hr := e_1875_13
  simp only [e_1875_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10571)) := by
  have c114447 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114448 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114449 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114450 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114451 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114452 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_14
  simp only [← c114447, ← c114448, ← c114449] at e_1875_14
  have e_1875_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_15
  simp only [← c114450, ← c114451, k2, ← c114452] at e_1875_15
  have hr := e_1875_15
  simp only [e_1875_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10572)) := by
  have c114453 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114454 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114455 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114456 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114457 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114458 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_16
  simp only [← c114453, ← c114454, ← c114455] at e_1875_16
  have e_1875_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_17
  simp only [← c114456, ← c114457, k2, ← c114458] at e_1875_17
  have hr := e_1875_17
  simp only [e_1875_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1875 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10573)) := by
  have c114459 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114460 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114461 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114462 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114463 := (publicBatchWrapper86_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114464 := (publicBatchWrapper86_copies3577 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1875_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_18
  simp only [← c114459, ← c114460, ← c114461] at e_1875_18
  have e_1875_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_19
  simp only [← c114462, ← c114463, k2, ← c114464] at e_1875_19
  have hr := e_1875_19
  simp only [e_1875_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10574)) := by
  have c114465 := (publicBatchWrapper86_copies3577 a h).2.1
  have c114466 := (publicBatchWrapper86_copies3577 a h).2.2.1
  have c114467 := (publicBatchWrapper86_copies3577 a h).2.2.2.1
  have c114468 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.1
  have c114469 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.1
  have c114470 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_0
  simp only [← c114465, ← c114466, ← c114467] at e_1876_0
  have e_1876_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_1
  simp only [← c114468, ← c114469, k2, ← c114470] at e_1876_1
  have hr := e_1876_1
  simp only [e_1876_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10575)) := by
  have c114471 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.1
  have c114472 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.1
  have c114473 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.1
  have c114474 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114475 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114476 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_2
  simp only [← c114471, ← c114472, ← c114473] at e_1876_2
  have e_1876_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_3
  simp only [← c114474, ← c114475, k2, ← c114476] at e_1876_3
  have hr := e_1876_3
  simp only [e_1876_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10576)) := by
  have c114477 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114478 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114479 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114480 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114481 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114482 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_4
  simp only [← c114477, ← c114478, ← c114479] at e_1876_4
  have e_1876_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_5
  simp only [← c114480, ← c114481, k2, ← c114482] at e_1876_5
  have hr := e_1876_5
  simp only [e_1876_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10577)) := by
  have c114483 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114484 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114485 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114486 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114487 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114488 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_6
  simp only [← c114483, ← c114484, ← c114485] at e_1876_6
  have e_1876_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_7
  simp only [← c114486, ← c114487, k2, ← c114488] at e_1876_7
  have hr := e_1876_7
  simp only [e_1876_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10578)) := by
  have c114489 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114490 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114491 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114492 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114493 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114494 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_8
  simp only [← c114489, ← c114490, ← c114491] at e_1876_8
  have e_1876_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_9
  simp only [← c114492, ← c114493, k2, ← c114494] at e_1876_9
  have hr := e_1876_9
  simp only [e_1876_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10579)) := by
  have c114495 := (publicBatchWrapper86_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114496 := (publicBatchWrapper86_copies3578 a h).1
  have c114497 := (publicBatchWrapper86_copies3578 a h).2.1
  have c114498 := (publicBatchWrapper86_copies3578 a h).2.2.1
  have c114499 := (publicBatchWrapper86_copies3578 a h).2.2.2.1
  have c114500 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_10
  simp only [← c114495, ← c114496, ← c114497] at e_1876_10
  have e_1876_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_11
  simp only [← c114498, ← c114499, k2, ← c114500] at e_1876_11
  have hr := e_1876_11
  simp only [e_1876_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10580)) := by
  have c114501 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.1
  have c114502 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.1
  have c114503 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.1
  have c114504 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.1
  have c114505 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.1
  have c114506 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_12
  simp only [← c114501, ← c114502, ← c114503] at e_1876_12
  have e_1876_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_13
  simp only [← c114504, ← c114505, k2, ← c114506] at e_1876_13
  have hr := e_1876_13
  simp only [e_1876_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10581)) := by
  have c114507 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114508 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114509 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114510 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114511 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114512 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_14
  simp only [← c114507, ← c114508, ← c114509] at e_1876_14
  have e_1876_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_15
  simp only [← c114510, ← c114511, k2, ← c114512] at e_1876_15
  have hr := e_1876_15
  simp only [e_1876_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10582)) := by
  have c114513 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114514 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114515 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114516 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114517 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114518 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_16
  simp only [← c114513, ← c114514, ← c114515] at e_1876_16
  have e_1876_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_17
  simp only [← c114516, ← c114517, k2, ← c114518] at e_1876_17
  have hr := e_1876_17
  simp only [e_1876_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1876 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10583)) := by
  have c114519 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114520 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114521 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114522 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114523 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114524 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1876_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_18
  simp only [← c114519, ← c114520, ← c114521] at e_1876_18
  have e_1876_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_19
  simp only [← c114522, ← c114523, k2, ← c114524] at e_1876_19
  have hr := e_1876_19
  simp only [e_1876_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10584)) := by
  have c114525 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114526 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114527 := (publicBatchWrapper86_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114528 := (publicBatchWrapper86_copies3579 a h).1
  have c114529 := (publicBatchWrapper86_copies3579 a h).2.1
  have c114530 := (publicBatchWrapper86_copies3579 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_0
  simp only [← c114525, ← c114526, ← c114527] at e_1877_0
  have e_1877_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_1
  simp only [← c114528, ← c114529, k2, ← c114530] at e_1877_1
  have hr := e_1877_1
  simp only [e_1877_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10585)) := by
  have c114531 := (publicBatchWrapper86_copies3579 a h).2.2.2.1
  have c114532 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.1
  have c114533 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.1
  have c114534 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.1
  have c114535 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.1
  have c114536 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_2
  simp only [← c114531, ← c114532, ← c114533] at e_1877_2
  have e_1877_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_3
  simp only [← c114534, ← c114535, k2, ← c114536] at e_1877_3
  have hr := e_1877_3
  simp only [e_1877_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10586)) := by
  have c114537 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.1
  have c114538 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114539 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114540 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114541 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114542 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_4
  simp only [← c114537, ← c114538, ← c114539] at e_1877_4
  have e_1877_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_5
  simp only [← c114540, ← c114541, k2, ← c114542] at e_1877_5
  have hr := e_1877_5
  simp only [e_1877_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10587)) := by
  have c114543 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114544 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114545 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114546 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114547 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114548 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_6
  simp only [← c114543, ← c114544, ← c114545] at e_1877_6
  have e_1877_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_7
  simp only [← c114546, ← c114547, k2, ← c114548] at e_1877_7
  have hr := e_1877_7
  simp only [e_1877_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10588)) := by
  have c114549 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114550 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114551 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114552 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114553 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114554 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_8
  simp only [← c114549, ← c114550, ← c114551] at e_1877_8
  have e_1877_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_9
  simp only [← c114552, ← c114553, k2, ← c114554] at e_1877_9
  have hr := e_1877_9
  simp only [e_1877_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10589)) := by
  have c114555 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114556 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114557 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114558 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114559 := (publicBatchWrapper86_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114560 := (publicBatchWrapper86_copies3580 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_10
  simp only [← c114555, ← c114556, ← c114557] at e_1877_10
  have e_1877_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_11
  simp only [← c114558, ← c114559, k2, ← c114560] at e_1877_11
  have hr := e_1877_11
  simp only [e_1877_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10590)) := by
  have c114561 := (publicBatchWrapper86_copies3580 a h).2.1
  have c114562 := (publicBatchWrapper86_copies3580 a h).2.2.1
  have c114563 := (publicBatchWrapper86_copies3580 a h).2.2.2.1
  have c114564 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.1
  have c114565 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.1
  have c114566 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_12
  simp only [← c114561, ← c114562, ← c114563] at e_1877_12
  have e_1877_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_13
  simp only [← c114564, ← c114565, k2, ← c114566] at e_1877_13
  have hr := e_1877_13
  simp only [e_1877_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10591)) := by
  have c114567 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.1
  have c114568 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.1
  have c114569 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.1
  have c114570 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114571 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114572 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_14
  simp only [← c114567, ← c114568, ← c114569] at e_1877_14
  have e_1877_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_15
  simp only [← c114570, ← c114571, k2, ← c114572] at e_1877_15
  have hr := e_1877_15
  simp only [e_1877_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10592)) := by
  have c114573 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114574 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114575 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114576 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114577 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114578 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_16
  simp only [← c114573, ← c114574, ← c114575] at e_1877_16
  have e_1877_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_17
  simp only [← c114576, ← c114577, k2, ← c114578] at e_1877_17
  have hr := e_1877_17
  simp only [e_1877_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1877 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10593)) := by
  have c114579 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114580 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114581 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114582 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114583 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114584 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1877_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_18
  simp only [← c114579, ← c114580, ← c114581] at e_1877_18
  have e_1877_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_19
  simp only [← c114582, ← c114583, k2, ← c114584] at e_1877_19
  have hr := e_1877_19
  simp only [e_1877_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10594)) := by
  have c114585 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114586 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114587 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114588 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114589 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114590 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_0
  simp only [← c114585, ← c114586, ← c114587] at e_1878_0
  have e_1878_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_1
  simp only [← c114588, ← c114589, k2, ← c114590] at e_1878_1
  have hr := e_1878_1
  simp only [e_1878_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10595)) := by
  have c114591 := (publicBatchWrapper86_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114592 := (publicBatchWrapper86_copies3581 a h).1
  have c114593 := (publicBatchWrapper86_copies3581 a h).2.1
  have c114594 := (publicBatchWrapper86_copies3581 a h).2.2.1
  have c114595 := (publicBatchWrapper86_copies3581 a h).2.2.2.1
  have c114596 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_2
  simp only [← c114591, ← c114592, ← c114593] at e_1878_2
  have e_1878_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_3
  simp only [← c114594, ← c114595, k2, ← c114596] at e_1878_3
  have hr := e_1878_3
  simp only [e_1878_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10596)) := by
  have c114597 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.1
  have c114598 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.1
  have c114599 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.1
  have c114600 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.1
  have c114601 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.1
  have c114602 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_4
  simp only [← c114597, ← c114598, ← c114599] at e_1878_4
  have e_1878_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_5
  simp only [← c114600, ← c114601, k2, ← c114602] at e_1878_5
  have hr := e_1878_5
  simp only [e_1878_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10597)) := by
  have c114603 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114604 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114605 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114606 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114607 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114608 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_6
  simp only [← c114603, ← c114604, ← c114605] at e_1878_6
  have e_1878_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_7
  simp only [← c114606, ← c114607, k2, ← c114608] at e_1878_7
  have hr := e_1878_7
  simp only [e_1878_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10598)) := by
  have c114609 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114610 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114611 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114612 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114613 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114614 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_8
  simp only [← c114609, ← c114610, ← c114611] at e_1878_8
  have e_1878_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_9
  simp only [← c114612, ← c114613, k2, ← c114614] at e_1878_9
  have hr := e_1878_9
  simp only [e_1878_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10599)) := by
  have c114615 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114616 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114617 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114618 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114619 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114620 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_10
  simp only [← c114615, ← c114616, ← c114617] at e_1878_10
  have e_1878_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_11
  simp only [← c114618, ← c114619, k2, ← c114620] at e_1878_11
  have hr := e_1878_11
  simp only [e_1878_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10600)) := by
  have c114621 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114622 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114623 := (publicBatchWrapper86_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114624 := (publicBatchWrapper86_copies3582 a h).1
  have c114625 := (publicBatchWrapper86_copies3582 a h).2.1
  have c114626 := (publicBatchWrapper86_copies3582 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_12
  simp only [← c114621, ← c114622, ← c114623] at e_1878_12
  have e_1878_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_13
  simp only [← c114624, ← c114625, k2, ← c114626] at e_1878_13
  have hr := e_1878_13
  simp only [e_1878_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10601)) := by
  have c114627 := (publicBatchWrapper86_copies3582 a h).2.2.2.1
  have c114628 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.1
  have c114629 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.1
  have c114630 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.1
  have c114631 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.1
  have c114632 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_14
  simp only [← c114627, ← c114628, ← c114629] at e_1878_14
  have e_1878_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_15
  simp only [← c114630, ← c114631, k2, ← c114632] at e_1878_15
  have hr := e_1878_15
  simp only [e_1878_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10770)) := by
  have c114633 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.1
  have c114634 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114635 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114636 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114637 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114638 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_16
  simp only [← c114633, ← c114634, ← c114635] at e_1878_16
  have e_1878_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_17
  simp only [← c114636, ← c114637, k2, ← c114638] at e_1878_17
  have hr := e_1878_17
  simp only [e_1878_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1878 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10771)) := by
  have c114639 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114640 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114641 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114642 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114643 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114644 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1878_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_18
  simp only [← c114639, ← c114640, ← c114641] at e_1878_18
  have e_1878_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_19
  simp only [← c114642, ← c114643, k2, ← c114644] at e_1878_19
  have hr := e_1878_19
  simp only [e_1878_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10772)) := by
  have c114645 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114646 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114647 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114648 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114649 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114650 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_0
  simp only [← c114645, ← c114646, ← c114647] at e_1879_0
  have e_1879_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_1
  simp only [← c114648, ← c114649, k2, ← c114650] at e_1879_1
  have hr := e_1879_1
  simp only [e_1879_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10773)) := by
  have c114651 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114652 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114653 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114654 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114655 := (publicBatchWrapper86_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114656 := (publicBatchWrapper86_copies3583 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_2
  simp only [← c114651, ← c114652, ← c114653] at e_1879_2
  have e_1879_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_3
  simp only [← c114654, ← c114655, k2, ← c114656] at e_1879_3
  have hr := e_1879_3
  simp only [e_1879_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10774)) := by
  have c114657 := (publicBatchWrapper86_copies3583 a h).2.1
  have c114658 := (publicBatchWrapper86_copies3583 a h).2.2.1
  have c114659 := (publicBatchWrapper86_copies3583 a h).2.2.2.1
  have c114660 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.1
  have c114661 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.1
  have c114662 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_4
  simp only [← c114657, ← c114658, ← c114659] at e_1879_4
  have e_1879_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_5
  simp only [← c114660, ← c114661, k2, ← c114662] at e_1879_5
  have hr := e_1879_5
  simp only [e_1879_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10775)) := by
  have c114663 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.1
  have c114664 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.1
  have c114665 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.1
  have c114666 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114667 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114668 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_6
  simp only [← c114663, ← c114664, ← c114665] at e_1879_6
  have e_1879_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_7
  simp only [← c114666, ← c114667, k2, ← c114668] at e_1879_7
  have hr := e_1879_7
  simp only [e_1879_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10776)) := by
  have c114669 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114670 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114671 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114672 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114673 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114674 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_8
  simp only [← c114669, ← c114670, ← c114671] at e_1879_8
  have e_1879_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_9
  simp only [← c114672, ← c114673, k2, ← c114674] at e_1879_9
  have hr := e_1879_9
  simp only [e_1879_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10777)) := by
  have c114675 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114676 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114677 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114678 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114679 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114680 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_10
  simp only [← c114675, ← c114676, ← c114677] at e_1879_10
  have e_1879_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_11
  simp only [← c114678, ← c114679, k2, ← c114680] at e_1879_11
  have hr := e_1879_11
  simp only [e_1879_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10778)) := by
  have c114681 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114682 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114683 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114684 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114685 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114686 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_12
  simp only [← c114681, ← c114682, ← c114683] at e_1879_12
  have e_1879_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_13
  simp only [← c114684, ← c114685, k2, ← c114686] at e_1879_13
  have hr := e_1879_13
  simp only [e_1879_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10779)) := by
  have c114687 := (publicBatchWrapper86_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114688 := (publicBatchWrapper86_copies3584 a h).1
  have c114689 := (publicBatchWrapper86_copies3584 a h).2.1
  have c114690 := (publicBatchWrapper86_copies3584 a h).2.2.1
  have c114691 := (publicBatchWrapper86_copies3584 a h).2.2.2.1
  have c114692 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_14
  simp only [← c114687, ← c114688, ← c114689] at e_1879_14
  have e_1879_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_15
  simp only [← c114690, ← c114691, k2, ← c114692] at e_1879_15
  have hr := e_1879_15
  simp only [e_1879_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10780)) := by
  have c114693 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.1
  have c114694 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.1
  have c114695 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.1
  have c114696 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.1
  have c114697 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.1
  have c114698 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_16
  simp only [← c114693, ← c114694, ← c114695] at e_1879_16
  have e_1879_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_17
  simp only [← c114696, ← c114697, k2, ← c114698] at e_1879_17
  have hr := e_1879_17
  simp only [e_1879_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1879 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10781)) := by
  have c114699 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114700 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114701 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114702 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114703 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114704 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1879_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_18
  simp only [← c114699, ← c114700, ← c114701] at e_1879_18
  have e_1879_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_19
  simp only [← c114702, ← c114703, k2, ← c114704] at e_1879_19
  have hr := e_1879_19
  simp only [e_1879_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10782)) := by
  have c114705 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114706 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114707 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114708 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114709 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114710 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_0
  simp only [← c114705, ← c114706, ← c114707] at e_1880_0
  have e_1880_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_1
  simp only [← c114708, ← c114709, k2, ← c114710] at e_1880_1
  have hr := e_1880_1
  simp only [e_1880_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10783)) := by
  have c114711 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114712 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114713 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114714 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114715 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114716 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_2
  simp only [← c114711, ← c114712, ← c114713] at e_1880_2
  have e_1880_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_3
  simp only [← c114714, ← c114715, k2, ← c114716] at e_1880_3
  have hr := e_1880_3
  simp only [e_1880_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10784)) := by
  have c114717 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114718 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114719 := (publicBatchWrapper86_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114720 := (publicBatchWrapper86_copies3585 a h).1
  have c114721 := (publicBatchWrapper86_copies3585 a h).2.1
  have c114722 := (publicBatchWrapper86_copies3585 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_4
  simp only [← c114717, ← c114718, ← c114719] at e_1880_4
  have e_1880_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_5
  simp only [← c114720, ← c114721, k2, ← c114722] at e_1880_5
  have hr := e_1880_5
  simp only [e_1880_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10785)) := by
  have c114723 := (publicBatchWrapper86_copies3585 a h).2.2.2.1
  have c114724 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.1
  have c114725 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.1
  have c114726 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.1
  have c114727 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.1
  have c114728 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_6
  simp only [← c114723, ← c114724, ← c114725] at e_1880_6
  have e_1880_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_7
  simp only [← c114726, ← c114727, k2, ← c114728] at e_1880_7
  have hr := e_1880_7
  simp only [e_1880_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10786)) := by
  have c114729 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.1
  have c114730 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114731 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114732 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114733 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114734 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_8
  simp only [← c114729, ← c114730, ← c114731] at e_1880_8
  have e_1880_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_9
  simp only [← c114732, ← c114733, k2, ← c114734] at e_1880_9
  have hr := e_1880_9
  simp only [e_1880_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10787)) := by
  have c114735 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114736 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114737 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114738 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114739 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114740 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_10
  simp only [← c114735, ← c114736, ← c114737] at e_1880_10
  have e_1880_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_11
  simp only [← c114738, ← c114739, k2, ← c114740] at e_1880_11
  have hr := e_1880_11
  simp only [e_1880_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10788)) := by
  have c114741 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114742 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114743 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114744 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114745 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114746 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_12
  simp only [← c114741, ← c114742, ← c114743] at e_1880_12
  have e_1880_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_13
  simp only [← c114744, ← c114745, k2, ← c114746] at e_1880_13
  have hr := e_1880_13
  simp only [e_1880_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10789)) := by
  have c114747 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114748 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114749 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114750 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114751 := (publicBatchWrapper86_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114752 := (publicBatchWrapper86_copies3586 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_14
  simp only [← c114747, ← c114748, ← c114749] at e_1880_14
  have e_1880_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_15
  simp only [← c114750, ← c114751, k2, ← c114752] at e_1880_15
  have hr := e_1880_15
  simp only [e_1880_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10790)) := by
  have c114753 := (publicBatchWrapper86_copies3586 a h).2.1
  have c114754 := (publicBatchWrapper86_copies3586 a h).2.2.1
  have c114755 := (publicBatchWrapper86_copies3586 a h).2.2.2.1
  have c114756 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.1
  have c114757 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.1
  have c114758 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_16
  simp only [← c114753, ← c114754, ← c114755] at e_1880_16
  have e_1880_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_17
  simp only [← c114756, ← c114757, k2, ← c114758] at e_1880_17
  have hr := e_1880_17
  simp only [e_1880_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1880 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10791)) := by
  have c114759 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.1
  have c114760 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.1
  have c114761 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.1
  have c114762 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114763 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114764 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1880_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_18
  simp only [← c114759, ← c114760, ← c114761] at e_1880_18
  have e_1880_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_19
  simp only [← c114762, ← c114763, k2, ← c114764] at e_1880_19
  have hr := e_1880_19
  simp only [e_1880_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10792)) := by
  have c114765 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114766 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114767 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114768 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114769 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114770 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_0
  simp only [← c114765, ← c114766, ← c114767] at e_1881_0
  have e_1881_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_1
  simp only [← c114768, ← c114769, k2, ← c114770] at e_1881_1
  have hr := e_1881_1
  simp only [e_1881_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10793)) := by
  have c114771 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114772 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114773 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114774 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114775 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114776 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_2
  simp only [← c114771, ← c114772, ← c114773] at e_1881_2
  have e_1881_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_3
  simp only [← c114774, ← c114775, k2, ← c114776] at e_1881_3
  have hr := e_1881_3
  simp only [e_1881_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10794)) := by
  have c114777 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114778 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114779 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114780 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114781 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114782 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_4
  simp only [← c114777, ← c114778, ← c114779] at e_1881_4
  have e_1881_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_5
  simp only [← c114780, ← c114781, k2, ← c114782] at e_1881_5
  have hr := e_1881_5
  simp only [e_1881_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10795)) := by
  have c114783 := (publicBatchWrapper86_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114784 := (publicBatchWrapper86_copies3587 a h).1
  have c114785 := (publicBatchWrapper86_copies3587 a h).2.1
  have c114786 := (publicBatchWrapper86_copies3587 a h).2.2.1
  have c114787 := (publicBatchWrapper86_copies3587 a h).2.2.2.1
  have c114788 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_6
  simp only [← c114783, ← c114784, ← c114785] at e_1881_6
  have e_1881_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_7
  simp only [← c114786, ← c114787, k2, ← c114788] at e_1881_7
  have hr := e_1881_7
  simp only [e_1881_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10796)) := by
  have c114789 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.1
  have c114790 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.1
  have c114791 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.1
  have c114792 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.1
  have c114793 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.1
  have c114794 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_8
  simp only [← c114789, ← c114790, ← c114791] at e_1881_8
  have e_1881_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_9
  simp only [← c114792, ← c114793, k2, ← c114794] at e_1881_9
  have hr := e_1881_9
  simp only [e_1881_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10797)) := by
  have c114795 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114796 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114797 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114798 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114799 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114800 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_10
  simp only [← c114795, ← c114796, ← c114797] at e_1881_10
  have e_1881_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_11
  simp only [← c114798, ← c114799, k2, ← c114800] at e_1881_11
  have hr := e_1881_11
  simp only [e_1881_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10798)) := by
  have c114801 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114802 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114803 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114804 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114805 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114806 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_12
  simp only [← c114801, ← c114802, ← c114803] at e_1881_12
  have e_1881_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_13
  simp only [← c114804, ← c114805, k2, ← c114806] at e_1881_13
  have hr := e_1881_13
  simp only [e_1881_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10799)) := by
  have c114807 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114808 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114809 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114810 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114811 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114812 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_14
  simp only [← c114807, ← c114808, ← c114809] at e_1881_14
  have e_1881_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_15
  simp only [← c114810, ← c114811, k2, ← c114812] at e_1881_15
  have hr := e_1881_15
  simp only [e_1881_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10800)) := by
  have c114813 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114814 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114815 := (publicBatchWrapper86_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114816 := (publicBatchWrapper86_copies3588 a h).1
  have c114817 := (publicBatchWrapper86_copies3588 a h).2.1
  have c114818 := (publicBatchWrapper86_copies3588 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_16
  simp only [← c114813, ← c114814, ← c114815] at e_1881_16
  have e_1881_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_17
  simp only [← c114816, ← c114817, k2, ← c114818] at e_1881_17
  have hr := e_1881_17
  simp only [e_1881_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1881 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10801)) := by
  have c114819 := (publicBatchWrapper86_copies3588 a h).2.2.2.1
  have c114820 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.1
  have c114821 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.1
  have c114822 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.1
  have c114823 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.1
  have c114824 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1881_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_18
  simp only [← c114819, ← c114820, ← c114821] at e_1881_18
  have e_1881_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_19
  simp only [← c114822, ← c114823, k2, ← c114824] at e_1881_19
  have hr := e_1881_19
  simp only [e_1881_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10970)) := by
  have c114825 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.1
  have c114826 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114827 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114828 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114829 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114830 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_0
  simp only [← c114825, ← c114826, ← c114827] at e_1882_0
  have e_1882_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_1
  simp only [← c114828, ← c114829, k2, ← c114830] at e_1882_1
  have hr := e_1882_1
  simp only [e_1882_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10971)) := by
  have c114831 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114832 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114833 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114834 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114835 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114836 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_2
  simp only [← c114831, ← c114832, ← c114833] at e_1882_2
  have e_1882_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_3
  simp only [← c114834, ← c114835, k2, ← c114836] at e_1882_3
  have hr := e_1882_3
  simp only [e_1882_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10972)) := by
  have c114837 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114838 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114839 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114840 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114841 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114842 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_4
  simp only [← c114837, ← c114838, ← c114839] at e_1882_4
  have e_1882_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_5
  simp only [← c114840, ← c114841, k2, ← c114842] at e_1882_5
  have hr := e_1882_5
  simp only [e_1882_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10973)) := by
  have c114843 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114844 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114845 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114846 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114847 := (publicBatchWrapper86_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114848 := (publicBatchWrapper86_copies3589 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_6
  simp only [← c114843, ← c114844, ← c114845] at e_1882_6
  have e_1882_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_7
  simp only [← c114846, ← c114847, k2, ← c114848] at e_1882_7
  have hr := e_1882_7
  simp only [e_1882_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10974)) := by
  have c114849 := (publicBatchWrapper86_copies3589 a h).2.1
  have c114850 := (publicBatchWrapper86_copies3589 a h).2.2.1
  have c114851 := (publicBatchWrapper86_copies3589 a h).2.2.2.1
  have c114852 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.1
  have c114853 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.1
  have c114854 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_8
  simp only [← c114849, ← c114850, ← c114851] at e_1882_8
  have e_1882_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_9
  simp only [← c114852, ← c114853, k2, ← c114854] at e_1882_9
  have hr := e_1882_9
  simp only [e_1882_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10975)) := by
  have c114855 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.1
  have c114856 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.1
  have c114857 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.1
  have c114858 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114859 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114860 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_10
  simp only [← c114855, ← c114856, ← c114857] at e_1882_10
  have e_1882_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_11
  simp only [← c114858, ← c114859, k2, ← c114860] at e_1882_11
  have hr := e_1882_11
  simp only [e_1882_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10976)) := by
  have c114861 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114862 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114863 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114864 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114865 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114866 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_12
  simp only [← c114861, ← c114862, ← c114863] at e_1882_12
  have e_1882_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_13
  simp only [← c114864, ← c114865, k2, ← c114866] at e_1882_13
  have hr := e_1882_13
  simp only [e_1882_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10977)) := by
  have c114867 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114868 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114869 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114870 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114871 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114872 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_14
  simp only [← c114867, ← c114868, ← c114869] at e_1882_14
  have e_1882_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_15
  simp only [← c114870, ← c114871, k2, ← c114872] at e_1882_15
  have hr := e_1882_15
  simp only [e_1882_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10978)) := by
  have c114873 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114874 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114875 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114876 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114877 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114878 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_16
  simp only [← c114873, ← c114874, ← c114875] at e_1882_16
  have e_1882_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_17
  simp only [← c114876, ← c114877, k2, ← c114878] at e_1882_17
  have hr := e_1882_17
  simp only [e_1882_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1882 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10979)) := by
  have c114879 := (publicBatchWrapper86_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114880 := (publicBatchWrapper86_copies3590 a h).1
  have c114881 := (publicBatchWrapper86_copies3590 a h).2.1
  have c114882 := (publicBatchWrapper86_copies3590 a h).2.2.1
  have c114883 := (publicBatchWrapper86_copies3590 a h).2.2.2.1
  have c114884 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1882_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_18
  simp only [← c114879, ← c114880, ← c114881] at e_1882_18
  have e_1882_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_19
  simp only [← c114882, ← c114883, k2, ← c114884] at e_1882_19
  have hr := e_1882_19
  simp only [e_1882_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10980)) := by
  have c114885 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.1
  have c114886 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.1
  have c114887 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.1
  have c114888 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.1
  have c114889 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.1
  have c114890 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_0
  simp only [← c114885, ← c114886, ← c114887] at e_1883_0
  have e_1883_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_1
  simp only [← c114888, ← c114889, k2, ← c114890] at e_1883_1
  have hr := e_1883_1
  simp only [e_1883_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10981)) := by
  have c114891 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114892 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114893 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114894 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114895 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114896 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_2
  simp only [← c114891, ← c114892, ← c114893] at e_1883_2
  have e_1883_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_3
  simp only [← c114894, ← c114895, k2, ← c114896] at e_1883_3
  have hr := e_1883_3
  simp only [e_1883_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10982)) := by
  have c114897 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114898 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114899 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114900 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114901 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114902 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_4
  simp only [← c114897, ← c114898, ← c114899] at e_1883_4
  have e_1883_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_5
  simp only [← c114900, ← c114901, k2, ← c114902] at e_1883_5
  have hr := e_1883_5
  simp only [e_1883_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10983)) := by
  have c114903 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114904 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114905 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114906 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114907 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114908 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_6
  simp only [← c114903, ← c114904, ← c114905] at e_1883_6
  have e_1883_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_7
  simp only [← c114906, ← c114907, k2, ← c114908] at e_1883_7
  have hr := e_1883_7
  simp only [e_1883_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10984)) := by
  have c114909 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114910 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114911 := (publicBatchWrapper86_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114912 := (publicBatchWrapper86_copies3591 a h).1
  have c114913 := (publicBatchWrapper86_copies3591 a h).2.1
  have c114914 := (publicBatchWrapper86_copies3591 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_8
  simp only [← c114909, ← c114910, ← c114911] at e_1883_8
  have e_1883_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_9
  simp only [← c114912, ← c114913, k2, ← c114914] at e_1883_9
  have hr := e_1883_9
  simp only [e_1883_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10985)) := by
  have c114915 := (publicBatchWrapper86_copies3591 a h).2.2.2.1
  have c114916 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.1
  have c114917 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.1
  have c114918 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.1
  have c114919 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.1
  have c114920 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_10
  simp only [← c114915, ← c114916, ← c114917] at e_1883_10
  have e_1883_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_11
  simp only [← c114918, ← c114919, k2, ← c114920] at e_1883_11
  have hr := e_1883_11
  simp only [e_1883_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10986)) := by
  have c114921 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.1
  have c114922 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114923 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114924 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114925 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114926 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_12
  simp only [← c114921, ← c114922, ← c114923] at e_1883_12
  have e_1883_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_13
  simp only [← c114924, ← c114925, k2, ← c114926] at e_1883_13
  have hr := e_1883_13
  simp only [e_1883_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10987)) := by
  have c114927 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114928 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114929 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114930 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114931 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114932 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_14
  simp only [← c114927, ← c114928, ← c114929] at e_1883_14
  have e_1883_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_15
  simp only [← c114930, ← c114931, k2, ← c114932] at e_1883_15
  have hr := e_1883_15
  simp only [e_1883_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10988)) := by
  have c114933 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114934 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114935 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114936 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114937 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114938 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_16
  simp only [← c114933, ← c114934, ← c114935] at e_1883_16
  have e_1883_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_17
  simp only [← c114936, ← c114937, k2, ← c114938] at e_1883_17
  have hr := e_1883_17
  simp only [e_1883_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1883 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10989)) := by
  have c114939 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114940 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114941 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114942 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114943 := (publicBatchWrapper86_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114944 := (publicBatchWrapper86_copies3592 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1883_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_18
  simp only [← c114939, ← c114940, ← c114941] at e_1883_18
  have e_1883_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_19
  simp only [← c114942, ← c114943, k2, ← c114944] at e_1883_19
  have hr := e_1883_19
  simp only [e_1883_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10990)) := by
  have c114945 := (publicBatchWrapper86_copies3592 a h).2.1
  have c114946 := (publicBatchWrapper86_copies3592 a h).2.2.1
  have c114947 := (publicBatchWrapper86_copies3592 a h).2.2.2.1
  have c114948 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.1
  have c114949 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.1
  have c114950 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_0
  simp only [← c114945, ← c114946, ← c114947] at e_1884_0
  have e_1884_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_1
  simp only [← c114948, ← c114949, k2, ← c114950] at e_1884_1
  have hr := e_1884_1
  simp only [e_1884_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10991)) := by
  have c114951 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.1
  have c114952 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.1
  have c114953 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.1
  have c114954 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114955 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114956 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_2
  simp only [← c114951, ← c114952, ← c114953] at e_1884_2
  have e_1884_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_3
  simp only [← c114954, ← c114955, k2, ← c114956] at e_1884_3
  have hr := e_1884_3
  simp only [e_1884_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10992)) := by
  have c114957 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114958 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114959 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114960 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114961 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114962 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_4
  simp only [← c114957, ← c114958, ← c114959] at e_1884_4
  have e_1884_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_5
  simp only [← c114960, ← c114961, k2, ← c114962] at e_1884_5
  have hr := e_1884_5
  simp only [e_1884_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10993)) := by
  have c114963 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114964 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114965 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114966 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114967 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114968 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_6
  simp only [← c114963, ← c114964, ← c114965] at e_1884_6
  have e_1884_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_7
  simp only [← c114966, ← c114967, k2, ← c114968] at e_1884_7
  have hr := e_1884_7
  simp only [e_1884_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10994)) := by
  have c114969 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114970 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114971 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114972 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114973 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114974 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_8
  simp only [← c114969, ← c114970, ← c114971] at e_1884_8
  have e_1884_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_9
  simp only [← c114972, ← c114973, k2, ← c114974] at e_1884_9
  have hr := e_1884_9
  simp only [e_1884_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10995)) := by
  have c114975 := (publicBatchWrapper86_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114976 := (publicBatchWrapper86_copies3593 a h).1
  have c114977 := (publicBatchWrapper86_copies3593 a h).2.1
  have c114978 := (publicBatchWrapper86_copies3593 a h).2.2.1
  have c114979 := (publicBatchWrapper86_copies3593 a h).2.2.2.1
  have c114980 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_10
  simp only [← c114975, ← c114976, ← c114977] at e_1884_10
  have e_1884_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_11
  simp only [← c114978, ← c114979, k2, ← c114980] at e_1884_11
  have hr := e_1884_11
  simp only [e_1884_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10996)) := by
  have c114981 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.1
  have c114982 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.1
  have c114983 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.1
  have c114984 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.1
  have c114985 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.1
  have c114986 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_12
  simp only [← c114981, ← c114982, ← c114983] at e_1884_12
  have e_1884_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_13
  simp only [← c114984, ← c114985, k2, ← c114986] at e_1884_13
  have hr := e_1884_13
  simp only [e_1884_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10997)) := by
  have c114987 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114988 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114989 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114990 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114991 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114992 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_14
  simp only [← c114987, ← c114988, ← c114989] at e_1884_14
  have e_1884_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_15
  simp only [← c114990, ← c114991, k2, ← c114992] at e_1884_15
  have hr := e_1884_15
  simp only [e_1884_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10998)) := by
  have c114993 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114994 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114995 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114996 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114997 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114998 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_16
  simp only [← c114993, ← c114994, ← c114995] at e_1884_16
  have e_1884_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_17
  simp only [← c114996, ← c114997, k2, ← c114998] at e_1884_17
  have hr := e_1884_17
  simp only [e_1884_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1884 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10999)) := by
  have c114999 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115000 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115001 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115002 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115003 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115004 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1884_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_18
  simp only [← c114999, ← c115000, ← c115001] at e_1884_18
  have e_1884_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_19
  simp only [← c115002, ← c115003, k2, ← c115004] at e_1884_19
  have hr := e_1884_19
  simp only [e_1884_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 11000)) := by
  have c115005 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115006 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115007 := (publicBatchWrapper86_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115008 := (publicBatchWrapper86_copies3594 a h).1
  have c115009 := (publicBatchWrapper86_copies3594 a h).2.1
  have c115010 := (publicBatchWrapper86_copies3594 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_0
  simp only [← c115005, ← c115006, ← c115007] at e_1885_0
  have e_1885_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_1
  simp only [← c115008, ← c115009, k2, ← c115010] at e_1885_1
  have hr := e_1885_1
  simp only [e_1885_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 11001)) := by
  have c115011 := (publicBatchWrapper86_copies3594 a h).2.2.2.1
  have c115012 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.1
  have c115013 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.1
  have c115014 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.1
  have c115015 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.1
  have c115016 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_2
  simp only [← c115011, ← c115012, ← c115013] at e_1885_2
  have e_1885_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_3
  simp only [← c115014, ← c115015, k2, ← c115016] at e_1885_3
  have hr := e_1885_3
  simp only [e_1885_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11170)) := by
  have c115017 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.1
  have c115018 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115019 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115020 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115021 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115022 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_4
  simp only [← c115017, ← c115018, ← c115019] at e_1885_4
  have e_1885_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_5
  simp only [← c115020, ← c115021, k2, ← c115022] at e_1885_5
  have hr := e_1885_5
  simp only [e_1885_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11171)) := by
  have c115023 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115024 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115025 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115026 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115027 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115028 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_6
  simp only [← c115023, ← c115024, ← c115025] at e_1885_6
  have e_1885_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_7
  simp only [← c115026, ← c115027, k2, ← c115028] at e_1885_7
  have hr := e_1885_7
  simp only [e_1885_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11172)) := by
  have c115029 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115030 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115031 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115032 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115033 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115034 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_8
  simp only [← c115029, ← c115030, ← c115031] at e_1885_8
  have e_1885_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_9
  simp only [← c115032, ← c115033, k2, ← c115034] at e_1885_9
  have hr := e_1885_9
  simp only [e_1885_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11173)) := by
  have c115035 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115036 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115037 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115038 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115039 := (publicBatchWrapper86_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115040 := (publicBatchWrapper86_copies3595 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_10
  simp only [← c115035, ← c115036, ← c115037] at e_1885_10
  have e_1885_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_11
  simp only [← c115038, ← c115039, k2, ← c115040] at e_1885_11
  have hr := e_1885_11
  simp only [e_1885_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11174)) := by
  have c115041 := (publicBatchWrapper86_copies3595 a h).2.1
  have c115042 := (publicBatchWrapper86_copies3595 a h).2.2.1
  have c115043 := (publicBatchWrapper86_copies3595 a h).2.2.2.1
  have c115044 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.1
  have c115045 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.1
  have c115046 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_12
  simp only [← c115041, ← c115042, ← c115043] at e_1885_12
  have e_1885_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_13
  simp only [← c115044, ← c115045, k2, ← c115046] at e_1885_13
  have hr := e_1885_13
  simp only [e_1885_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11175)) := by
  have c115047 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.1
  have c115048 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.1
  have c115049 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.1
  have c115050 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115051 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115052 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_14
  simp only [← c115047, ← c115048, ← c115049] at e_1885_14
  have e_1885_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_15
  simp only [← c115050, ← c115051, k2, ← c115052] at e_1885_15
  have hr := e_1885_15
  simp only [e_1885_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11176)) := by
  have c115053 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115054 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115055 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115056 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115057 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115058 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_16
  simp only [← c115053, ← c115054, ← c115055] at e_1885_16
  have e_1885_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_17
  simp only [← c115056, ← c115057, k2, ← c115058] at e_1885_17
  have hr := e_1885_17
  simp only [e_1885_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1885 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11177)) := by
  have c115059 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115060 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115061 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115062 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115063 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115064 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1885_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_18
  simp only [← c115059, ← c115060, ← c115061] at e_1885_18
  have e_1885_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_19
  simp only [← c115062, ← c115063, k2, ← c115064] at e_1885_19
  have hr := e_1885_19
  simp only [e_1885_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11178)) := by
  have c115065 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115066 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115067 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115068 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115069 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115070 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_0
  simp only [← c115065, ← c115066, ← c115067] at e_1886_0
  have e_1886_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_1
  simp only [← c115068, ← c115069, k2, ← c115070] at e_1886_1
  have hr := e_1886_1
  simp only [e_1886_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11179)) := by
  have c115071 := (publicBatchWrapper86_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115072 := (publicBatchWrapper86_copies3596 a h).1
  have c115073 := (publicBatchWrapper86_copies3596 a h).2.1
  have c115074 := (publicBatchWrapper86_copies3596 a h).2.2.1
  have c115075 := (publicBatchWrapper86_copies3596 a h).2.2.2.1
  have c115076 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_2
  simp only [← c115071, ← c115072, ← c115073] at e_1886_2
  have e_1886_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_3
  simp only [← c115074, ← c115075, k2, ← c115076] at e_1886_3
  have hr := e_1886_3
  simp only [e_1886_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11180)) := by
  have c115077 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.1
  have c115078 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.1
  have c115079 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.1
  have c115080 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.1
  have c115081 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.1
  have c115082 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_4
  simp only [← c115077, ← c115078, ← c115079] at e_1886_4
  have e_1886_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_5
  simp only [← c115080, ← c115081, k2, ← c115082] at e_1886_5
  have hr := e_1886_5
  simp only [e_1886_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11181)) := by
  have c115083 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115084 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115085 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115086 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115087 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115088 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_6
  simp only [← c115083, ← c115084, ← c115085] at e_1886_6
  have e_1886_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_7
  simp only [← c115086, ← c115087, k2, ← c115088] at e_1886_7
  have hr := e_1886_7
  simp only [e_1886_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11182)) := by
  have c115089 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115090 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115091 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115092 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115093 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115094 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_8
  simp only [← c115089, ← c115090, ← c115091] at e_1886_8
  have e_1886_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_9
  simp only [← c115092, ← c115093, k2, ← c115094] at e_1886_9
  have hr := e_1886_9
  simp only [e_1886_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11183)) := by
  have c115095 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115096 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115097 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115098 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115099 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115100 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_10
  simp only [← c115095, ← c115096, ← c115097] at e_1886_10
  have e_1886_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_11
  simp only [← c115098, ← c115099, k2, ← c115100] at e_1886_11
  have hr := e_1886_11
  simp only [e_1886_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11184)) := by
  have c115101 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115102 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115103 := (publicBatchWrapper86_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115104 := (publicBatchWrapper86_copies3597 a h).1
  have c115105 := (publicBatchWrapper86_copies3597 a h).2.1
  have c115106 := (publicBatchWrapper86_copies3597 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_12
  simp only [← c115101, ← c115102, ← c115103] at e_1886_12
  have e_1886_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_13
  simp only [← c115104, ← c115105, k2, ← c115106] at e_1886_13
  have hr := e_1886_13
  simp only [e_1886_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11185)) := by
  have c115107 := (publicBatchWrapper86_copies3597 a h).2.2.2.1
  have c115108 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.1
  have c115109 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.1
  have c115110 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.1
  have c115111 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.1
  have c115112 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_14
  simp only [← c115107, ← c115108, ← c115109] at e_1886_14
  have e_1886_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_15
  simp only [← c115110, ← c115111, k2, ← c115112] at e_1886_15
  have hr := e_1886_15
  simp only [e_1886_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11186)) := by
  have c115113 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.1
  have c115114 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115115 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115116 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115117 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115118 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_16
  simp only [← c115113, ← c115114, ← c115115] at e_1886_16
  have e_1886_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_17
  simp only [← c115116, ← c115117, k2, ← c115118] at e_1886_17
  have hr := e_1886_17
  simp only [e_1886_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1886 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11187)) := by
  have c115119 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115120 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115121 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115122 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115123 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115124 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1886_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_18
  simp only [← c115119, ← c115120, ← c115121] at e_1886_18
  have e_1886_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_19
  simp only [← c115122, ← c115123, k2, ← c115124] at e_1886_19
  have hr := e_1886_19
  simp only [e_1886_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11188)) := by
  have c115125 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115126 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115127 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115128 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115129 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115130 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_0
  simp only [← c115125, ← c115126, ← c115127] at e_1887_0
  have e_1887_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_1
  simp only [← c115128, ← c115129, k2, ← c115130] at e_1887_1
  have hr := e_1887_1
  simp only [e_1887_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11189)) := by
  have c115131 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115132 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115133 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115134 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115135 := (publicBatchWrapper86_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115136 := (publicBatchWrapper86_copies3598 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_2
  simp only [← c115131, ← c115132, ← c115133] at e_1887_2
  have e_1887_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_3
  simp only [← c115134, ← c115135, k2, ← c115136] at e_1887_3
  have hr := e_1887_3
  simp only [e_1887_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11190)) := by
  have c115137 := (publicBatchWrapper86_copies3598 a h).2.1
  have c115138 := (publicBatchWrapper86_copies3598 a h).2.2.1
  have c115139 := (publicBatchWrapper86_copies3598 a h).2.2.2.1
  have c115140 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.1
  have c115141 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.1
  have c115142 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_4
  simp only [← c115137, ← c115138, ← c115139] at e_1887_4
  have e_1887_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_5
  simp only [← c115140, ← c115141, k2, ← c115142] at e_1887_5
  have hr := e_1887_5
  simp only [e_1887_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11191)) := by
  have c115143 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.1
  have c115144 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.1
  have c115145 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.1
  have c115146 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115147 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115148 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_6
  simp only [← c115143, ← c115144, ← c115145] at e_1887_6
  have e_1887_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_7
  simp only [← c115146, ← c115147, k2, ← c115148] at e_1887_7
  have hr := e_1887_7
  simp only [e_1887_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11192)) := by
  have c115149 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115150 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115151 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115152 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115153 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115154 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_8
  simp only [← c115149, ← c115150, ← c115151] at e_1887_8
  have e_1887_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_9
  simp only [← c115152, ← c115153, k2, ← c115154] at e_1887_9
  have hr := e_1887_9
  simp only [e_1887_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11193)) := by
  have c115155 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115156 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115157 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115158 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115159 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115160 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_10
  simp only [← c115155, ← c115156, ← c115157] at e_1887_10
  have e_1887_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_11
  simp only [← c115158, ← c115159, k2, ← c115160] at e_1887_11
  have hr := e_1887_11
  simp only [e_1887_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11194)) := by
  have c115161 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115162 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115163 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115164 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115165 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115166 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_12
  simp only [← c115161, ← c115162, ← c115163] at e_1887_12
  have e_1887_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_13
  simp only [← c115164, ← c115165, k2, ← c115166] at e_1887_13
  have hr := e_1887_13
  simp only [e_1887_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11195)) := by
  have c115167 := (publicBatchWrapper86_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115168 := (publicBatchWrapper86_copies3599 a h).1
  have c115169 := (publicBatchWrapper86_copies3599 a h).2.1
  have c115170 := (publicBatchWrapper86_copies3599 a h).2.2.1
  have c115171 := (publicBatchWrapper86_copies3599 a h).2.2.2.1
  have c115172 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_14
  simp only [← c115167, ← c115168, ← c115169] at e_1887_14
  have e_1887_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_15
  simp only [← c115170, ← c115171, k2, ← c115172] at e_1887_15
  have hr := e_1887_15
  simp only [e_1887_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11196)) := by
  have c115173 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.1
  have c115174 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.1
  have c115175 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.1
  have c115176 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.1
  have c115177 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.1
  have c115178 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_16
  simp only [← c115173, ← c115174, ← c115175] at e_1887_16
  have e_1887_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_17
  simp only [← c115176, ← c115177, k2, ← c115178] at e_1887_17
  have hr := e_1887_17
  simp only [e_1887_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1887 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11197)) := by
  have c115179 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115180 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115181 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115182 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115183 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115184 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1887_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_18
  simp only [← c115179, ← c115180, ← c115181] at e_1887_18
  have e_1887_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_19
  simp only [← c115182, ← c115183, k2, ← c115184] at e_1887_19
  have hr := e_1887_19
  simp only [e_1887_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11198)) := by
  have c115185 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115186 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115187 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115188 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115189 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115190 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_0
  simp only [← c115185, ← c115186, ← c115187] at e_1888_0
  have e_1888_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_1
  simp only [← c115188, ← c115189, k2, ← c115190] at e_1888_1
  have hr := e_1888_1
  simp only [e_1888_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11199)) := by
  have c115191 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115192 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115193 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115194 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115195 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115196 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_2
  simp only [← c115191, ← c115192, ← c115193] at e_1888_2
  have e_1888_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_3
  simp only [← c115194, ← c115195, k2, ← c115196] at e_1888_3
  have hr := e_1888_3
  simp only [e_1888_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11200)) := by
  have c115197 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115198 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115199 := (publicBatchWrapper86_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115200 := (publicBatchWrapper86_copies3600 a h).1
  have c115201 := (publicBatchWrapper86_copies3600 a h).2.1
  have c115202 := (publicBatchWrapper86_copies3600 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_4
  simp only [← c115197, ← c115198, ← c115199] at e_1888_4
  have e_1888_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_5
  simp only [← c115200, ← c115201, k2, ← c115202] at e_1888_5
  have hr := e_1888_5
  simp only [e_1888_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11201)) := by
  have c115203 := (publicBatchWrapper86_copies3600 a h).2.2.2.1
  have c115204 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.1
  have c115205 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.1
  have c115206 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.1
  have c115207 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.1
  have c115208 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_6
  simp only [← c115203, ← c115204, ← c115205] at e_1888_6
  have e_1888_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_7
  simp only [← c115206, ← c115207, k2, ← c115208] at e_1888_7
  have hr := e_1888_7
  simp only [e_1888_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11370)) := by
  have c115209 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.1
  have c115210 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115211 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115212 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115213 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115214 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_8
  simp only [← c115209, ← c115210, ← c115211] at e_1888_8
  have e_1888_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_9
  simp only [← c115212, ← c115213, k2, ← c115214] at e_1888_9
  have hr := e_1888_9
  simp only [e_1888_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11371)) := by
  have c115215 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115216 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115217 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115218 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115219 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115220 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_10
  simp only [← c115215, ← c115216, ← c115217] at e_1888_10
  have e_1888_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_11
  simp only [← c115218, ← c115219, k2, ← c115220] at e_1888_11
  have hr := e_1888_11
  simp only [e_1888_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11372)) := by
  have c115221 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115222 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115223 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115224 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115225 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115226 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_12
  simp only [← c115221, ← c115222, ← c115223] at e_1888_12
  have e_1888_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_13
  simp only [← c115224, ← c115225, k2, ← c115226] at e_1888_13
  have hr := e_1888_13
  simp only [e_1888_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11373)) := by
  have c115227 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115228 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115229 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115230 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115231 := (publicBatchWrapper86_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115232 := (publicBatchWrapper86_copies3601 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_14
  simp only [← c115227, ← c115228, ← c115229] at e_1888_14
  have e_1888_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_15
  simp only [← c115230, ← c115231, k2, ← c115232] at e_1888_15
  have hr := e_1888_15
  simp only [e_1888_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11374)) := by
  have c115233 := (publicBatchWrapper86_copies3601 a h).2.1
  have c115234 := (publicBatchWrapper86_copies3601 a h).2.2.1
  have c115235 := (publicBatchWrapper86_copies3601 a h).2.2.2.1
  have c115236 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.1
  have c115237 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.1
  have c115238 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_16
  simp only [← c115233, ← c115234, ← c115235] at e_1888_16
  have e_1888_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_17
  simp only [← c115236, ← c115237, k2, ← c115238] at e_1888_17
  have hr := e_1888_17
  simp only [e_1888_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1888 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11375)) := by
  have c115239 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.1
  have c115240 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.1
  have c115241 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.1
  have c115242 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115243 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115244 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1888_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_18
  simp only [← c115239, ← c115240, ← c115241] at e_1888_18
  have e_1888_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_19
  simp only [← c115242, ← c115243, k2, ← c115244] at e_1888_19
  have hr := e_1888_19
  simp only [e_1888_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11376)) := by
  have c115245 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115246 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115247 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115248 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115249 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115250 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_0
  simp only [← c115245, ← c115246, ← c115247] at e_1889_0
  have e_1889_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_1
  simp only [← c115248, ← c115249, k2, ← c115250] at e_1889_1
  have hr := e_1889_1
  simp only [e_1889_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11377)) := by
  have c115251 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115252 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115253 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115254 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115255 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115256 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_2
  simp only [← c115251, ← c115252, ← c115253] at e_1889_2
  have e_1889_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_3
  simp only [← c115254, ← c115255, k2, ← c115256] at e_1889_3
  have hr := e_1889_3
  simp only [e_1889_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11378)) := by
  have c115257 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115258 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115259 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115260 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115261 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115262 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_4
  simp only [← c115257, ← c115258, ← c115259] at e_1889_4
  have e_1889_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_5
  simp only [← c115260, ← c115261, k2, ← c115262] at e_1889_5
  have hr := e_1889_5
  simp only [e_1889_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11379)) := by
  have c115263 := (publicBatchWrapper86_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115264 := (publicBatchWrapper86_copies3602 a h).1
  have c115265 := (publicBatchWrapper86_copies3602 a h).2.1
  have c115266 := (publicBatchWrapper86_copies3602 a h).2.2.1
  have c115267 := (publicBatchWrapper86_copies3602 a h).2.2.2.1
  have c115268 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_6
  simp only [← c115263, ← c115264, ← c115265] at e_1889_6
  have e_1889_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_7
  simp only [← c115266, ← c115267, k2, ← c115268] at e_1889_7
  have hr := e_1889_7
  simp only [e_1889_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11380)) := by
  have c115269 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.1
  have c115270 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.1
  have c115271 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.1
  have c115272 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.1
  have c115273 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.1
  have c115274 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_8
  simp only [← c115269, ← c115270, ← c115271] at e_1889_8
  have e_1889_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_9
  simp only [← c115272, ← c115273, k2, ← c115274] at e_1889_9
  have hr := e_1889_9
  simp only [e_1889_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11381)) := by
  have c115275 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115276 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115277 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115278 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115279 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115280 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_10
  simp only [← c115275, ← c115276, ← c115277] at e_1889_10
  have e_1889_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_11
  simp only [← c115278, ← c115279, k2, ← c115280] at e_1889_11
  have hr := e_1889_11
  simp only [e_1889_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11382)) := by
  have c115281 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115282 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115283 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115284 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115285 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115286 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_12
  simp only [← c115281, ← c115282, ← c115283] at e_1889_12
  have e_1889_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_13
  simp only [← c115284, ← c115285, k2, ← c115286] at e_1889_13
  have hr := e_1889_13
  simp only [e_1889_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11383)) := by
  have c115287 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115288 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115289 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115290 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115291 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115292 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_14
  simp only [← c115287, ← c115288, ← c115289] at e_1889_14
  have e_1889_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_15
  simp only [← c115290, ← c115291, k2, ← c115292] at e_1889_15
  have hr := e_1889_15
  simp only [e_1889_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11384)) := by
  have c115293 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115294 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115295 := (publicBatchWrapper86_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115296 := (publicBatchWrapper86_copies3603 a h).1
  have c115297 := (publicBatchWrapper86_copies3603 a h).2.1
  have c115298 := (publicBatchWrapper86_copies3603 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_16
  simp only [← c115293, ← c115294, ← c115295] at e_1889_16
  have e_1889_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_17
  simp only [← c115296, ← c115297, k2, ← c115298] at e_1889_17
  have hr := e_1889_17
  simp only [e_1889_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1889 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11385)) := by
  have c115299 := (publicBatchWrapper86_copies3603 a h).2.2.2.1
  have c115300 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.1
  have c115301 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.1
  have c115302 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.1
  have c115303 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.1
  have c115304 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1889_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_18
  simp only [← c115299, ← c115300, ← c115301] at e_1889_18
  have e_1889_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_19
  simp only [← c115302, ← c115303, k2, ← c115304] at e_1889_19
  have hr := e_1889_19
  simp only [e_1889_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11386)) := by
  have c115305 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.1
  have c115306 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115307 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115308 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115309 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115310 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_0
  simp only [← c115305, ← c115306, ← c115307] at e_1890_0
  have e_1890_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_1
  simp only [← c115308, ← c115309, k2, ← c115310] at e_1890_1
  have hr := e_1890_1
  simp only [e_1890_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11387)) := by
  have c115311 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115312 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115313 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115314 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115315 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115316 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_2
  simp only [← c115311, ← c115312, ← c115313] at e_1890_2
  have e_1890_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_3
  simp only [← c115314, ← c115315, k2, ← c115316] at e_1890_3
  have hr := e_1890_3
  simp only [e_1890_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11388)) := by
  have c115317 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115318 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115319 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115320 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115321 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115322 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_4
  simp only [← c115317, ← c115318, ← c115319] at e_1890_4
  have e_1890_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_5
  simp only [← c115320, ← c115321, k2, ← c115322] at e_1890_5
  have hr := e_1890_5
  simp only [e_1890_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11389)) := by
  have c115323 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115324 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115325 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115326 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115327 := (publicBatchWrapper86_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115328 := (publicBatchWrapper86_copies3604 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_6
  simp only [← c115323, ← c115324, ← c115325] at e_1890_6
  have e_1890_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_7
  simp only [← c115326, ← c115327, k2, ← c115328] at e_1890_7
  have hr := e_1890_7
  simp only [e_1890_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11390)) := by
  have c115329 := (publicBatchWrapper86_copies3604 a h).2.1
  have c115330 := (publicBatchWrapper86_copies3604 a h).2.2.1
  have c115331 := (publicBatchWrapper86_copies3604 a h).2.2.2.1
  have c115332 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.1
  have c115333 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.1
  have c115334 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_8
  simp only [← c115329, ← c115330, ← c115331] at e_1890_8
  have e_1890_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_9
  simp only [← c115332, ← c115333, k2, ← c115334] at e_1890_9
  have hr := e_1890_9
  simp only [e_1890_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11391)) := by
  have c115335 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.1
  have c115336 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.1
  have c115337 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.1
  have c115338 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115339 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115340 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_10
  simp only [← c115335, ← c115336, ← c115337] at e_1890_10
  have e_1890_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_11
  simp only [← c115338, ← c115339, k2, ← c115340] at e_1890_11
  have hr := e_1890_11
  simp only [e_1890_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11392)) := by
  have c115341 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115342 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115343 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115344 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115345 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115346 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_12
  simp only [← c115341, ← c115342, ← c115343] at e_1890_12
  have e_1890_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_13
  simp only [← c115344, ← c115345, k2, ← c115346] at e_1890_13
  have hr := e_1890_13
  simp only [e_1890_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11393)) := by
  have c115347 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115348 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115349 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115350 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115351 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115352 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_14
  simp only [← c115347, ← c115348, ← c115349] at e_1890_14
  have e_1890_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_15
  simp only [← c115350, ← c115351, k2, ← c115352] at e_1890_15
  have hr := e_1890_15
  simp only [e_1890_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11394)) := by
  have c115353 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115354 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115355 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115356 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115357 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115358 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_16
  simp only [← c115353, ← c115354, ← c115355] at e_1890_16
  have e_1890_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_17
  simp only [← c115356, ← c115357, k2, ← c115358] at e_1890_17
  have hr := e_1890_17
  simp only [e_1890_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1890 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11395)) := by
  have c115359 := (publicBatchWrapper86_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115360 := (publicBatchWrapper86_copies3605 a h).1
  have c115361 := (publicBatchWrapper86_copies3605 a h).2.1
  have c115362 := (publicBatchWrapper86_copies3605 a h).2.2.1
  have c115363 := (publicBatchWrapper86_copies3605 a h).2.2.2.1
  have c115364 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1890_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_18
  simp only [← c115359, ← c115360, ← c115361] at e_1890_18
  have e_1890_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_19
  simp only [← c115362, ← c115363, k2, ← c115364] at e_1890_19
  have hr := e_1890_19
  simp only [e_1890_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11396)) := by
  have c115365 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.1
  have c115366 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.1
  have c115367 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.1
  have c115368 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.1
  have c115369 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.1
  have c115370 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_0
  simp only [← c115365, ← c115366, ← c115367] at e_1891_0
  have e_1891_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_1
  simp only [← c115368, ← c115369, k2, ← c115370] at e_1891_1
  have hr := e_1891_1
  simp only [e_1891_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11397)) := by
  have c115371 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115372 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115373 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115374 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115375 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115376 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_2
  simp only [← c115371, ← c115372, ← c115373] at e_1891_2
  have e_1891_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_3
  simp only [← c115374, ← c115375, k2, ← c115376] at e_1891_3
  have hr := e_1891_3
  simp only [e_1891_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11398)) := by
  have c115377 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115378 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115379 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115380 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115381 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115382 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_4
  simp only [← c115377, ← c115378, ← c115379] at e_1891_4
  have e_1891_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_5
  simp only [← c115380, ← c115381, k2, ← c115382] at e_1891_5
  have hr := e_1891_5
  simp only [e_1891_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11399)) := by
  have c115383 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115384 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115385 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115386 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115387 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115388 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_6
  simp only [← c115383, ← c115384, ← c115385] at e_1891_6
  have e_1891_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_7
  simp only [← c115386, ← c115387, k2, ← c115388] at e_1891_7
  have hr := e_1891_7
  simp only [e_1891_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11400)) := by
  have c115389 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115390 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115391 := (publicBatchWrapper86_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115392 := (publicBatchWrapper86_copies3606 a h).1
  have c115393 := (publicBatchWrapper86_copies3606 a h).2.1
  have c115394 := (publicBatchWrapper86_copies3606 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_8
  simp only [← c115389, ← c115390, ← c115391] at e_1891_8
  have e_1891_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_9
  simp only [← c115392, ← c115393, k2, ← c115394] at e_1891_9
  have hr := e_1891_9
  simp only [e_1891_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11401)) := by
  have c115395 := (publicBatchWrapper86_copies3606 a h).2.2.2.1
  have c115396 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.1
  have c115397 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.1
  have c115398 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.1
  have c115399 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.1
  have c115400 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_10
  simp only [← c115395, ← c115396, ← c115397] at e_1891_10
  have e_1891_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_11
  simp only [← c115398, ← c115399, k2, ← c115400] at e_1891_11
  have hr := e_1891_11
  simp only [e_1891_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11570)) := by
  have c115401 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.1
  have c115402 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115403 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115404 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115405 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115406 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_12
  simp only [← c115401, ← c115402, ← c115403] at e_1891_12
  have e_1891_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_13
  simp only [← c115404, ← c115405, k2, ← c115406] at e_1891_13
  have hr := e_1891_13
  simp only [e_1891_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11571)) := by
  have c115407 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115408 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115409 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115410 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115411 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115412 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_14
  simp only [← c115407, ← c115408, ← c115409] at e_1891_14
  have e_1891_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_15
  simp only [← c115410, ← c115411, k2, ← c115412] at e_1891_15
  have hr := e_1891_15
  simp only [e_1891_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11572)) := by
  have c115413 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115414 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115415 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115416 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115417 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115418 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_16
  simp only [← c115413, ← c115414, ← c115415] at e_1891_16
  have e_1891_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_17
  simp only [← c115416, ← c115417, k2, ← c115418] at e_1891_17
  have hr := e_1891_17
  simp only [e_1891_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1891 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11573)) := by
  have c115419 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115420 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115421 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115422 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115423 := (publicBatchWrapper86_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115424 := (publicBatchWrapper86_copies3607 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1891_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_18
  simp only [← c115419, ← c115420, ← c115421] at e_1891_18
  have e_1891_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_19
  simp only [← c115422, ← c115423, k2, ← c115424] at e_1891_19
  have hr := e_1891_19
  simp only [e_1891_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11574)) := by
  have c115425 := (publicBatchWrapper86_copies3607 a h).2.1
  have c115426 := (publicBatchWrapper86_copies3607 a h).2.2.1
  have c115427 := (publicBatchWrapper86_copies3607 a h).2.2.2.1
  have c115428 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.1
  have c115429 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.1
  have c115430 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_0
  simp only [← c115425, ← c115426, ← c115427] at e_1892_0
  have e_1892_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_1
  simp only [← c115428, ← c115429, k2, ← c115430] at e_1892_1
  have hr := e_1892_1
  simp only [e_1892_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11575)) := by
  have c115431 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.1
  have c115432 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.1
  have c115433 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.1
  have c115434 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115435 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115436 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_2
  simp only [← c115431, ← c115432, ← c115433] at e_1892_2
  have e_1892_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_3
  simp only [← c115434, ← c115435, k2, ← c115436] at e_1892_3
  have hr := e_1892_3
  simp only [e_1892_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11576)) := by
  have c115437 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115438 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115439 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115440 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115441 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115442 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_4
  simp only [← c115437, ← c115438, ← c115439] at e_1892_4
  have e_1892_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_5
  simp only [← c115440, ← c115441, k2, ← c115442] at e_1892_5
  have hr := e_1892_5
  simp only [e_1892_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11577)) := by
  have c115443 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115444 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115445 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115446 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115447 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115448 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_6
  simp only [← c115443, ← c115444, ← c115445] at e_1892_6
  have e_1892_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_7
  simp only [← c115446, ← c115447, k2, ← c115448] at e_1892_7
  have hr := e_1892_7
  simp only [e_1892_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11578)) := by
  have c115449 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115450 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115451 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115452 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115453 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115454 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_8
  simp only [← c115449, ← c115450, ← c115451] at e_1892_8
  have e_1892_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_9
  simp only [← c115452, ← c115453, k2, ← c115454] at e_1892_9
  have hr := e_1892_9
  simp only [e_1892_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11579)) := by
  have c115455 := (publicBatchWrapper86_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115456 := (publicBatchWrapper86_copies3608 a h).1
  have c115457 := (publicBatchWrapper86_copies3608 a h).2.1
  have c115458 := (publicBatchWrapper86_copies3608 a h).2.2.1
  have c115459 := (publicBatchWrapper86_copies3608 a h).2.2.2.1
  have c115460 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_10
  simp only [← c115455, ← c115456, ← c115457] at e_1892_10
  have e_1892_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_11
  simp only [← c115458, ← c115459, k2, ← c115460] at e_1892_11
  have hr := e_1892_11
  simp only [e_1892_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11580)) := by
  have c115461 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.1
  have c115462 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.1
  have c115463 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.1
  have c115464 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.1
  have c115465 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.1
  have c115466 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_12
  simp only [← c115461, ← c115462, ← c115463] at e_1892_12
  have e_1892_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_13
  simp only [← c115464, ← c115465, k2, ← c115466] at e_1892_13
  have hr := e_1892_13
  simp only [e_1892_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11581)) := by
  have c115467 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115468 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115469 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115470 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115471 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115472 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_14
  simp only [← c115467, ← c115468, ← c115469] at e_1892_14
  have e_1892_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_15
  simp only [← c115470, ← c115471, k2, ← c115472] at e_1892_15
  have hr := e_1892_15
  simp only [e_1892_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11582)) := by
  have c115473 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115474 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115475 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115476 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115477 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115478 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_16
  simp only [← c115473, ← c115474, ← c115475] at e_1892_16
  have e_1892_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_17
  simp only [← c115476, ← c115477, k2, ← c115478] at e_1892_17
  have hr := e_1892_17
  simp only [e_1892_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1892 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11583)) := by
  have c115479 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115480 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115481 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115482 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115483 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115484 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1892_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_18
  simp only [← c115479, ← c115480, ← c115481] at e_1892_18
  have e_1892_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_19
  simp only [← c115482, ← c115483, k2, ← c115484] at e_1892_19
  have hr := e_1892_19
  simp only [e_1892_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11584)) := by
  have c115485 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115486 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115487 := (publicBatchWrapper86_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115488 := (publicBatchWrapper86_copies3609 a h).1
  have c115489 := (publicBatchWrapper86_copies3609 a h).2.1
  have c115490 := (publicBatchWrapper86_copies3609 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_0
  simp only [← c115485, ← c115486, ← c115487] at e_1893_0
  have e_1893_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_1
  simp only [← c115488, ← c115489, k2, ← c115490] at e_1893_1
  have hr := e_1893_1
  simp only [e_1893_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11585)) := by
  have c115491 := (publicBatchWrapper86_copies3609 a h).2.2.2.1
  have c115492 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.1
  have c115493 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.1
  have c115494 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.1
  have c115495 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.1
  have c115496 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_2
  simp only [← c115491, ← c115492, ← c115493] at e_1893_2
  have e_1893_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_3
  simp only [← c115494, ← c115495, k2, ← c115496] at e_1893_3
  have hr := e_1893_3
  simp only [e_1893_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11586)) := by
  have c115497 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.1
  have c115498 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115499 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115500 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115501 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115502 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_4
  simp only [← c115497, ← c115498, ← c115499] at e_1893_4
  have e_1893_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_5
  simp only [← c115500, ← c115501, k2, ← c115502] at e_1893_5
  have hr := e_1893_5
  simp only [e_1893_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11587)) := by
  have c115503 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115504 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115505 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115506 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115507 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115508 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_6
  simp only [← c115503, ← c115504, ← c115505] at e_1893_6
  have e_1893_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_7
  simp only [← c115506, ← c115507, k2, ← c115508] at e_1893_7
  have hr := e_1893_7
  simp only [e_1893_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11588)) := by
  have c115509 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115510 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115511 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115512 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115513 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115514 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_8
  simp only [← c115509, ← c115510, ← c115511] at e_1893_8
  have e_1893_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_9
  simp only [← c115512, ← c115513, k2, ← c115514] at e_1893_9
  have hr := e_1893_9
  simp only [e_1893_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11589)) := by
  have c115515 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115516 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115517 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115518 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115519 := (publicBatchWrapper86_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115520 := (publicBatchWrapper86_copies3610 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_10
  simp only [← c115515, ← c115516, ← c115517] at e_1893_10
  have e_1893_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_11
  simp only [← c115518, ← c115519, k2, ← c115520] at e_1893_11
  have hr := e_1893_11
  simp only [e_1893_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11590)) := by
  have c115521 := (publicBatchWrapper86_copies3610 a h).2.1
  have c115522 := (publicBatchWrapper86_copies3610 a h).2.2.1
  have c115523 := (publicBatchWrapper86_copies3610 a h).2.2.2.1
  have c115524 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.1
  have c115525 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.1
  have c115526 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_12
  simp only [← c115521, ← c115522, ← c115523] at e_1893_12
  have e_1893_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_13
  simp only [← c115524, ← c115525, k2, ← c115526] at e_1893_13
  have hr := e_1893_13
  simp only [e_1893_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11591)) := by
  have c115527 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.1
  have c115528 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.1
  have c115529 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.1
  have c115530 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115531 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115532 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_14
  simp only [← c115527, ← c115528, ← c115529] at e_1893_14
  have e_1893_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_15
  simp only [← c115530, ← c115531, k2, ← c115532] at e_1893_15
  have hr := e_1893_15
  simp only [e_1893_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11592)) := by
  have c115533 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115534 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115535 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115536 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115537 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115538 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_16
  simp only [← c115533, ← c115534, ← c115535] at e_1893_16
  have e_1893_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_17
  simp only [← c115536, ← c115537, k2, ← c115538] at e_1893_17
  have hr := e_1893_17
  simp only [e_1893_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1893 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11593)) := by
  have c115539 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115540 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115541 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115542 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115543 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115544 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1893_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_18
  simp only [← c115539, ← c115540, ← c115541] at e_1893_18
  have e_1893_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_19
  simp only [← c115542, ← c115543, k2, ← c115544] at e_1893_19
  have hr := e_1893_19
  simp only [e_1893_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11594)) := by
  have c115545 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115546 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115547 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115548 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115549 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115550 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_0
  simp only [← c115545, ← c115546, ← c115547] at e_1894_0
  have e_1894_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_1
  simp only [← c115548, ← c115549, k2, ← c115550] at e_1894_1
  have hr := e_1894_1
  simp only [e_1894_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11595)) := by
  have c115551 := (publicBatchWrapper86_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115552 := (publicBatchWrapper86_copies3611 a h).1
  have c115553 := (publicBatchWrapper86_copies3611 a h).2.1
  have c115554 := (publicBatchWrapper86_copies3611 a h).2.2.1
  have c115555 := (publicBatchWrapper86_copies3611 a h).2.2.2.1
  have c115556 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_2
  simp only [← c115551, ← c115552, ← c115553] at e_1894_2
  have e_1894_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_3
  simp only [← c115554, ← c115555, k2, ← c115556] at e_1894_3
  have hr := e_1894_3
  simp only [e_1894_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11596)) := by
  have c115557 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.1
  have c115558 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.1
  have c115559 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.1
  have c115560 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.1
  have c115561 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.1
  have c115562 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_4
  simp only [← c115557, ← c115558, ← c115559] at e_1894_4
  have e_1894_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_5
  simp only [← c115560, ← c115561, k2, ← c115562] at e_1894_5
  have hr := e_1894_5
  simp only [e_1894_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11597)) := by
  have c115563 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115564 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115565 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115566 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115567 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115568 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_6
  simp only [← c115563, ← c115564, ← c115565] at e_1894_6
  have e_1894_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_7
  simp only [← c115566, ← c115567, k2, ← c115568] at e_1894_7
  have hr := e_1894_7
  simp only [e_1894_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11598)) := by
  have c115569 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115570 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115571 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115572 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115573 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115574 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_8
  simp only [← c115569, ← c115570, ← c115571] at e_1894_8
  have e_1894_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_9
  simp only [← c115572, ← c115573, k2, ← c115574] at e_1894_9
  have hr := e_1894_9
  simp only [e_1894_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11599)) := by
  have c115575 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115576 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115577 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115578 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115579 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115580 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_10
  simp only [← c115575, ← c115576, ← c115577] at e_1894_10
  have e_1894_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_11
  simp only [← c115578, ← c115579, k2, ← c115580] at e_1894_11
  have hr := e_1894_11
  simp only [e_1894_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11600)) := by
  have c115581 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115582 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115583 := (publicBatchWrapper86_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115584 := (publicBatchWrapper86_copies3612 a h).1
  have c115585 := (publicBatchWrapper86_copies3612 a h).2.1
  have c115586 := (publicBatchWrapper86_copies3612 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_12
  simp only [← c115581, ← c115582, ← c115583] at e_1894_12
  have e_1894_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_13
  simp only [← c115584, ← c115585, k2, ← c115586] at e_1894_13
  have hr := e_1894_13
  simp only [e_1894_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11601)) := by
  have c115587 := (publicBatchWrapper86_copies3612 a h).2.2.2.1
  have c115588 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.1
  have c115589 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.1
  have c115590 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.1
  have c115591 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.1
  have c115592 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_14
  simp only [← c115587, ← c115588, ← c115589] at e_1894_14
  have e_1894_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_15
  simp only [← c115590, ← c115591, k2, ← c115592] at e_1894_15
  have hr := e_1894_15
  simp only [e_1894_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11770)) := by
  have c115593 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.1
  have c115594 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115595 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115596 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115597 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115598 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_16
  simp only [← c115593, ← c115594, ← c115595] at e_1894_16
  have e_1894_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_17
  simp only [← c115596, ← c115597, k2, ← c115598] at e_1894_17
  have hr := e_1894_17
  simp only [e_1894_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1894 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11771)) := by
  have c115599 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115600 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115601 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115602 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115603 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115604 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1894_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_18
  simp only [← c115599, ← c115600, ← c115601] at e_1894_18
  have e_1894_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_19
  simp only [← c115602, ← c115603, k2, ← c115604] at e_1894_19
  have hr := e_1894_19
  simp only [e_1894_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11772)) := by
  have c115605 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115606 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115607 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115608 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115609 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115610 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_0
  simp only [← c115605, ← c115606, ← c115607] at e_1895_0
  have e_1895_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_1
  simp only [← c115608, ← c115609, k2, ← c115610] at e_1895_1
  have hr := e_1895_1
  simp only [e_1895_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11773)) := by
  have c115611 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115612 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115613 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115614 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115615 := (publicBatchWrapper86_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115616 := (publicBatchWrapper86_copies3613 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_2
  simp only [← c115611, ← c115612, ← c115613] at e_1895_2
  have e_1895_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_3
  simp only [← c115614, ← c115615, k2, ← c115616] at e_1895_3
  have hr := e_1895_3
  simp only [e_1895_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11774)) := by
  have c115617 := (publicBatchWrapper86_copies3613 a h).2.1
  have c115618 := (publicBatchWrapper86_copies3613 a h).2.2.1
  have c115619 := (publicBatchWrapper86_copies3613 a h).2.2.2.1
  have c115620 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.1
  have c115621 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.1
  have c115622 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_4
  simp only [← c115617, ← c115618, ← c115619] at e_1895_4
  have e_1895_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_5
  simp only [← c115620, ← c115621, k2, ← c115622] at e_1895_5
  have hr := e_1895_5
  simp only [e_1895_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11775)) := by
  have c115623 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.1
  have c115624 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.1
  have c115625 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.1
  have c115626 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115627 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115628 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_6
  simp only [← c115623, ← c115624, ← c115625] at e_1895_6
  have e_1895_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_7
  simp only [← c115626, ← c115627, k2, ← c115628] at e_1895_7
  have hr := e_1895_7
  simp only [e_1895_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11776)) := by
  have c115629 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115630 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115631 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115632 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115633 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115634 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_8
  simp only [← c115629, ← c115630, ← c115631] at e_1895_8
  have e_1895_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_9
  simp only [← c115632, ← c115633, k2, ← c115634] at e_1895_9
  have hr := e_1895_9
  simp only [e_1895_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11777)) := by
  have c115635 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115636 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115637 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115638 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115639 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115640 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_10
  simp only [← c115635, ← c115636, ← c115637] at e_1895_10
  have e_1895_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_11
  simp only [← c115638, ← c115639, k2, ← c115640] at e_1895_11
  have hr := e_1895_11
  simp only [e_1895_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11778)) := by
  have c115641 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115642 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115643 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115644 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115645 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115646 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_12
  simp only [← c115641, ← c115642, ← c115643] at e_1895_12
  have e_1895_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_13
  simp only [← c115644, ← c115645, k2, ← c115646] at e_1895_13
  have hr := e_1895_13
  simp only [e_1895_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11779)) := by
  have c115647 := (publicBatchWrapper86_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115648 := (publicBatchWrapper86_copies3614 a h).1
  have c115649 := (publicBatchWrapper86_copies3614 a h).2.1
  have c115650 := (publicBatchWrapper86_copies3614 a h).2.2.1
  have c115651 := (publicBatchWrapper86_copies3614 a h).2.2.2.1
  have c115652 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_14
  simp only [← c115647, ← c115648, ← c115649] at e_1895_14
  have e_1895_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_15
  simp only [← c115650, ← c115651, k2, ← c115652] at e_1895_15
  have hr := e_1895_15
  simp only [e_1895_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11780)) := by
  have c115653 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.1
  have c115654 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.1
  have c115655 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.1
  have c115656 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.1
  have c115657 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.1
  have c115658 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_16
  simp only [← c115653, ← c115654, ← c115655] at e_1895_16
  have e_1895_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_17
  simp only [← c115656, ← c115657, k2, ← c115658] at e_1895_17
  have hr := e_1895_17
  simp only [e_1895_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1895 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11781)) := by
  have c115659 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115660 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115661 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115662 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115663 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115664 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1895_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_18
  simp only [← c115659, ← c115660, ← c115661] at e_1895_18
  have e_1895_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_19
  simp only [← c115662, ← c115663, k2, ← c115664] at e_1895_19
  have hr := e_1895_19
  simp only [e_1895_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11782)) := by
  have c115665 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115666 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115667 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115668 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115669 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115670 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_0
  simp only [← c115665, ← c115666, ← c115667] at e_1896_0
  have e_1896_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_1
  simp only [← c115668, ← c115669, k2, ← c115670] at e_1896_1
  have hr := e_1896_1
  simp only [e_1896_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11783)) := by
  have c115671 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115672 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115673 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115674 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115675 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115676 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_2
  simp only [← c115671, ← c115672, ← c115673] at e_1896_2
  have e_1896_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_3
  simp only [← c115674, ← c115675, k2, ← c115676] at e_1896_3
  have hr := e_1896_3
  simp only [e_1896_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11784)) := by
  have c115677 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115678 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115679 := (publicBatchWrapper86_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115680 := (publicBatchWrapper86_copies3615 a h).1
  have c115681 := (publicBatchWrapper86_copies3615 a h).2.1
  have c115682 := (publicBatchWrapper86_copies3615 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_4
  simp only [← c115677, ← c115678, ← c115679] at e_1896_4
  have e_1896_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_5
  simp only [← c115680, ← c115681, k2, ← c115682] at e_1896_5
  have hr := e_1896_5
  simp only [e_1896_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11785)) := by
  have c115683 := (publicBatchWrapper86_copies3615 a h).2.2.2.1
  have c115684 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.1
  have c115685 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.1
  have c115686 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.1
  have c115687 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.1
  have c115688 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_6
  simp only [← c115683, ← c115684, ← c115685] at e_1896_6
  have e_1896_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_7
  simp only [← c115686, ← c115687, k2, ← c115688] at e_1896_7
  have hr := e_1896_7
  simp only [e_1896_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11786)) := by
  have c115689 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.1
  have c115690 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115691 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115692 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115693 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115694 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_8
  simp only [← c115689, ← c115690, ← c115691] at e_1896_8
  have e_1896_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_9
  simp only [← c115692, ← c115693, k2, ← c115694] at e_1896_9
  have hr := e_1896_9
  simp only [e_1896_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11787)) := by
  have c115695 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115696 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115697 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115698 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115699 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115700 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_10
  simp only [← c115695, ← c115696, ← c115697] at e_1896_10
  have e_1896_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_11
  simp only [← c115698, ← c115699, k2, ← c115700] at e_1896_11
  have hr := e_1896_11
  simp only [e_1896_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11788)) := by
  have c115701 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115702 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115703 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115704 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115705 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115706 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_12
  simp only [← c115701, ← c115702, ← c115703] at e_1896_12
  have e_1896_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_13
  simp only [← c115704, ← c115705, k2, ← c115706] at e_1896_13
  have hr := e_1896_13
  simp only [e_1896_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11789)) := by
  have c115707 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115708 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115709 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115710 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115711 := (publicBatchWrapper86_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115712 := (publicBatchWrapper86_copies3616 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_14
  simp only [← c115707, ← c115708, ← c115709] at e_1896_14
  have e_1896_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_15
  simp only [← c115710, ← c115711, k2, ← c115712] at e_1896_15
  have hr := e_1896_15
  simp only [e_1896_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11790)) := by
  have c115713 := (publicBatchWrapper86_copies3616 a h).2.1
  have c115714 := (publicBatchWrapper86_copies3616 a h).2.2.1
  have c115715 := (publicBatchWrapper86_copies3616 a h).2.2.2.1
  have c115716 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.1
  have c115717 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.1
  have c115718 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_16
  simp only [← c115713, ← c115714, ← c115715] at e_1896_16
  have e_1896_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_17
  simp only [← c115716, ← c115717, k2, ← c115718] at e_1896_17
  have hr := e_1896_17
  simp only [e_1896_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1896 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11791)) := by
  have c115719 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.1
  have c115720 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.1
  have c115721 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.1
  have c115722 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115723 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115724 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1896_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_18
  simp only [← c115719, ← c115720, ← c115721] at e_1896_18
  have e_1896_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_19
  simp only [← c115722, ← c115723, k2, ← c115724] at e_1896_19
  have hr := e_1896_19
  simp only [e_1896_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 7) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11792)) := by
  have c115725 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115726 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115727 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115728 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115729 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115730 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_0
  simp only [← c115725, ← c115726, ← c115727] at e_1897_0
  have e_1897_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_1
  simp only [← c115728, ← c115729, k2, ← c115730] at e_1897_1
  have hr := e_1897_1
  simp only [e_1897_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 15) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11793)) := by
  have c115731 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115732 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115733 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115734 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115735 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115736 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_2
  simp only [← c115731, ← c115732, ← c115733] at e_1897_2
  have e_1897_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_3
  simp only [← c115734, ← c115735, k2, ← c115736] at e_1897_3
  have hr := e_1897_3
  simp only [e_1897_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 23) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11794)) := by
  have c115737 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115738 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115739 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115740 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115741 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115742 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_4
  simp only [← c115737, ← c115738, ← c115739] at e_1897_4
  have e_1897_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_5
  simp only [← c115740, ← c115741, k2, ← c115742] at e_1897_5
  have hr := e_1897_5
  simp only [e_1897_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 31) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11795)) := by
  have c115743 := (publicBatchWrapper86_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115744 := (publicBatchWrapper86_copies3617 a h).1
  have c115745 := (publicBatchWrapper86_copies3617 a h).2.1
  have c115746 := (publicBatchWrapper86_copies3617 a h).2.2.1
  have c115747 := (publicBatchWrapper86_copies3617 a h).2.2.2.1
  have c115748 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_6
  simp only [← c115743, ← c115744, ← c115745] at e_1897_6
  have e_1897_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_7
  simp only [← c115746, ← c115747, k2, ← c115748] at e_1897_7
  have hr := e_1897_7
  simp only [e_1897_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 39) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11796)) := by
  have c115749 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.1
  have c115750 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.1
  have c115751 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.1
  have c115752 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.1
  have c115753 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.1
  have c115754 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_8
  simp only [← c115749, ← c115750, ← c115751] at e_1897_8
  have e_1897_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_9
  simp only [← c115752, ← c115753, k2, ← c115754] at e_1897_9
  have hr := e_1897_9
  simp only [e_1897_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 47) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11797)) := by
  have c115755 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115756 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115757 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115758 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115759 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115760 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_10
  simp only [← c115755, ← c115756, ← c115757] at e_1897_10
  have e_1897_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_11
  simp only [← c115758, ← c115759, k2, ← c115760] at e_1897_11
  have hr := e_1897_11
  simp only [e_1897_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 55) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11798)) := by
  have c115761 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115762 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115763 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115764 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115765 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115766 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_12
  simp only [← c115761, ← c115762, ← c115763] at e_1897_12
  have e_1897_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_13
  simp only [← c115764, ← c115765, k2, ← c115766] at e_1897_13
  have hr := e_1897_13
  simp only [e_1897_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 63) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11799)) := by
  have c115767 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115768 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115769 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115770 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115771 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115772 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_14
  simp only [← c115767, ← c115768, ← c115769] at e_1897_14
  have e_1897_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_15
  simp only [← c115770, ← c115771, k2, ← c115772] at e_1897_15
  have hr := e_1897_15
  simp only [e_1897_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 71) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11800)) := by
  have c115773 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115774 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115775 := (publicBatchWrapper86_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115776 := (publicBatchWrapper86_copies3618 a h).1
  have c115777 := (publicBatchWrapper86_copies3618 a h).2.1
  have c115778 := (publicBatchWrapper86_copies3618 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_16
  simp only [← c115773, ← c115774, ← c115775] at e_1897_16
  have e_1897_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_17
  simp only [← c115776, ← c115777, k2, ← c115778] at e_1897_17
  have hr := e_1897_17
  simp only [e_1897_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1897 79) = bselect (a (.wire 58 19)) (a (.virt 200)) (a (.virt 11801)) := by
  have c115779 := (publicBatchWrapper86_copies3618 a h).2.2.2.1
  have c115780 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.1
  have c115781 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.1
  have c115782 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.1
  have c115783 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.1
  have c115784 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1897_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_18
  simp only [← c115779, ← c115780, ← c115781] at e_1897_18
  have e_1897_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_19
  simp only [← c115782, ← c115783, k2, ← c115784] at e_1897_19
  have hr := e_1897_19
  simp only [e_1897_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11970)) := by
  have c115785 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.1
  have c115786 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115787 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115788 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115789 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115790 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_0
  simp only [← c115785, ← c115786, ← c115787] at e_1898_0
  have e_1898_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_1
  simp only [← c115788, ← c115789, k2, ← c115790] at e_1898_1
  have hr := e_1898_1
  simp only [e_1898_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11971)) := by
  have c115791 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115792 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115793 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115794 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115795 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115796 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_2
  simp only [← c115791, ← c115792, ← c115793] at e_1898_2
  have e_1898_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_3
  simp only [← c115794, ← c115795, k2, ← c115796] at e_1898_3
  have hr := e_1898_3
  simp only [e_1898_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11972)) := by
  have c115797 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115798 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115799 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115800 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115801 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115802 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_4
  simp only [← c115797, ← c115798, ← c115799] at e_1898_4
  have e_1898_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_5
  simp only [← c115800, ← c115801, k2, ← c115802] at e_1898_5
  have hr := e_1898_5
  simp only [e_1898_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11973)) := by
  have c115803 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115804 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115805 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115806 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115807 := (publicBatchWrapper86_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115808 := (publicBatchWrapper86_copies3619 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_6
  simp only [← c115803, ← c115804, ← c115805] at e_1898_6
  have e_1898_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_7
  simp only [← c115806, ← c115807, k2, ← c115808] at e_1898_7
  have hr := e_1898_7
  simp only [e_1898_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11974)) := by
  have c115809 := (publicBatchWrapper86_copies3619 a h).2.1
  have c115810 := (publicBatchWrapper86_copies3619 a h).2.2.1
  have c115811 := (publicBatchWrapper86_copies3619 a h).2.2.2.1
  have c115812 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.1
  have c115813 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.1
  have c115814 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_8
  simp only [← c115809, ← c115810, ← c115811] at e_1898_8
  have e_1898_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_9
  simp only [← c115812, ← c115813, k2, ← c115814] at e_1898_9
  have hr := e_1898_9
  simp only [e_1898_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11975)) := by
  have c115815 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.1
  have c115816 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.1
  have c115817 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.1
  have c115818 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115819 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115820 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_10
  simp only [← c115815, ← c115816, ← c115817] at e_1898_10
  have e_1898_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_11
  simp only [← c115818, ← c115819, k2, ← c115820] at e_1898_11
  have hr := e_1898_11
  simp only [e_1898_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11976)) := by
  have c115821 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115822 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115823 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115824 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115825 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115826 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_12
  simp only [← c115821, ← c115822, ← c115823] at e_1898_12
  have e_1898_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_13
  simp only [← c115824, ← c115825, k2, ← c115826] at e_1898_13
  have hr := e_1898_13
  simp only [e_1898_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11977)) := by
  have c115827 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115828 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115829 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115830 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115831 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115832 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_14
  simp only [← c115827, ← c115828, ← c115829] at e_1898_14
  have e_1898_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_15
  simp only [← c115830, ← c115831, k2, ← c115832] at e_1898_15
  have hr := e_1898_15
  simp only [e_1898_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11978)) := by
  have c115833 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115834 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115835 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115836 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115837 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115838 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_16
  simp only [← c115833, ← c115834, ← c115835] at e_1898_16
  have e_1898_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_17
  simp only [← c115836, ← c115837, k2, ← c115838] at e_1898_17
  have hr := e_1898_17
  simp only [e_1898_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1898 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11979)) := by
  have c115839 := (publicBatchWrapper86_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115840 := (publicBatchWrapper86_copies3620 a h).1
  have c115841 := (publicBatchWrapper86_copies3620 a h).2.1
  have c115842 := (publicBatchWrapper86_copies3620 a h).2.2.1
  have c115843 := (publicBatchWrapper86_copies3620 a h).2.2.2.1
  have c115844 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1898_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_18
  simp only [← c115839, ← c115840, ← c115841] at e_1898_18
  have e_1898_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_19
  simp only [← c115842, ← c115843, k2, ← c115844] at e_1898_19
  have hr := e_1898_19
  simp only [e_1898_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11980)) := by
  have c115845 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.1
  have c115846 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.1
  have c115847 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.1
  have c115848 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.1
  have c115849 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.1
  have c115850 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_0
  simp only [← c115845, ← c115846, ← c115847] at e_1899_0
  have e_1899_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_1
  simp only [← c115848, ← c115849, k2, ← c115850] at e_1899_1
  have hr := e_1899_1
  simp only [e_1899_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11981)) := by
  have c115851 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115852 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115853 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115854 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115855 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115856 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_2
  simp only [← c115851, ← c115852, ← c115853] at e_1899_2
  have e_1899_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_3
  simp only [← c115854, ← c115855, k2, ← c115856] at e_1899_3
  have hr := e_1899_3
  simp only [e_1899_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11982)) := by
  have c115857 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115858 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115859 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115860 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115861 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115862 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_4
  simp only [← c115857, ← c115858, ← c115859] at e_1899_4
  have e_1899_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_5
  simp only [← c115860, ← c115861, k2, ← c115862] at e_1899_5
  have hr := e_1899_5
  simp only [e_1899_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11983)) := by
  have c115863 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115864 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115865 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115866 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115867 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115868 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_6
  simp only [← c115863, ← c115864, ← c115865] at e_1899_6
  have e_1899_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_7
  simp only [← c115866, ← c115867, k2, ← c115868] at e_1899_7
  have hr := e_1899_7
  simp only [e_1899_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11984)) := by
  have c115869 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115870 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115871 := (publicBatchWrapper86_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115872 := (publicBatchWrapper86_copies3621 a h).1
  have c115873 := (publicBatchWrapper86_copies3621 a h).2.1
  have c115874 := (publicBatchWrapper86_copies3621 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_8
  simp only [← c115869, ← c115870, ← c115871] at e_1899_8
  have e_1899_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_9
  simp only [← c115872, ← c115873, k2, ← c115874] at e_1899_9
  have hr := e_1899_9
  simp only [e_1899_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11985)) := by
  have c115875 := (publicBatchWrapper86_copies3621 a h).2.2.2.1
  have c115876 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.1
  have c115877 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.1
  have c115878 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.1
  have c115879 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.1
  have c115880 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_10
  simp only [← c115875, ← c115876, ← c115877] at e_1899_10
  have e_1899_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_11
  simp only [← c115878, ← c115879, k2, ← c115880] at e_1899_11
  have hr := e_1899_11
  simp only [e_1899_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11986)) := by
  have c115881 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.1
  have c115882 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115883 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115884 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115885 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115886 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_12
  simp only [← c115881, ← c115882, ← c115883] at e_1899_12
  have e_1899_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_13
  simp only [← c115884, ← c115885, k2, ← c115886] at e_1899_13
  have hr := e_1899_13
  simp only [e_1899_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11987)) := by
  have c115887 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115888 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115889 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115890 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115891 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115892 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_14
  simp only [← c115887, ← c115888, ← c115889] at e_1899_14
  have e_1899_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_15
  simp only [← c115890, ← c115891, k2, ← c115892] at e_1899_15
  have hr := e_1899_15
  simp only [e_1899_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11988)) := by
  have c115893 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115894 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115895 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115896 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115897 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115898 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_16
  simp only [← c115893, ← c115894, ← c115895] at e_1899_16
  have e_1899_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_17
  simp only [← c115896, ← c115897, k2, ← c115898] at e_1899_17
  have hr := e_1899_17
  simp only [e_1899_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1899 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11989)) := by
  have c115899 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115900 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115901 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115902 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115903 := (publicBatchWrapper86_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115904 := (publicBatchWrapper86_copies3622 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1899_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_18
  simp only [← c115899, ← c115900, ← c115901] at e_1899_18
  have e_1899_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_19
  simp only [← c115902, ← c115903, k2, ← c115904] at e_1899_19
  have hr := e_1899_19
  simp only [e_1899_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11990)) := by
  have c115905 := (publicBatchWrapper86_copies3622 a h).2.1
  have c115906 := (publicBatchWrapper86_copies3622 a h).2.2.1
  have c115907 := (publicBatchWrapper86_copies3622 a h).2.2.2.1
  have c115908 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.1
  have c115909 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.1
  have c115910 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_0
  simp only [← c115905, ← c115906, ← c115907] at e_1900_0
  have e_1900_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_1
  simp only [← c115908, ← c115909, k2, ← c115910] at e_1900_1
  have hr := e_1900_1
  simp only [e_1900_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11991)) := by
  have c115911 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.1
  have c115912 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.1
  have c115913 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.1
  have c115914 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115915 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115916 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_2
  simp only [← c115911, ← c115912, ← c115913] at e_1900_2
  have e_1900_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_3
  simp only [← c115914, ← c115915, k2, ← c115916] at e_1900_3
  have hr := e_1900_3
  simp only [e_1900_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 23) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11992)) := by
  have c115917 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115918 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115919 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115920 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115921 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115922 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_4
  simp only [← c115917, ← c115918, ← c115919] at e_1900_4
  have e_1900_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_5
  simp only [← c115920, ← c115921, k2, ← c115922] at e_1900_5
  have hr := e_1900_5
  simp only [e_1900_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 31) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11993)) := by
  have c115923 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115924 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115925 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115926 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115927 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115928 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_6
  simp only [← c115923, ← c115924, ← c115925] at e_1900_6
  have e_1900_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_7
  simp only [← c115926, ← c115927, k2, ← c115928] at e_1900_7
  have hr := e_1900_7
  simp only [e_1900_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 39) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11994)) := by
  have c115929 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115930 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115931 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115932 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115933 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115934 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_8
  simp only [← c115929, ← c115930, ← c115931] at e_1900_8
  have e_1900_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_9
  simp only [← c115932, ← c115933, k2, ← c115934] at e_1900_9
  have hr := e_1900_9
  simp only [e_1900_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 47) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11995)) := by
  have c115935 := (publicBatchWrapper86_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115936 := (publicBatchWrapper86_copies3623 a h).1
  have c115937 := (publicBatchWrapper86_copies3623 a h).2.1
  have c115938 := (publicBatchWrapper86_copies3623 a h).2.2.1
  have c115939 := (publicBatchWrapper86_copies3623 a h).2.2.2.1
  have c115940 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_10
  simp only [← c115935, ← c115936, ← c115937] at e_1900_10
  have e_1900_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_11
  simp only [← c115938, ← c115939, k2, ← c115940] at e_1900_11
  have hr := e_1900_11
  simp only [e_1900_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 55) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11996)) := by
  have c115941 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.1
  have c115942 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.1
  have c115943 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.1
  have c115944 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.1
  have c115945 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.1
  have c115946 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_12
  simp only [← c115941, ← c115942, ← c115943] at e_1900_12
  have e_1900_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_13
  simp only [← c115944, ← c115945, k2, ← c115946] at e_1900_13
  have hr := e_1900_13
  simp only [e_1900_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 63) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11997)) := by
  have c115947 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115948 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115949 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115950 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115951 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115952 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_14
  simp only [← c115947, ← c115948, ← c115949] at e_1900_14
  have e_1900_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_15
  simp only [← c115950, ← c115951, k2, ← c115952] at e_1900_15
  have hr := e_1900_15
  simp only [e_1900_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 71) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11998)) := by
  have c115953 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115954 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115955 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115956 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115957 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115958 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_16
  simp only [← c115953, ← c115954, ← c115955] at e_1900_16
  have e_1900_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_17
  simp only [← c115956, ← c115957, k2, ← c115958] at e_1900_17
  have hr := e_1900_17
  simp only [e_1900_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1900 79) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 11999)) := by
  have c115959 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115960 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115961 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115962 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115963 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115964 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1900_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_18
  simp only [← c115959, ← c115960, ← c115961] at e_1900_18
  have e_1900_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_19
  simp only [← c115962, ← c115963, k2, ← c115964] at e_1900_19
  have hr := e_1900_19
  simp only [e_1900_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 7) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 12000)) := by
  have c115965 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115966 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115967 := (publicBatchWrapper86_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115968 := (publicBatchWrapper86_copies3624 a h).1
  have c115969 := (publicBatchWrapper86_copies3624 a h).2.1
  have c115970 := (publicBatchWrapper86_copies3624 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_0
  simp only [← c115965, ← c115966, ← c115967] at e_1901_0
  have e_1901_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_1
  simp only [← c115968, ← c115969, k2, ← c115970] at e_1901_1
  have hr := e_1901_1
  simp only [e_1901_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 15) = bselect (a (.wire 58 63)) (a (.virt 200)) (a (.virt 12001)) := by
  have c115971 := (publicBatchWrapper86_copies3624 a h).2.2.2.1
  have c115972 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.1
  have c115973 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.1
  have c115974 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.1
  have c115975 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.1
  have c115976 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_2
  simp only [← c115971, ← c115972, ← c115973] at e_1901_2
  have e_1901_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_3
  simp only [← c115974, ← c115975, k2, ← c115976] at e_1901_3
  have hr := e_1901_3
  simp only [e_1901_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12170)) := by
  have c115977 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.1
  have c115978 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115979 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115980 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115981 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115982 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_4
  simp only [← c115977, ← c115978, ← c115979] at e_1901_4
  have e_1901_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_5
  simp only [← c115980, ← c115981, k2, ← c115982] at e_1901_5
  have hr := e_1901_5
  simp only [e_1901_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12171)) := by
  have c115983 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115984 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115985 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115986 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115987 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115988 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_6
  simp only [← c115983, ← c115984, ← c115985] at e_1901_6
  have e_1901_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_7
  simp only [← c115986, ← c115987, k2, ← c115988] at e_1901_7
  have hr := e_1901_7
  simp only [e_1901_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12172)) := by
  have c115989 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115990 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115991 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115992 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115993 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115994 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_8
  simp only [← c115989, ← c115990, ← c115991] at e_1901_8
  have e_1901_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_9
  simp only [← c115992, ← c115993, k2, ← c115994] at e_1901_9
  have hr := e_1901_9
  simp only [e_1901_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12173)) := by
  have c115995 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115996 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115997 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115998 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115999 := (publicBatchWrapper86_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116000 := (publicBatchWrapper86_copies3625 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_10
  simp only [← c115995, ← c115996, ← c115997] at e_1901_10
  have e_1901_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_11
  simp only [← c115998, ← c115999, k2, ← c116000] at e_1901_11
  have hr := e_1901_11
  simp only [e_1901_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12174)) := by
  have c116001 := (publicBatchWrapper86_copies3625 a h).2.1
  have c116002 := (publicBatchWrapper86_copies3625 a h).2.2.1
  have c116003 := (publicBatchWrapper86_copies3625 a h).2.2.2.1
  have c116004 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.1
  have c116005 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.1
  have c116006 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_12
  simp only [← c116001, ← c116002, ← c116003] at e_1901_12
  have e_1901_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_13
  simp only [← c116004, ← c116005, k2, ← c116006] at e_1901_13
  have hr := e_1901_13
  simp only [e_1901_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12175)) := by
  have c116007 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.1
  have c116008 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.1
  have c116009 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.1
  have c116010 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116011 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116012 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_14
  simp only [← c116007, ← c116008, ← c116009] at e_1901_14
  have e_1901_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_15
  simp only [← c116010, ← c116011, k2, ← c116012] at e_1901_15
  have hr := e_1901_15
  simp only [e_1901_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12176)) := by
  have c116013 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116014 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116015 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116016 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116017 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116018 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_16
  simp only [← c116013, ← c116014, ← c116015] at e_1901_16
  have e_1901_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_17
  simp only [← c116016, ← c116017, k2, ← c116018] at e_1901_17
  have hr := e_1901_17
  simp only [e_1901_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1901 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12177)) := by
  have c116019 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116020 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116021 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116022 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116023 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116024 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1901_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_18
  simp only [← c116019, ← c116020, ← c116021] at e_1901_18
  have e_1901_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_19
  simp only [← c116022, ← c116023, k2, ← c116024] at e_1901_19
  have hr := e_1901_19
  simp only [e_1901_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12178)) := by
  have c116025 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116026 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116027 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116028 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116029 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116030 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_0
  simp only [← c116025, ← c116026, ← c116027] at e_1902_0
  have e_1902_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_1
  simp only [← c116028, ← c116029, k2, ← c116030] at e_1902_1
  have hr := e_1902_1
  simp only [e_1902_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12179)) := by
  have c116031 := (publicBatchWrapper86_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116032 := (publicBatchWrapper86_copies3626 a h).1
  have c116033 := (publicBatchWrapper86_copies3626 a h).2.1
  have c116034 := (publicBatchWrapper86_copies3626 a h).2.2.1
  have c116035 := (publicBatchWrapper86_copies3626 a h).2.2.2.1
  have c116036 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_2
  simp only [← c116031, ← c116032, ← c116033] at e_1902_2
  have e_1902_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_3
  simp only [← c116034, ← c116035, k2, ← c116036] at e_1902_3
  have hr := e_1902_3
  simp only [e_1902_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12180)) := by
  have c116037 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.1
  have c116038 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.1
  have c116039 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.1
  have c116040 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.1
  have c116041 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.1
  have c116042 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_4
  simp only [← c116037, ← c116038, ← c116039] at e_1902_4
  have e_1902_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_5
  simp only [← c116040, ← c116041, k2, ← c116042] at e_1902_5
  have hr := e_1902_5
  simp only [e_1902_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12181)) := by
  have c116043 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116044 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116045 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116046 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116047 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116048 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_6
  simp only [← c116043, ← c116044, ← c116045] at e_1902_6
  have e_1902_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_7
  simp only [← c116046, ← c116047, k2, ← c116048] at e_1902_7
  have hr := e_1902_7
  simp only [e_1902_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12182)) := by
  have c116049 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116050 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116051 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116052 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116053 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116054 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_8
  simp only [← c116049, ← c116050, ← c116051] at e_1902_8
  have e_1902_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_9
  simp only [← c116052, ← c116053, k2, ← c116054] at e_1902_9
  have hr := e_1902_9
  simp only [e_1902_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12183)) := by
  have c116055 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116056 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116057 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116058 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116059 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116060 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_10
  simp only [← c116055, ← c116056, ← c116057] at e_1902_10
  have e_1902_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_11
  simp only [← c116058, ← c116059, k2, ← c116060] at e_1902_11
  have hr := e_1902_11
  simp only [e_1902_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12184)) := by
  have c116061 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116062 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116063 := (publicBatchWrapper86_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116064 := (publicBatchWrapper86_copies3627 a h).1
  have c116065 := (publicBatchWrapper86_copies3627 a h).2.1
  have c116066 := (publicBatchWrapper86_copies3627 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_12
  simp only [← c116061, ← c116062, ← c116063] at e_1902_12
  have e_1902_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_13
  simp only [← c116064, ← c116065, k2, ← c116066] at e_1902_13
  have hr := e_1902_13
  simp only [e_1902_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12185)) := by
  have c116067 := (publicBatchWrapper86_copies3627 a h).2.2.2.1
  have c116068 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.1
  have c116069 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.1
  have c116070 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.1
  have c116071 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.1
  have c116072 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_14
  simp only [← c116067, ← c116068, ← c116069] at e_1902_14
  have e_1902_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_15
  simp only [← c116070, ← c116071, k2, ← c116072] at e_1902_15
  have hr := e_1902_15
  simp only [e_1902_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12186)) := by
  have c116073 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.1
  have c116074 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116075 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116076 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116077 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116078 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_16
  simp only [← c116073, ← c116074, ← c116075] at e_1902_16
  have e_1902_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_17
  simp only [← c116076, ← c116077, k2, ← c116078] at e_1902_17
  have hr := e_1902_17
  simp only [e_1902_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1902 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12187)) := by
  have c116079 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116080 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116081 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116082 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116083 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116084 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1902_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_18
  simp only [← c116079, ← c116080, ← c116081] at e_1902_18
  have e_1902_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_19
  simp only [← c116082, ← c116083, k2, ← c116084] at e_1902_19
  have hr := e_1902_19
  simp only [e_1902_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12188)) := by
  have c116085 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116086 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116087 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116088 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116089 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116090 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_0
  simp only [← c116085, ← c116086, ← c116087] at e_1903_0
  have e_1903_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_1
  simp only [← c116088, ← c116089, k2, ← c116090] at e_1903_1
  have hr := e_1903_1
  simp only [e_1903_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12189)) := by
  have c116091 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116092 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116093 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116094 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116095 := (publicBatchWrapper86_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116096 := (publicBatchWrapper86_copies3628 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_2
  simp only [← c116091, ← c116092, ← c116093] at e_1903_2
  have e_1903_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_3
  simp only [← c116094, ← c116095, k2, ← c116096] at e_1903_3
  have hr := e_1903_3
  simp only [e_1903_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12190)) := by
  have c116097 := (publicBatchWrapper86_copies3628 a h).2.1
  have c116098 := (publicBatchWrapper86_copies3628 a h).2.2.1
  have c116099 := (publicBatchWrapper86_copies3628 a h).2.2.2.1
  have c116100 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.1
  have c116101 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.1
  have c116102 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_4
  simp only [← c116097, ← c116098, ← c116099] at e_1903_4
  have e_1903_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_5
  simp only [← c116100, ← c116101, k2, ← c116102] at e_1903_5
  have hr := e_1903_5
  simp only [e_1903_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12191)) := by
  have c116103 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.1
  have c116104 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.1
  have c116105 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.1
  have c116106 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116107 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116108 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_6
  simp only [← c116103, ← c116104, ← c116105] at e_1903_6
  have e_1903_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_7
  simp only [← c116106, ← c116107, k2, ← c116108] at e_1903_7
  have hr := e_1903_7
  simp only [e_1903_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 39) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12192)) := by
  have c116109 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116110 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116111 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116112 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116113 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116114 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_8
  simp only [← c116109, ← c116110, ← c116111] at e_1903_8
  have e_1903_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_9
  simp only [← c116112, ← c116113, k2, ← c116114] at e_1903_9
  have hr := e_1903_9
  simp only [e_1903_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 47) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12193)) := by
  have c116115 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116116 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116117 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116118 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116119 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116120 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_10
  simp only [← c116115, ← c116116, ← c116117] at e_1903_10
  have e_1903_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_11
  simp only [← c116118, ← c116119, k2, ← c116120] at e_1903_11
  have hr := e_1903_11
  simp only [e_1903_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 55) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12194)) := by
  have c116121 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116122 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116123 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116124 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116125 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116126 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_12
  simp only [← c116121, ← c116122, ← c116123] at e_1903_12
  have e_1903_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_13
  simp only [← c116124, ← c116125, k2, ← c116126] at e_1903_13
  have hr := e_1903_13
  simp only [e_1903_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 63) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12195)) := by
  have c116127 := (publicBatchWrapper86_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116128 := (publicBatchWrapper86_copies3629 a h).1
  have c116129 := (publicBatchWrapper86_copies3629 a h).2.1
  have c116130 := (publicBatchWrapper86_copies3629 a h).2.2.1
  have c116131 := (publicBatchWrapper86_copies3629 a h).2.2.2.1
  have c116132 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_14
  simp only [← c116127, ← c116128, ← c116129] at e_1903_14
  have e_1903_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_15
  simp only [← c116130, ← c116131, k2, ← c116132] at e_1903_15
  have hr := e_1903_15
  simp only [e_1903_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 71) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12196)) := by
  have c116133 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.1
  have c116134 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.1
  have c116135 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.1
  have c116136 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.1
  have c116137 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.1
  have c116138 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_16
  simp only [← c116133, ← c116134, ← c116135] at e_1903_16
  have e_1903_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_17
  simp only [← c116136, ← c116137, k2, ← c116138] at e_1903_17
  have hr := e_1903_17
  simp only [e_1903_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1903 79) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12197)) := by
  have c116139 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116140 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116141 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116142 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116143 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116144 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1903_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_18
  simp only [← c116139, ← c116140, ← c116141] at e_1903_18
  have e_1903_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_19
  simp only [← c116142, ← c116143, k2, ← c116144] at e_1903_19
  have hr := e_1903_19
  simp only [e_1903_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 7) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12198)) := by
  have c116145 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116146 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116147 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116148 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116149 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116150 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_0
  simp only [← c116145, ← c116146, ← c116147] at e_1904_0
  have e_1904_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_1
  simp only [← c116148, ← c116149, k2, ← c116150] at e_1904_1
  have hr := e_1904_1
  simp only [e_1904_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 15) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12199)) := by
  have c116151 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116152 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116153 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116154 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116155 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116156 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_2
  simp only [← c116151, ← c116152, ← c116153] at e_1904_2
  have e_1904_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_3
  simp only [← c116154, ← c116155, k2, ← c116156] at e_1904_3
  have hr := e_1904_3
  simp only [e_1904_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 23) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12200)) := by
  have c116157 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116158 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116159 := (publicBatchWrapper86_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116160 := (publicBatchWrapper86_copies3630 a h).1
  have c116161 := (publicBatchWrapper86_copies3630 a h).2.1
  have c116162 := (publicBatchWrapper86_copies3630 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_4
  simp only [← c116157, ← c116158, ← c116159] at e_1904_4
  have e_1904_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_5
  simp only [← c116160, ← c116161, k2, ← c116162] at e_1904_5
  have hr := e_1904_5
  simp only [e_1904_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 31) = bselect (a (.wire 60 27)) (a (.virt 200)) (a (.virt 12201)) := by
  have c116163 := (publicBatchWrapper86_copies3630 a h).2.2.2.1
  have c116164 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.1
  have c116165 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.1
  have c116166 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.1
  have c116167 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.1
  have c116168 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_6
  simp only [← c116163, ← c116164, ← c116165] at e_1904_6
  have e_1904_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_7
  simp only [← c116166, ← c116167, k2, ← c116168] at e_1904_7
  have hr := e_1904_7
  simp only [e_1904_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12370)) := by
  have c116169 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.1
  have c116170 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116171 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116172 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116173 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116174 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_8
  simp only [← c116169, ← c116170, ← c116171] at e_1904_8
  have e_1904_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_9
  simp only [← c116172, ← c116173, k2, ← c116174] at e_1904_9
  have hr := e_1904_9
  simp only [e_1904_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12371)) := by
  have c116175 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116176 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116177 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116178 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116179 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116180 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_10
  simp only [← c116175, ← c116176, ← c116177] at e_1904_10
  have e_1904_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_11
  simp only [← c116178, ← c116179, k2, ← c116180] at e_1904_11
  have hr := e_1904_11
  simp only [e_1904_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12372)) := by
  have c116181 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116182 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116183 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116184 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116185 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116186 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_12
  simp only [← c116181, ← c116182, ← c116183] at e_1904_12
  have e_1904_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_13
  simp only [← c116184, ← c116185, k2, ← c116186] at e_1904_13
  have hr := e_1904_13
  simp only [e_1904_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12373)) := by
  have c116187 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116188 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116189 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116190 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116191 := (publicBatchWrapper86_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116192 := (publicBatchWrapper86_copies3631 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_14
  simp only [← c116187, ← c116188, ← c116189] at e_1904_14
  have e_1904_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_15
  simp only [← c116190, ← c116191, k2, ← c116192] at e_1904_15
  have hr := e_1904_15
  simp only [e_1904_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12374)) := by
  have c116193 := (publicBatchWrapper86_copies3631 a h).2.1
  have c116194 := (publicBatchWrapper86_copies3631 a h).2.2.1
  have c116195 := (publicBatchWrapper86_copies3631 a h).2.2.2.1
  have c116196 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.1
  have c116197 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.1
  have c116198 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_16
  simp only [← c116193, ← c116194, ← c116195] at e_1904_16
  have e_1904_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_17
  simp only [← c116196, ← c116197, k2, ← c116198] at e_1904_17
  have hr := e_1904_17
  simp only [e_1904_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1904 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12375)) := by
  have c116199 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.1
  have c116200 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.1
  have c116201 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.1
  have c116202 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116203 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116204 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1904_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_18
  simp only [← c116199, ← c116200, ← c116201] at e_1904_18
  have e_1904_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_19
  simp only [← c116202, ← c116203, k2, ← c116204] at e_1904_19
  have hr := e_1904_19
  simp only [e_1904_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12376)) := by
  have c116205 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116206 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116207 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116208 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116209 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116210 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_0
  simp only [← c116205, ← c116206, ← c116207] at e_1905_0
  have e_1905_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_1
  simp only [← c116208, ← c116209, k2, ← c116210] at e_1905_1
  have hr := e_1905_1
  simp only [e_1905_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12377)) := by
  have c116211 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116212 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116213 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116214 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116215 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116216 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_2
  simp only [← c116211, ← c116212, ← c116213] at e_1905_2
  have e_1905_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_3
  simp only [← c116214, ← c116215, k2, ← c116216] at e_1905_3
  have hr := e_1905_3
  simp only [e_1905_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12378)) := by
  have c116217 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116218 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116219 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116220 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116221 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116222 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_4
  simp only [← c116217, ← c116218, ← c116219] at e_1905_4
  have e_1905_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_5
  simp only [← c116220, ← c116221, k2, ← c116222] at e_1905_5
  have hr := e_1905_5
  simp only [e_1905_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12379)) := by
  have c116223 := (publicBatchWrapper86_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116224 := (publicBatchWrapper86_copies3632 a h).1
  have c116225 := (publicBatchWrapper86_copies3632 a h).2.1
  have c116226 := (publicBatchWrapper86_copies3632 a h).2.2.1
  have c116227 := (publicBatchWrapper86_copies3632 a h).2.2.2.1
  have c116228 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_6
  simp only [← c116223, ← c116224, ← c116225] at e_1905_6
  have e_1905_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_7
  simp only [← c116226, ← c116227, k2, ← c116228] at e_1905_7
  have hr := e_1905_7
  simp only [e_1905_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12380)) := by
  have c116229 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.1
  have c116230 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.1
  have c116231 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.1
  have c116232 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.1
  have c116233 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.1
  have c116234 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_8
  simp only [← c116229, ← c116230, ← c116231] at e_1905_8
  have e_1905_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_9
  simp only [← c116232, ← c116233, k2, ← c116234] at e_1905_9
  have hr := e_1905_9
  simp only [e_1905_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12381)) := by
  have c116235 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116236 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116237 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116238 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116239 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116240 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_10
  simp only [← c116235, ← c116236, ← c116237] at e_1905_10
  have e_1905_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_11
  simp only [← c116238, ← c116239, k2, ← c116240] at e_1905_11
  have hr := e_1905_11
  simp only [e_1905_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12382)) := by
  have c116241 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116242 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116243 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116244 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116245 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116246 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_12
  simp only [← c116241, ← c116242, ← c116243] at e_1905_12
  have e_1905_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_13
  simp only [← c116244, ← c116245, k2, ← c116246] at e_1905_13
  have hr := e_1905_13
  simp only [e_1905_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12383)) := by
  have c116247 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116248 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116249 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116250 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116251 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116252 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_14
  simp only [← c116247, ← c116248, ← c116249] at e_1905_14
  have e_1905_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_15
  simp only [← c116250, ← c116251, k2, ← c116252] at e_1905_15
  have hr := e_1905_15
  simp only [e_1905_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12384)) := by
  have c116253 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116254 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116255 := (publicBatchWrapper86_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116256 := (publicBatchWrapper86_copies3633 a h).1
  have c116257 := (publicBatchWrapper86_copies3633 a h).2.1
  have c116258 := (publicBatchWrapper86_copies3633 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_16
  simp only [← c116253, ← c116254, ← c116255] at e_1905_16
  have e_1905_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_17
  simp only [← c116256, ← c116257, k2, ← c116258] at e_1905_17
  have hr := e_1905_17
  simp only [e_1905_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1905 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12385)) := by
  have c116259 := (publicBatchWrapper86_copies3633 a h).2.2.2.1
  have c116260 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.1
  have c116261 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.1
  have c116262 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.1
  have c116263 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.1
  have c116264 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1905_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_18
  simp only [← c116259, ← c116260, ← c116261] at e_1905_18
  have e_1905_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_19
  simp only [← c116262, ← c116263, k2, ← c116264] at e_1905_19
  have hr := e_1905_19
  simp only [e_1905_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12386)) := by
  have c116265 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.1
  have c116266 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116267 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116268 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116269 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116270 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_0
  simp only [← c116265, ← c116266, ← c116267] at e_1906_0
  have e_1906_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_1
  simp only [← c116268, ← c116269, k2, ← c116270] at e_1906_1
  have hr := e_1906_1
  simp only [e_1906_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12387)) := by
  have c116271 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116272 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116273 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116274 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116275 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116276 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_2
  simp only [← c116271, ← c116272, ← c116273] at e_1906_2
  have e_1906_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_3
  simp only [← c116274, ← c116275, k2, ← c116276] at e_1906_3
  have hr := e_1906_3
  simp only [e_1906_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12388)) := by
  have c116277 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116278 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116279 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116280 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116281 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116282 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_4
  simp only [← c116277, ← c116278, ← c116279] at e_1906_4
  have e_1906_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_5
  simp only [← c116280, ← c116281, k2, ← c116282] at e_1906_5
  have hr := e_1906_5
  simp only [e_1906_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12389)) := by
  have c116283 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116284 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116285 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116286 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116287 := (publicBatchWrapper86_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116288 := (publicBatchWrapper86_copies3634 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_6
  simp only [← c116283, ← c116284, ← c116285] at e_1906_6
  have e_1906_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_7
  simp only [← c116286, ← c116287, k2, ← c116288] at e_1906_7
  have hr := e_1906_7
  simp only [e_1906_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12390)) := by
  have c116289 := (publicBatchWrapper86_copies3634 a h).2.1
  have c116290 := (publicBatchWrapper86_copies3634 a h).2.2.1
  have c116291 := (publicBatchWrapper86_copies3634 a h).2.2.2.1
  have c116292 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.1
  have c116293 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.1
  have c116294 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_8
  simp only [← c116289, ← c116290, ← c116291] at e_1906_8
  have e_1906_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_9
  simp only [← c116292, ← c116293, k2, ← c116294] at e_1906_9
  have hr := e_1906_9
  simp only [e_1906_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12391)) := by
  have c116295 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.1
  have c116296 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.1
  have c116297 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.1
  have c116298 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116299 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116300 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_10
  simp only [← c116295, ← c116296, ← c116297] at e_1906_10
  have e_1906_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_11
  simp only [← c116298, ← c116299, k2, ← c116300] at e_1906_11
  have hr := e_1906_11
  simp only [e_1906_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 55) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12392)) := by
  have c116301 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116302 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116303 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116304 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116305 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116306 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_12
  simp only [← c116301, ← c116302, ← c116303] at e_1906_12
  have e_1906_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_13
  simp only [← c116304, ← c116305, k2, ← c116306] at e_1906_13
  have hr := e_1906_13
  simp only [e_1906_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 63) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12393)) := by
  have c116307 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116308 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116309 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116310 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116311 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116312 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_14
  simp only [← c116307, ← c116308, ← c116309] at e_1906_14
  have e_1906_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_15
  simp only [← c116310, ← c116311, k2, ← c116312] at e_1906_15
  have hr := e_1906_15
  simp only [e_1906_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 71) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12394)) := by
  have c116313 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116314 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116315 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116316 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116317 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116318 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_16
  simp only [← c116313, ← c116314, ← c116315] at e_1906_16
  have e_1906_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_17
  simp only [← c116316, ← c116317, k2, ← c116318] at e_1906_17
  have hr := e_1906_17
  simp only [e_1906_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1906 79) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12395)) := by
  have c116319 := (publicBatchWrapper86_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116320 := (publicBatchWrapper86_copies3635 a h).1
  have c116321 := (publicBatchWrapper86_copies3635 a h).2.1
  have c116322 := (publicBatchWrapper86_copies3635 a h).2.2.1
  have c116323 := (publicBatchWrapper86_copies3635 a h).2.2.2.1
  have c116324 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1906_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_18
  simp only [← c116319, ← c116320, ← c116321] at e_1906_18
  have e_1906_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_19
  simp only [← c116322, ← c116323, k2, ← c116324] at e_1906_19
  have hr := e_1906_19
  simp only [e_1906_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 7) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12396)) := by
  have c116325 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.1
  have c116326 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.1
  have c116327 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.1
  have c116328 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.1
  have c116329 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.1
  have c116330 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_0
  simp only [← c116325, ← c116326, ← c116327] at e_1907_0
  have e_1907_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_1
  simp only [← c116328, ← c116329, k2, ← c116330] at e_1907_1
  have hr := e_1907_1
  simp only [e_1907_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 15) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12397)) := by
  have c116331 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116332 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116333 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116334 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116335 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116336 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_2
  simp only [← c116331, ← c116332, ← c116333] at e_1907_2
  have e_1907_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_3
  simp only [← c116334, ← c116335, k2, ← c116336] at e_1907_3
  have hr := e_1907_3
  simp only [e_1907_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 23) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12398)) := by
  have c116337 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116338 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116339 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116340 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116341 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116342 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_4
  simp only [← c116337, ← c116338, ← c116339] at e_1907_4
  have e_1907_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_5
  simp only [← c116340, ← c116341, k2, ← c116342] at e_1907_5
  have hr := e_1907_5
  simp only [e_1907_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 31) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12399)) := by
  have c116343 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116344 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116345 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116346 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116347 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116348 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_6
  simp only [← c116343, ← c116344, ← c116345] at e_1907_6
  have e_1907_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_7
  simp only [← c116346, ← c116347, k2, ← c116348] at e_1907_7
  have hr := e_1907_7
  simp only [e_1907_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 39) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12400)) := by
  have c116349 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116350 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116351 := (publicBatchWrapper86_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116352 := (publicBatchWrapper86_copies3636 a h).1
  have c116353 := (publicBatchWrapper86_copies3636 a h).2.1
  have c116354 := (publicBatchWrapper86_copies3636 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_8
  simp only [← c116349, ← c116350, ← c116351] at e_1907_8
  have e_1907_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_9
  simp only [← c116352, ← c116353, k2, ← c116354] at e_1907_9
  have hr := e_1907_9
  simp only [e_1907_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 47) = bselect (a (.wire 60 71)) (a (.virt 200)) (a (.virt 12401)) := by
  have c116355 := (publicBatchWrapper86_copies3636 a h).2.2.2.1
  have c116356 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.1
  have c116357 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.1
  have c116358 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.1
  have c116359 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.1
  have c116360 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_10
  simp only [← c116355, ← c116356, ← c116357] at e_1907_10
  have e_1907_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_11
  simp only [← c116358, ← c116359, k2, ← c116360] at e_1907_11
  have hr := e_1907_11
  simp only [e_1907_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12570)) := by
  have c116361 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.1
  have c116362 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116363 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116364 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116365 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116366 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_12
  simp only [← c116361, ← c116362, ← c116363] at e_1907_12
  have e_1907_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_13
  simp only [← c116364, ← c116365, k2, ← c116366] at e_1907_13
  have hr := e_1907_13
  simp only [e_1907_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12571)) := by
  have c116367 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116368 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116369 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116370 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116371 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116372 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_14
  simp only [← c116367, ← c116368, ← c116369] at e_1907_14
  have e_1907_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_15
  simp only [← c116370, ← c116371, k2, ← c116372] at e_1907_15
  have hr := e_1907_15
  simp only [e_1907_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12572)) := by
  have c116373 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116374 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116375 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116376 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116377 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116378 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_16
  simp only [← c116373, ← c116374, ← c116375] at e_1907_16
  have e_1907_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_17
  simp only [← c116376, ← c116377, k2, ← c116378] at e_1907_17
  have hr := e_1907_17
  simp only [e_1907_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1907 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12573)) := by
  have c116379 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116380 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116381 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116382 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116383 := (publicBatchWrapper86_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116384 := (publicBatchWrapper86_copies3637 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1907_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_18
  simp only [← c116379, ← c116380, ← c116381] at e_1907_18
  have e_1907_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_19
  simp only [← c116382, ← c116383, k2, ← c116384] at e_1907_19
  have hr := e_1907_19
  simp only [e_1907_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12574)) := by
  have c116385 := (publicBatchWrapper86_copies3637 a h).2.1
  have c116386 := (publicBatchWrapper86_copies3637 a h).2.2.1
  have c116387 := (publicBatchWrapper86_copies3637 a h).2.2.2.1
  have c116388 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.1
  have c116389 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.1
  have c116390 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_0
  simp only [← c116385, ← c116386, ← c116387] at e_1908_0
  have e_1908_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_1
  simp only [← c116388, ← c116389, k2, ← c116390] at e_1908_1
  have hr := e_1908_1
  simp only [e_1908_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12575)) := by
  have c116391 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.1
  have c116392 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.1
  have c116393 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.1
  have c116394 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116395 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116396 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_2
  simp only [← c116391, ← c116392, ← c116393] at e_1908_2
  have e_1908_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_3
  simp only [← c116394, ← c116395, k2, ← c116396] at e_1908_3
  have hr := e_1908_3
  simp only [e_1908_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12576)) := by
  have c116397 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116398 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116399 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116400 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116401 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116402 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_4
  simp only [← c116397, ← c116398, ← c116399] at e_1908_4
  have e_1908_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_5
  simp only [← c116400, ← c116401, k2, ← c116402] at e_1908_5
  have hr := e_1908_5
  simp only [e_1908_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12577)) := by
  have c116403 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116404 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116405 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116406 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116407 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116408 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_6
  simp only [← c116403, ← c116404, ← c116405] at e_1908_6
  have e_1908_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_7
  simp only [← c116406, ← c116407, k2, ← c116408] at e_1908_7
  have hr := e_1908_7
  simp only [e_1908_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12578)) := by
  have c116409 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116410 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116411 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116412 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116413 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116414 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_8
  simp only [← c116409, ← c116410, ← c116411] at e_1908_8
  have e_1908_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_9
  simp only [← c116412, ← c116413, k2, ← c116414] at e_1908_9
  have hr := e_1908_9
  simp only [e_1908_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12579)) := by
  have c116415 := (publicBatchWrapper86_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116416 := (publicBatchWrapper86_copies3638 a h).1
  have c116417 := (publicBatchWrapper86_copies3638 a h).2.1
  have c116418 := (publicBatchWrapper86_copies3638 a h).2.2.1
  have c116419 := (publicBatchWrapper86_copies3638 a h).2.2.2.1
  have c116420 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_10
  simp only [← c116415, ← c116416, ← c116417] at e_1908_10
  have e_1908_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_11
  simp only [← c116418, ← c116419, k2, ← c116420] at e_1908_11
  have hr := e_1908_11
  simp only [e_1908_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12580)) := by
  have c116421 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.1
  have c116422 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.1
  have c116423 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.1
  have c116424 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.1
  have c116425 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.1
  have c116426 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_12
  simp only [← c116421, ← c116422, ← c116423] at e_1908_12
  have e_1908_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_13
  simp only [← c116424, ← c116425, k2, ← c116426] at e_1908_13
  have hr := e_1908_13
  simp only [e_1908_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12581)) := by
  have c116427 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116428 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116429 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116430 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116431 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116432 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_14
  simp only [← c116427, ← c116428, ← c116429] at e_1908_14
  have e_1908_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_15
  simp only [← c116430, ← c116431, k2, ← c116432] at e_1908_15
  have hr := e_1908_15
  simp only [e_1908_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12582)) := by
  have c116433 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116434 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116435 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116436 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116437 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116438 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_16
  simp only [← c116433, ← c116434, ← c116435] at e_1908_16
  have e_1908_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_17
  simp only [← c116436, ← c116437, k2, ← c116438] at e_1908_17
  have hr := e_1908_17
  simp only [e_1908_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1908 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12583)) := by
  have c116439 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116440 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116441 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116442 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116443 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116444 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1908_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_18
  simp only [← c116439, ← c116440, ← c116441] at e_1908_18
  have e_1908_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_19
  simp only [← c116442, ← c116443, k2, ← c116444] at e_1908_19
  have hr := e_1908_19
  simp only [e_1908_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12584)) := by
  have c116445 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116446 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116447 := (publicBatchWrapper86_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116448 := (publicBatchWrapper86_copies3639 a h).1
  have c116449 := (publicBatchWrapper86_copies3639 a h).2.1
  have c116450 := (publicBatchWrapper86_copies3639 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_0
  simp only [← c116445, ← c116446, ← c116447] at e_1909_0
  have e_1909_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_1
  simp only [← c116448, ← c116449, k2, ← c116450] at e_1909_1
  have hr := e_1909_1
  simp only [e_1909_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12585)) := by
  have c116451 := (publicBatchWrapper86_copies3639 a h).2.2.2.1
  have c116452 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.1
  have c116453 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.1
  have c116454 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.1
  have c116455 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.1
  have c116456 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_2
  simp only [← c116451, ← c116452, ← c116453] at e_1909_2
  have e_1909_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_3
  simp only [← c116454, ← c116455, k2, ← c116456] at e_1909_3
  have hr := e_1909_3
  simp only [e_1909_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12586)) := by
  have c116457 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.1
  have c116458 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116459 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116460 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116461 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116462 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_4
  simp only [← c116457, ← c116458, ← c116459] at e_1909_4
  have e_1909_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_5
  simp only [← c116460, ← c116461, k2, ← c116462] at e_1909_5
  have hr := e_1909_5
  simp only [e_1909_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12587)) := by
  have c116463 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116464 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116465 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116466 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116467 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116468 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_6
  simp only [← c116463, ← c116464, ← c116465] at e_1909_6
  have e_1909_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_7
  simp only [← c116466, ← c116467, k2, ← c116468] at e_1909_7
  have hr := e_1909_7
  simp only [e_1909_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12588)) := by
  have c116469 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116470 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116471 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116472 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116473 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116474 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_8
  simp only [← c116469, ← c116470, ← c116471] at e_1909_8
  have e_1909_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_9
  simp only [← c116472, ← c116473, k2, ← c116474] at e_1909_9
  have hr := e_1909_9
  simp only [e_1909_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12589)) := by
  have c116475 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116476 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116477 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116478 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116479 := (publicBatchWrapper86_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116480 := (publicBatchWrapper86_copies3640 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_10
  simp only [← c116475, ← c116476, ← c116477] at e_1909_10
  have e_1909_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_11
  simp only [← c116478, ← c116479, k2, ← c116480] at e_1909_11
  have hr := e_1909_11
  simp only [e_1909_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12590)) := by
  have c116481 := (publicBatchWrapper86_copies3640 a h).2.1
  have c116482 := (publicBatchWrapper86_copies3640 a h).2.2.1
  have c116483 := (publicBatchWrapper86_copies3640 a h).2.2.2.1
  have c116484 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.1
  have c116485 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.1
  have c116486 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_12
  simp only [← c116481, ← c116482, ← c116483] at e_1909_12
  have e_1909_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_13
  simp only [← c116484, ← c116485, k2, ← c116486] at e_1909_13
  have hr := e_1909_13
  simp only [e_1909_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12591)) := by
  have c116487 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.1
  have c116488 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.1
  have c116489 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.1
  have c116490 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116491 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116492 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_14
  simp only [← c116487, ← c116488, ← c116489] at e_1909_14
  have e_1909_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_15
  simp only [← c116490, ← c116491, k2, ← c116492] at e_1909_15
  have hr := e_1909_15
  simp only [e_1909_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 71) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12592)) := by
  have c116493 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116494 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116495 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116496 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116497 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116498 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_16
  simp only [← c116493, ← c116494, ← c116495] at e_1909_16
  have e_1909_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_17
  simp only [← c116496, ← c116497, k2, ← c116498] at e_1909_17
  have hr := e_1909_17
  simp only [e_1909_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1909 79) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12593)) := by
  have c116499 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116500 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116501 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116502 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116503 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116504 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1909_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_18
  simp only [← c116499, ← c116500, ← c116501] at e_1909_18
  have e_1909_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_19
  simp only [← c116502, ← c116503, k2, ← c116504] at e_1909_19
  have hr := e_1909_19
  simp only [e_1909_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 7) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12594)) := by
  have c116505 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116506 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116507 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116508 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116509 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116510 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_0
  simp only [← c116505, ← c116506, ← c116507] at e_1910_0
  have e_1910_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_1
  simp only [← c116508, ← c116509, k2, ← c116510] at e_1910_1
  have hr := e_1910_1
  simp only [e_1910_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 15) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12595)) := by
  have c116511 := (publicBatchWrapper86_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116512 := (publicBatchWrapper86_copies3641 a h).1
  have c116513 := (publicBatchWrapper86_copies3641 a h).2.1
  have c116514 := (publicBatchWrapper86_copies3641 a h).2.2.1
  have c116515 := (publicBatchWrapper86_copies3641 a h).2.2.2.1
  have c116516 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_2
  simp only [← c116511, ← c116512, ← c116513] at e_1910_2
  have e_1910_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_3
  simp only [← c116514, ← c116515, k2, ← c116516] at e_1910_3
  have hr := e_1910_3
  simp only [e_1910_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 23) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12596)) := by
  have c116517 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.1
  have c116518 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.1
  have c116519 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.1
  have c116520 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.1
  have c116521 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.1
  have c116522 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_4
  simp only [← c116517, ← c116518, ← c116519] at e_1910_4
  have e_1910_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_5
  simp only [← c116520, ← c116521, k2, ← c116522] at e_1910_5
  have hr := e_1910_5
  simp only [e_1910_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 31) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12597)) := by
  have c116523 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116524 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116525 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116526 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116527 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116528 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_6
  simp only [← c116523, ← c116524, ← c116525] at e_1910_6
  have e_1910_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_7
  simp only [← c116526, ← c116527, k2, ← c116528] at e_1910_7
  have hr := e_1910_7
  simp only [e_1910_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 39) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12598)) := by
  have c116529 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116530 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116531 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116532 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116533 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116534 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_8
  simp only [← c116529, ← c116530, ← c116531] at e_1910_8
  have e_1910_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_9
  simp only [← c116532, ← c116533, k2, ← c116534] at e_1910_9
  have hr := e_1910_9
  simp only [e_1910_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 47) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12599)) := by
  have c116535 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116536 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116537 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116538 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116539 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116540 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_10
  simp only [← c116535, ← c116536, ← c116537] at e_1910_10
  have e_1910_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_11
  simp only [← c116538, ← c116539, k2, ← c116540] at e_1910_11
  have hr := e_1910_11
  simp only [e_1910_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 55) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12600)) := by
  have c116541 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116542 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116543 := (publicBatchWrapper86_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116544 := (publicBatchWrapper86_copies3642 a h).1
  have c116545 := (publicBatchWrapper86_copies3642 a h).2.1
  have c116546 := (publicBatchWrapper86_copies3642 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_12
  simp only [← c116541, ← c116542, ← c116543] at e_1910_12
  have e_1910_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_13
  simp only [← c116544, ← c116545, k2, ← c116546] at e_1910_13
  have hr := e_1910_13
  simp only [e_1910_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 63) = bselect (a (.wire 61 35)) (a (.virt 200)) (a (.virt 12601)) := by
  have c116547 := (publicBatchWrapper86_copies3642 a h).2.2.2.1
  have c116548 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.1
  have c116549 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.1
  have c116550 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.1
  have c116551 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.1
  have c116552 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_14
  simp only [← c116547, ← c116548, ← c116549] at e_1910_14
  have e_1910_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_15
  simp only [← c116550, ← c116551, k2, ← c116552] at e_1910_15
  have hr := e_1910_15
  simp only [e_1910_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12770)) := by
  have c116553 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.1
  have c116554 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116555 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116556 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116557 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116558 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_16
  simp only [← c116553, ← c116554, ← c116555] at e_1910_16
  have e_1910_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_17
  simp only [← c116556, ← c116557, k2, ← c116558] at e_1910_17
  have hr := e_1910_17
  simp only [e_1910_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1910 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12771)) := by
  have c116559 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116560 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116561 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116562 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116563 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116564 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1910_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_18
  simp only [← c116559, ← c116560, ← c116561] at e_1910_18
  have e_1910_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_19
  simp only [← c116562, ← c116563, k2, ← c116564] at e_1910_19
  have hr := e_1910_19
  simp only [e_1910_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12772)) := by
  have c116565 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116566 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116567 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116568 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116569 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116570 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_0
  simp only [← c116565, ← c116566, ← c116567] at e_1911_0
  have e_1911_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_1
  simp only [← c116568, ← c116569, k2, ← c116570] at e_1911_1
  have hr := e_1911_1
  simp only [e_1911_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12773)) := by
  have c116571 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116572 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116573 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116574 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116575 := (publicBatchWrapper86_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116576 := (publicBatchWrapper86_copies3643 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_2
  simp only [← c116571, ← c116572, ← c116573] at e_1911_2
  have e_1911_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_3
  simp only [← c116574, ← c116575, k2, ← c116576] at e_1911_3
  have hr := e_1911_3
  simp only [e_1911_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12774)) := by
  have c116577 := (publicBatchWrapper86_copies3643 a h).2.1
  have c116578 := (publicBatchWrapper86_copies3643 a h).2.2.1
  have c116579 := (publicBatchWrapper86_copies3643 a h).2.2.2.1
  have c116580 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.1
  have c116581 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.1
  have c116582 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_4
  simp only [← c116577, ← c116578, ← c116579] at e_1911_4
  have e_1911_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_5
  simp only [← c116580, ← c116581, k2, ← c116582] at e_1911_5
  have hr := e_1911_5
  simp only [e_1911_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12775)) := by
  have c116583 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.1
  have c116584 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.1
  have c116585 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.1
  have c116586 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116587 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116588 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_6
  simp only [← c116583, ← c116584, ← c116585] at e_1911_6
  have e_1911_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_7
  simp only [← c116586, ← c116587, k2, ← c116588] at e_1911_7
  have hr := e_1911_7
  simp only [e_1911_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12776)) := by
  have c116589 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116590 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116591 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116592 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116593 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116594 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_8
  simp only [← c116589, ← c116590, ← c116591] at e_1911_8
  have e_1911_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_9
  simp only [← c116592, ← c116593, k2, ← c116594] at e_1911_9
  have hr := e_1911_9
  simp only [e_1911_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12777)) := by
  have c116595 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116596 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116597 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116598 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116599 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116600 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_10
  simp only [← c116595, ← c116596, ← c116597] at e_1911_10
  have e_1911_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_11
  simp only [← c116598, ← c116599, k2, ← c116600] at e_1911_11
  have hr := e_1911_11
  simp only [e_1911_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12778)) := by
  have c116601 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116602 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116603 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116604 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116605 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116606 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_12
  simp only [← c116601, ← c116602, ← c116603] at e_1911_12
  have e_1911_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_13
  simp only [← c116604, ← c116605, k2, ← c116606] at e_1911_13
  have hr := e_1911_13
  simp only [e_1911_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12779)) := by
  have c116607 := (publicBatchWrapper86_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116608 := (publicBatchWrapper86_copies3644 a h).1
  have c116609 := (publicBatchWrapper86_copies3644 a h).2.1
  have c116610 := (publicBatchWrapper86_copies3644 a h).2.2.1
  have c116611 := (publicBatchWrapper86_copies3644 a h).2.2.2.1
  have c116612 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_14
  simp only [← c116607, ← c116608, ← c116609] at e_1911_14
  have e_1911_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_15
  simp only [← c116610, ← c116611, k2, ← c116612] at e_1911_15
  have hr := e_1911_15
  simp only [e_1911_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12780)) := by
  have c116613 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.1
  have c116614 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.1
  have c116615 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.1
  have c116616 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.1
  have c116617 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.1
  have c116618 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_16
  simp only [← c116613, ← c116614, ← c116615] at e_1911_16
  have e_1911_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_17
  simp only [← c116616, ← c116617, k2, ← c116618] at e_1911_17
  have hr := e_1911_17
  simp only [e_1911_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1911 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12781)) := by
  have c116619 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116620 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116621 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116622 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116623 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116624 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1911_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_18
  simp only [← c116619, ← c116620, ← c116621] at e_1911_18
  have e_1911_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_19
  simp only [← c116622, ← c116623, k2, ← c116624] at e_1911_19
  have hr := e_1911_19
  simp only [e_1911_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12782)) := by
  have c116625 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116626 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116627 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116628 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116629 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116630 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_0
  simp only [← c116625, ← c116626, ← c116627] at e_1912_0
  have e_1912_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_1
  simp only [← c116628, ← c116629, k2, ← c116630] at e_1912_1
  have hr := e_1912_1
  simp only [e_1912_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12783)) := by
  have c116631 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116632 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116633 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116634 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116635 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116636 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_2
  simp only [← c116631, ← c116632, ← c116633] at e_1912_2
  have e_1912_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_3
  simp only [← c116634, ← c116635, k2, ← c116636] at e_1912_3
  have hr := e_1912_3
  simp only [e_1912_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12784)) := by
  have c116637 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116638 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116639 := (publicBatchWrapper86_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116640 := (publicBatchWrapper86_copies3645 a h).1
  have c116641 := (publicBatchWrapper86_copies3645 a h).2.1
  have c116642 := (publicBatchWrapper86_copies3645 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_4
  simp only [← c116637, ← c116638, ← c116639] at e_1912_4
  have e_1912_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_5
  simp only [← c116640, ← c116641, k2, ← c116642] at e_1912_5
  have hr := e_1912_5
  simp only [e_1912_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12785)) := by
  have c116643 := (publicBatchWrapper86_copies3645 a h).2.2.2.1
  have c116644 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.1
  have c116645 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.1
  have c116646 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.1
  have c116647 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.1
  have c116648 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_6
  simp only [← c116643, ← c116644, ← c116645] at e_1912_6
  have e_1912_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_7
  simp only [← c116646, ← c116647, k2, ← c116648] at e_1912_7
  have hr := e_1912_7
  simp only [e_1912_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12786)) := by
  have c116649 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.1
  have c116650 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116651 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116652 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116653 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116654 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_8
  simp only [← c116649, ← c116650, ← c116651] at e_1912_8
  have e_1912_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_9
  simp only [← c116652, ← c116653, k2, ← c116654] at e_1912_9
  have hr := e_1912_9
  simp only [e_1912_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12787)) := by
  have c116655 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116656 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116657 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116658 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116659 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116660 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_10
  simp only [← c116655, ← c116656, ← c116657] at e_1912_10
  have e_1912_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_11
  simp only [← c116658, ← c116659, k2, ← c116660] at e_1912_11
  have hr := e_1912_11
  simp only [e_1912_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12788)) := by
  have c116661 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116662 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116663 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116664 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116665 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116666 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_12
  simp only [← c116661, ← c116662, ← c116663] at e_1912_12
  have e_1912_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_13
  simp only [← c116664, ← c116665, k2, ← c116666] at e_1912_13
  have hr := e_1912_13
  simp only [e_1912_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12789)) := by
  have c116667 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116668 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116669 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116670 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116671 := (publicBatchWrapper86_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116672 := (publicBatchWrapper86_copies3646 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_14
  simp only [← c116667, ← c116668, ← c116669] at e_1912_14
  have e_1912_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_15
  simp only [← c116670, ← c116671, k2, ← c116672] at e_1912_15
  have hr := e_1912_15
  simp only [e_1912_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12790)) := by
  have c116673 := (publicBatchWrapper86_copies3646 a h).2.1
  have c116674 := (publicBatchWrapper86_copies3646 a h).2.2.1
  have c116675 := (publicBatchWrapper86_copies3646 a h).2.2.2.1
  have c116676 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.1
  have c116677 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.1
  have c116678 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_16
  simp only [← c116673, ← c116674, ← c116675] at e_1912_16
  have e_1912_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_17
  simp only [← c116676, ← c116677, k2, ← c116678] at e_1912_17
  have hr := e_1912_17
  simp only [e_1912_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1912 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12791)) := by
  have c116679 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.1
  have c116680 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.1
  have c116681 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.1
  have c116682 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116683 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116684 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1912_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_18
  simp only [← c116679, ← c116680, ← c116681] at e_1912_18
  have e_1912_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_19
  simp only [← c116682, ← c116683, k2, ← c116684] at e_1912_19
  have hr := e_1912_19
  simp only [e_1912_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 7) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12792)) := by
  have c116685 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116686 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116687 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116688 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116689 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116690 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_0
  simp only [← c116685, ← c116686, ← c116687] at e_1913_0
  have e_1913_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_1
  simp only [← c116688, ← c116689, k2, ← c116690] at e_1913_1
  have hr := e_1913_1
  simp only [e_1913_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 15) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12793)) := by
  have c116691 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116692 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116693 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116694 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116695 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116696 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_2
  simp only [← c116691, ← c116692, ← c116693] at e_1913_2
  have e_1913_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_3
  simp only [← c116694, ← c116695, k2, ← c116696] at e_1913_3
  have hr := e_1913_3
  simp only [e_1913_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 23) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12794)) := by
  have c116697 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116698 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116699 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116700 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116701 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116702 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_4
  simp only [← c116697, ← c116698, ← c116699] at e_1913_4
  have e_1913_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_5
  simp only [← c116700, ← c116701, k2, ← c116702] at e_1913_5
  have hr := e_1913_5
  simp only [e_1913_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 31) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12795)) := by
  have c116703 := (publicBatchWrapper86_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116704 := (publicBatchWrapper86_copies3647 a h).1
  have c116705 := (publicBatchWrapper86_copies3647 a h).2.1
  have c116706 := (publicBatchWrapper86_copies3647 a h).2.2.1
  have c116707 := (publicBatchWrapper86_copies3647 a h).2.2.2.1
  have c116708 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_6
  simp only [← c116703, ← c116704, ← c116705] at e_1913_6
  have e_1913_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_7
  simp only [← c116706, ← c116707, k2, ← c116708] at e_1913_7
  have hr := e_1913_7
  simp only [e_1913_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 39) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12796)) := by
  have c116709 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.1
  have c116710 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.1
  have c116711 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.1
  have c116712 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.1
  have c116713 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.1
  have c116714 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_8
  simp only [← c116709, ← c116710, ← c116711] at e_1913_8
  have e_1913_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_9
  simp only [← c116712, ← c116713, k2, ← c116714] at e_1913_9
  have hr := e_1913_9
  simp only [e_1913_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 47) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12797)) := by
  have c116715 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116716 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116717 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116718 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116719 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116720 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_10
  simp only [← c116715, ← c116716, ← c116717] at e_1913_10
  have e_1913_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_11
  simp only [← c116718, ← c116719, k2, ← c116720] at e_1913_11
  have hr := e_1913_11
  simp only [e_1913_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 55) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12798)) := by
  have c116721 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116722 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116723 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116724 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116725 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116726 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_12
  simp only [← c116721, ← c116722, ← c116723] at e_1913_12
  have e_1913_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_13
  simp only [← c116724, ← c116725, k2, ← c116726] at e_1913_13
  have hr := e_1913_13
  simp only [e_1913_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 63) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12799)) := by
  have c116727 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116728 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116729 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116730 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116731 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116732 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_14
  simp only [← c116727, ← c116728, ← c116729] at e_1913_14
  have e_1913_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_15
  simp only [← c116730, ← c116731, k2, ← c116732] at e_1913_15
  have hr := e_1913_15
  simp only [e_1913_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 71) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12800)) := by
  have c116733 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116734 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116735 := (publicBatchWrapper86_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116736 := (publicBatchWrapper86_copies3648 a h).1
  have c116737 := (publicBatchWrapper86_copies3648 a h).2.1
  have c116738 := (publicBatchWrapper86_copies3648 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_16
  simp only [← c116733, ← c116734, ← c116735] at e_1913_16
  have e_1913_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_17
  simp only [← c116736, ← c116737, k2, ← c116738] at e_1913_17
  have hr := e_1913_17
  simp only [e_1913_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1913 79) = bselect (a (.wire 61 79)) (a (.virt 200)) (a (.virt 12801)) := by
  have c116739 := (publicBatchWrapper86_copies3648 a h).2.2.2.1
  have c116740 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.1
  have c116741 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.1
  have c116742 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.1
  have c116743 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.1
  have c116744 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1913_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_18
  simp only [← c116739, ← c116740, ← c116741] at e_1913_18
  have e_1913_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_19
  simp only [← c116742, ← c116743, k2, ← c116744] at e_1913_19
  have hr := e_1913_19
  simp only [e_1913_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12970)) := by
  have c116745 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.1
  have c116746 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116747 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116748 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116749 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116750 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_0
  simp only [← c116745, ← c116746, ← c116747] at e_1914_0
  have e_1914_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_1
  simp only [← c116748, ← c116749, k2, ← c116750] at e_1914_1
  have hr := e_1914_1
  simp only [e_1914_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12971)) := by
  have c116751 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116752 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116753 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116754 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116755 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116756 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_2
  simp only [← c116751, ← c116752, ← c116753] at e_1914_2
  have e_1914_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_3
  simp only [← c116754, ← c116755, k2, ← c116756] at e_1914_3
  have hr := e_1914_3
  simp only [e_1914_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12972)) := by
  have c116757 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116758 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116759 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116760 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116761 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116762 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_4
  simp only [← c116757, ← c116758, ← c116759] at e_1914_4
  have e_1914_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_5
  simp only [← c116760, ← c116761, k2, ← c116762] at e_1914_5
  have hr := e_1914_5
  simp only [e_1914_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12973)) := by
  have c116763 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116764 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116765 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116766 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116767 := (publicBatchWrapper86_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116768 := (publicBatchWrapper86_copies3649 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_6
  simp only [← c116763, ← c116764, ← c116765] at e_1914_6
  have e_1914_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_7
  simp only [← c116766, ← c116767, k2, ← c116768] at e_1914_7
  have hr := e_1914_7
  simp only [e_1914_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12974)) := by
  have c116769 := (publicBatchWrapper86_copies3649 a h).2.1
  have c116770 := (publicBatchWrapper86_copies3649 a h).2.2.1
  have c116771 := (publicBatchWrapper86_copies3649 a h).2.2.2.1
  have c116772 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.1
  have c116773 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.1
  have c116774 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_8
  simp only [← c116769, ← c116770, ← c116771] at e_1914_8
  have e_1914_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_9
  simp only [← c116772, ← c116773, k2, ← c116774] at e_1914_9
  have hr := e_1914_9
  simp only [e_1914_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12975)) := by
  have c116775 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.1
  have c116776 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.1
  have c116777 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.1
  have c116778 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116779 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116780 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_10
  simp only [← c116775, ← c116776, ← c116777] at e_1914_10
  have e_1914_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_11
  simp only [← c116778, ← c116779, k2, ← c116780] at e_1914_11
  have hr := e_1914_11
  simp only [e_1914_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12976)) := by
  have c116781 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116782 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116783 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116784 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116785 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116786 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_12
  simp only [← c116781, ← c116782, ← c116783] at e_1914_12
  have e_1914_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_13
  simp only [← c116784, ← c116785, k2, ← c116786] at e_1914_13
  have hr := e_1914_13
  simp only [e_1914_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12977)) := by
  have c116787 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116788 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116789 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116790 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116791 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116792 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_14
  simp only [← c116787, ← c116788, ← c116789] at e_1914_14
  have e_1914_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_15
  simp only [← c116790, ← c116791, k2, ← c116792] at e_1914_15
  have hr := e_1914_15
  simp only [e_1914_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12978)) := by
  have c116793 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116794 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116795 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116796 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116797 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116798 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_16
  simp only [← c116793, ← c116794, ← c116795] at e_1914_16
  have e_1914_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_17
  simp only [← c116796, ← c116797, k2, ← c116798] at e_1914_17
  have hr := e_1914_17
  simp only [e_1914_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1914 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12979)) := by
  have c116799 := (publicBatchWrapper86_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116800 := (publicBatchWrapper86_copies3650 a h).1
  have c116801 := (publicBatchWrapper86_copies3650 a h).2.1
  have c116802 := (publicBatchWrapper86_copies3650 a h).2.2.1
  have c116803 := (publicBatchWrapper86_copies3650 a h).2.2.2.1
  have c116804 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1914_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_18
  simp only [← c116799, ← c116800, ← c116801] at e_1914_18
  have e_1914_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_19
  simp only [← c116802, ← c116803, k2, ← c116804] at e_1914_19
  have hr := e_1914_19
  simp only [e_1914_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12980)) := by
  have c116805 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.1
  have c116806 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.1
  have c116807 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.1
  have c116808 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.1
  have c116809 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.1
  have c116810 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_0
  simp only [← c116805, ← c116806, ← c116807] at e_1915_0
  have e_1915_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_1
  simp only [← c116808, ← c116809, k2, ← c116810] at e_1915_1
  have hr := e_1915_1
  simp only [e_1915_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12981)) := by
  have c116811 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116812 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116813 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116814 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116815 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116816 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_2
  simp only [← c116811, ← c116812, ← c116813] at e_1915_2
  have e_1915_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_3
  simp only [← c116814, ← c116815, k2, ← c116816] at e_1915_3
  have hr := e_1915_3
  simp only [e_1915_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12982)) := by
  have c116817 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116818 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116819 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116820 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116821 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116822 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_4
  simp only [← c116817, ← c116818, ← c116819] at e_1915_4
  have e_1915_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_5
  simp only [← c116820, ← c116821, k2, ← c116822] at e_1915_5
  have hr := e_1915_5
  simp only [e_1915_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12983)) := by
  have c116823 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116824 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116825 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116826 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116827 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116828 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_6
  simp only [← c116823, ← c116824, ← c116825] at e_1915_6
  have e_1915_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_7
  simp only [← c116826, ← c116827, k2, ← c116828] at e_1915_7
  have hr := e_1915_7
  simp only [e_1915_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12984)) := by
  have c116829 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116830 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116831 := (publicBatchWrapper86_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116832 := (publicBatchWrapper86_copies3651 a h).1
  have c116833 := (publicBatchWrapper86_copies3651 a h).2.1
  have c116834 := (publicBatchWrapper86_copies3651 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_8
  simp only [← c116829, ← c116830, ← c116831] at e_1915_8
  have e_1915_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_9
  simp only [← c116832, ← c116833, k2, ← c116834] at e_1915_9
  have hr := e_1915_9
  simp only [e_1915_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12985)) := by
  have c116835 := (publicBatchWrapper86_copies3651 a h).2.2.2.1
  have c116836 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.1
  have c116837 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.1
  have c116838 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.1
  have c116839 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.1
  have c116840 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_10
  simp only [← c116835, ← c116836, ← c116837] at e_1915_10
  have e_1915_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_11
  simp only [← c116838, ← c116839, k2, ← c116840] at e_1915_11
  have hr := e_1915_11
  simp only [e_1915_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12986)) := by
  have c116841 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.1
  have c116842 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116843 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116844 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116845 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116846 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_12
  simp only [← c116841, ← c116842, ← c116843] at e_1915_12
  have e_1915_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_13
  simp only [← c116844, ← c116845, k2, ← c116846] at e_1915_13
  have hr := e_1915_13
  simp only [e_1915_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12987)) := by
  have c116847 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116848 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116849 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116850 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116851 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116852 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_14
  simp only [← c116847, ← c116848, ← c116849] at e_1915_14
  have e_1915_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_15
  simp only [← c116850, ← c116851, k2, ← c116852] at e_1915_15
  have hr := e_1915_15
  simp only [e_1915_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12988)) := by
  have c116853 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116854 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116855 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116856 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116857 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116858 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_16
  simp only [← c116853, ← c116854, ← c116855] at e_1915_16
  have e_1915_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_17
  simp only [← c116856, ← c116857, k2, ← c116858] at e_1915_17
  have hr := e_1915_17
  simp only [e_1915_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1915 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12989)) := by
  have c116859 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116860 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116861 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116862 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116863 := (publicBatchWrapper86_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116864 := (publicBatchWrapper86_copies3652 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1915_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_18
  simp only [← c116859, ← c116860, ← c116861] at e_1915_18
  have e_1915_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_19
  simp only [← c116862, ← c116863, k2, ← c116864] at e_1915_19
  have hr := e_1915_19
  simp only [e_1915_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12990)) := by
  have c116865 := (publicBatchWrapper86_copies3652 a h).2.1
  have c116866 := (publicBatchWrapper86_copies3652 a h).2.2.1
  have c116867 := (publicBatchWrapper86_copies3652 a h).2.2.2.1
  have c116868 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.1
  have c116869 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.1
  have c116870 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_0
  simp only [← c116865, ← c116866, ← c116867] at e_1916_0
  have e_1916_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_1
  simp only [← c116868, ← c116869, k2, ← c116870] at e_1916_1
  have hr := e_1916_1
  simp only [e_1916_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12991)) := by
  have c116871 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.1
  have c116872 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.1
  have c116873 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.1
  have c116874 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116875 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116876 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_2
  simp only [← c116871, ← c116872, ← c116873] at e_1916_2
  have e_1916_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_3
  simp only [← c116874, ← c116875, k2, ← c116876] at e_1916_3
  have hr := e_1916_3
  simp only [e_1916_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 23) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12992)) := by
  have c116877 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116878 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116879 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116880 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116881 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116882 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_4
  simp only [← c116877, ← c116878, ← c116879] at e_1916_4
  have e_1916_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_5
  simp only [← c116880, ← c116881, k2, ← c116882] at e_1916_5
  have hr := e_1916_5
  simp only [e_1916_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 31) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12993)) := by
  have c116883 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116884 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116885 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116886 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116887 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116888 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_6
  simp only [← c116883, ← c116884, ← c116885] at e_1916_6
  have e_1916_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_7
  simp only [← c116886, ← c116887, k2, ← c116888] at e_1916_7
  have hr := e_1916_7
  simp only [e_1916_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 39) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12994)) := by
  have c116889 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116890 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116891 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116892 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116893 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116894 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_8
  simp only [← c116889, ← c116890, ← c116891] at e_1916_8
  have e_1916_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_9
  simp only [← c116892, ← c116893, k2, ← c116894] at e_1916_9
  have hr := e_1916_9
  simp only [e_1916_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 47) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12995)) := by
  have c116895 := (publicBatchWrapper86_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116896 := (publicBatchWrapper86_copies3653 a h).1
  have c116897 := (publicBatchWrapper86_copies3653 a h).2.1
  have c116898 := (publicBatchWrapper86_copies3653 a h).2.2.1
  have c116899 := (publicBatchWrapper86_copies3653 a h).2.2.2.1
  have c116900 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_10
  simp only [← c116895, ← c116896, ← c116897] at e_1916_10
  have e_1916_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_11
  simp only [← c116898, ← c116899, k2, ← c116900] at e_1916_11
  have hr := e_1916_11
  simp only [e_1916_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 55) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12996)) := by
  have c116901 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.1
  have c116902 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.1
  have c116903 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.1
  have c116904 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.1
  have c116905 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.1
  have c116906 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_12
  simp only [← c116901, ← c116902, ← c116903] at e_1916_12
  have e_1916_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_13
  simp only [← c116904, ← c116905, k2, ← c116906] at e_1916_13
  have hr := e_1916_13
  simp only [e_1916_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 63) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12997)) := by
  have c116907 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116908 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116909 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116910 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116911 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116912 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_14
  simp only [← c116907, ← c116908, ← c116909] at e_1916_14
  have e_1916_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_15
  simp only [← c116910, ← c116911, k2, ← c116912] at e_1916_15
  have hr := e_1916_15
  simp only [e_1916_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 71) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12998)) := by
  have c116913 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116914 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116915 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116916 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116917 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116918 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_16
  simp only [← c116913, ← c116914, ← c116915] at e_1916_16
  have e_1916_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_17
  simp only [← c116916, ← c116917, k2, ← c116918] at e_1916_17
  have hr := e_1916_17
  simp only [e_1916_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1916 79) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 12999)) := by
  have c116919 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116920 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116921 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116922 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116923 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116924 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1916_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_18
  simp only [← c116919, ← c116920, ← c116921] at e_1916_18
  have e_1916_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_19
  simp only [← c116922, ← c116923, k2, ← c116924] at e_1916_19
  have hr := e_1916_19
  simp only [e_1916_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 7) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 13000)) := by
  have c116925 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116926 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116927 := (publicBatchWrapper86_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116928 := (publicBatchWrapper86_copies3654 a h).1
  have c116929 := (publicBatchWrapper86_copies3654 a h).2.1
  have c116930 := (publicBatchWrapper86_copies3654 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_0
  simp only [← c116925, ← c116926, ← c116927] at e_1917_0
  have e_1917_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_1
  simp only [← c116928, ← c116929, k2, ← c116930] at e_1917_1
  have hr := e_1917_1
  simp only [e_1917_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 15) = bselect (a (.wire 63 43)) (a (.virt 200)) (a (.virt 13001)) := by
  have c116931 := (publicBatchWrapper86_copies3654 a h).2.2.2.1
  have c116932 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.1
  have c116933 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.1
  have c116934 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.1
  have c116935 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.1
  have c116936 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_2
  simp only [← c116931, ← c116932, ← c116933] at e_1917_2
  have e_1917_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_3
  simp only [← c116934, ← c116935, k2, ← c116936] at e_1917_3
  have hr := e_1917_3
  simp only [e_1917_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13170)) := by
  have c116937 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.1
  have c116938 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116939 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116940 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116941 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116942 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_4
  simp only [← c116937, ← c116938, ← c116939] at e_1917_4
  have e_1917_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_5
  simp only [← c116940, ← c116941, k2, ← c116942] at e_1917_5
  have hr := e_1917_5
  simp only [e_1917_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13171)) := by
  have c116943 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116944 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116945 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116946 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116947 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116948 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_6
  simp only [← c116943, ← c116944, ← c116945] at e_1917_6
  have e_1917_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_7
  simp only [← c116946, ← c116947, k2, ← c116948] at e_1917_7
  have hr := e_1917_7
  simp only [e_1917_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13172)) := by
  have c116949 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116950 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116951 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116952 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116953 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116954 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_8
  simp only [← c116949, ← c116950, ← c116951] at e_1917_8
  have e_1917_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_9
  simp only [← c116952, ← c116953, k2, ← c116954] at e_1917_9
  have hr := e_1917_9
  simp only [e_1917_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13173)) := by
  have c116955 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116956 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116957 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116958 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116959 := (publicBatchWrapper86_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116960 := (publicBatchWrapper86_copies3655 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_10
  simp only [← c116955, ← c116956, ← c116957] at e_1917_10
  have e_1917_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_11
  simp only [← c116958, ← c116959, k2, ← c116960] at e_1917_11
  have hr := e_1917_11
  simp only [e_1917_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13174)) := by
  have c116961 := (publicBatchWrapper86_copies3655 a h).2.1
  have c116962 := (publicBatchWrapper86_copies3655 a h).2.2.1
  have c116963 := (publicBatchWrapper86_copies3655 a h).2.2.2.1
  have c116964 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.1
  have c116965 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.1
  have c116966 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_12
  simp only [← c116961, ← c116962, ← c116963] at e_1917_12
  have e_1917_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_13
  simp only [← c116964, ← c116965, k2, ← c116966] at e_1917_13
  have hr := e_1917_13
  simp only [e_1917_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13175)) := by
  have c116967 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.1
  have c116968 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.1
  have c116969 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.1
  have c116970 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116971 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116972 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_14
  simp only [← c116967, ← c116968, ← c116969] at e_1917_14
  have e_1917_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_15
  simp only [← c116970, ← c116971, k2, ← c116972] at e_1917_15
  have hr := e_1917_15
  simp only [e_1917_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13176)) := by
  have c116973 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116974 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116975 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116976 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116977 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116978 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_16
  simp only [← c116973, ← c116974, ← c116975] at e_1917_16
  have e_1917_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_17
  simp only [← c116976, ← c116977, k2, ← c116978] at e_1917_17
  have hr := e_1917_17
  simp only [e_1917_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1917 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13177)) := by
  have c116979 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116980 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116981 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116982 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116983 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116984 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1917_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_18
  simp only [← c116979, ← c116980, ← c116981] at e_1917_18
  have e_1917_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_19
  simp only [← c116982, ← c116983, k2, ← c116984] at e_1917_19
  have hr := e_1917_19
  simp only [e_1917_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13178)) := by
  have c116985 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116986 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116987 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116988 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116989 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116990 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_0
  simp only [← c116985, ← c116986, ← c116987] at e_1918_0
  have e_1918_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_1
  simp only [← c116988, ← c116989, k2, ← c116990] at e_1918_1
  have hr := e_1918_1
  simp only [e_1918_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13179)) := by
  have c116991 := (publicBatchWrapper86_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116992 := (publicBatchWrapper86_copies3656 a h).1
  have c116993 := (publicBatchWrapper86_copies3656 a h).2.1
  have c116994 := (publicBatchWrapper86_copies3656 a h).2.2.1
  have c116995 := (publicBatchWrapper86_copies3656 a h).2.2.2.1
  have c116996 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_2
  simp only [← c116991, ← c116992, ← c116993] at e_1918_2
  have e_1918_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_3
  simp only [← c116994, ← c116995, k2, ← c116996] at e_1918_3
  have hr := e_1918_3
  simp only [e_1918_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13180)) := by
  have c116997 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.1
  have c116998 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.1
  have c116999 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.1
  have c117000 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.1
  have c117001 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.1
  have c117002 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_4
  simp only [← c116997, ← c116998, ← c116999] at e_1918_4
  have e_1918_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_5
  simp only [← c117000, ← c117001, k2, ← c117002] at e_1918_5
  have hr := e_1918_5
  simp only [e_1918_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13181)) := by
  have c117003 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117004 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117005 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117006 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117007 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117008 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_6
  simp only [← c117003, ← c117004, ← c117005] at e_1918_6
  have e_1918_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_7
  simp only [← c117006, ← c117007, k2, ← c117008] at e_1918_7
  have hr := e_1918_7
  simp only [e_1918_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13182)) := by
  have c117009 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117010 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117011 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117012 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117013 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117014 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_8
  simp only [← c117009, ← c117010, ← c117011] at e_1918_8
  have e_1918_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_9
  simp only [← c117012, ← c117013, k2, ← c117014] at e_1918_9
  have hr := e_1918_9
  simp only [e_1918_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13183)) := by
  have c117015 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117016 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117017 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117018 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117019 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117020 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_10
  simp only [← c117015, ← c117016, ← c117017] at e_1918_10
  have e_1918_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_11
  simp only [← c117018, ← c117019, k2, ← c117020] at e_1918_11
  have hr := e_1918_11
  simp only [e_1918_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13184)) := by
  have c117021 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117022 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117023 := (publicBatchWrapper86_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117024 := (publicBatchWrapper86_copies3657 a h).1
  have c117025 := (publicBatchWrapper86_copies3657 a h).2.1
  have c117026 := (publicBatchWrapper86_copies3657 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_12
  simp only [← c117021, ← c117022, ← c117023] at e_1918_12
  have e_1918_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_13
  simp only [← c117024, ← c117025, k2, ← c117026] at e_1918_13
  have hr := e_1918_13
  simp only [e_1918_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13185)) := by
  have c117027 := (publicBatchWrapper86_copies3657 a h).2.2.2.1
  have c117028 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.1
  have c117029 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.1
  have c117030 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.1
  have c117031 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.1
  have c117032 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_14
  simp only [← c117027, ← c117028, ← c117029] at e_1918_14
  have e_1918_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_15
  simp only [← c117030, ← c117031, k2, ← c117032] at e_1918_15
  have hr := e_1918_15
  simp only [e_1918_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13186)) := by
  have c117033 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.1
  have c117034 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117035 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117036 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117037 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117038 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_16
  simp only [← c117033, ← c117034, ← c117035] at e_1918_16
  have e_1918_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_17
  simp only [← c117036, ← c117037, k2, ← c117038] at e_1918_17
  have hr := e_1918_17
  simp only [e_1918_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1918 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13187)) := by
  have c117039 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117040 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117041 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117042 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117043 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117044 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1918_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_18
  simp only [← c117039, ← c117040, ← c117041] at e_1918_18
  have e_1918_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_19
  simp only [← c117042, ← c117043, k2, ← c117044] at e_1918_19
  have hr := e_1918_19
  simp only [e_1918_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13188)) := by
  have c117045 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117046 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117047 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117048 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117049 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117050 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_0
  simp only [← c117045, ← c117046, ← c117047] at e_1919_0
  have e_1919_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_1
  simp only [← c117048, ← c117049, k2, ← c117050] at e_1919_1
  have hr := e_1919_1
  simp only [e_1919_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13189)) := by
  have c117051 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117052 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117053 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117054 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117055 := (publicBatchWrapper86_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117056 := (publicBatchWrapper86_copies3658 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_2
  simp only [← c117051, ← c117052, ← c117053] at e_1919_2
  have e_1919_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_3
  simp only [← c117054, ← c117055, k2, ← c117056] at e_1919_3
  have hr := e_1919_3
  simp only [e_1919_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13190)) := by
  have c117057 := (publicBatchWrapper86_copies3658 a h).2.1
  have c117058 := (publicBatchWrapper86_copies3658 a h).2.2.1
  have c117059 := (publicBatchWrapper86_copies3658 a h).2.2.2.1
  have c117060 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.1
  have c117061 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.1
  have c117062 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_4
  simp only [← c117057, ← c117058, ← c117059] at e_1919_4
  have e_1919_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_5
  simp only [← c117060, ← c117061, k2, ← c117062] at e_1919_5
  have hr := e_1919_5
  simp only [e_1919_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13191)) := by
  have c117063 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.1
  have c117064 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.1
  have c117065 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.1
  have c117066 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117067 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117068 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_6
  simp only [← c117063, ← c117064, ← c117065] at e_1919_6
  have e_1919_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_7
  simp only [← c117066, ← c117067, k2, ← c117068] at e_1919_7
  have hr := e_1919_7
  simp only [e_1919_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 39) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13192)) := by
  have c117069 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117070 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117071 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117072 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117073 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117074 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_8
  simp only [← c117069, ← c117070, ← c117071] at e_1919_8
  have e_1919_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_9
  simp only [← c117072, ← c117073, k2, ← c117074] at e_1919_9
  have hr := e_1919_9
  simp only [e_1919_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 47) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13193)) := by
  have c117075 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117076 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117077 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117078 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117079 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117080 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_10
  simp only [← c117075, ← c117076, ← c117077] at e_1919_10
  have e_1919_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_11
  simp only [← c117078, ← c117079, k2, ← c117080] at e_1919_11
  have hr := e_1919_11
  simp only [e_1919_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 55) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13194)) := by
  have c117081 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117082 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117083 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117084 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117085 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117086 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_12
  simp only [← c117081, ← c117082, ← c117083] at e_1919_12
  have e_1919_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_13
  simp only [← c117084, ← c117085, k2, ← c117086] at e_1919_13
  have hr := e_1919_13
  simp only [e_1919_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 63) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13195)) := by
  have c117087 := (publicBatchWrapper86_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117088 := (publicBatchWrapper86_copies3659 a h).1
  have c117089 := (publicBatchWrapper86_copies3659 a h).2.1
  have c117090 := (publicBatchWrapper86_copies3659 a h).2.2.1
  have c117091 := (publicBatchWrapper86_copies3659 a h).2.2.2.1
  have c117092 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_14
  simp only [← c117087, ← c117088, ← c117089] at e_1919_14
  have e_1919_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_15
  simp only [← c117090, ← c117091, k2, ← c117092] at e_1919_15
  have hr := e_1919_15
  simp only [e_1919_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 71) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13196)) := by
  have c117093 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.1
  have c117094 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.1
  have c117095 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.1
  have c117096 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.1
  have c117097 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.1
  have c117098 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_16
  simp only [← c117093, ← c117094, ← c117095] at e_1919_16
  have e_1919_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_17
  simp only [← c117096, ← c117097, k2, ← c117098] at e_1919_17
  have hr := e_1919_17
  simp only [e_1919_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1919 79) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13197)) := by
  have c117099 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117100 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117101 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117102 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117103 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117104 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1919_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_18
  simp only [← c117099, ← c117100, ← c117101] at e_1919_18
  have e_1919_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_19
  simp only [← c117102, ← c117103, k2, ← c117104] at e_1919_19
  have hr := e_1919_19
  simp only [e_1919_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 7) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13198)) := by
  have c117105 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117106 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117107 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117108 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117109 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117110 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_0
  simp only [← c117105, ← c117106, ← c117107] at e_1920_0
  have e_1920_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_1
  simp only [← c117108, ← c117109, k2, ← c117110] at e_1920_1
  have hr := e_1920_1
  simp only [e_1920_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 15) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13199)) := by
  have c117111 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117112 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117113 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117114 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117115 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117116 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_2
  simp only [← c117111, ← c117112, ← c117113] at e_1920_2
  have e_1920_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_3
  simp only [← c117114, ← c117115, k2, ← c117116] at e_1920_3
  have hr := e_1920_3
  simp only [e_1920_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 23) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13200)) := by
  have c117117 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117118 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117119 := (publicBatchWrapper86_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117120 := (publicBatchWrapper86_copies3660 a h).1
  have c117121 := (publicBatchWrapper86_copies3660 a h).2.1
  have c117122 := (publicBatchWrapper86_copies3660 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_4
  simp only [← c117117, ← c117118, ← c117119] at e_1920_4
  have e_1920_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_5
  simp only [← c117120, ← c117121, k2, ← c117122] at e_1920_5
  have hr := e_1920_5
  simp only [e_1920_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 31) = bselect (a (.wire 65 7)) (a (.virt 200)) (a (.virt 13201)) := by
  have c117123 := (publicBatchWrapper86_copies3660 a h).2.2.2.1
  have c117124 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.1
  have c117125 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.1
  have c117126 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.1
  have c117127 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.1
  have c117128 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_6
  simp only [← c117123, ← c117124, ← c117125] at e_1920_6
  have e_1920_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_7
  simp only [← c117126, ← c117127, k2, ← c117128] at e_1920_7
  have hr := e_1920_7
  simp only [e_1920_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13370)) := by
  have c117129 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.1
  have c117130 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117131 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117132 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117133 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117134 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_8
  simp only [← c117129, ← c117130, ← c117131] at e_1920_8
  have e_1920_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_9
  simp only [← c117132, ← c117133, k2, ← c117134] at e_1920_9
  have hr := e_1920_9
  simp only [e_1920_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13371)) := by
  have c117135 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117136 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117137 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117138 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117139 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117140 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_10
  simp only [← c117135, ← c117136, ← c117137] at e_1920_10
  have e_1920_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_11
  simp only [← c117138, ← c117139, k2, ← c117140] at e_1920_11
  have hr := e_1920_11
  simp only [e_1920_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13372)) := by
  have c117141 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117142 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117143 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117144 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117145 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117146 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_12
  simp only [← c117141, ← c117142, ← c117143] at e_1920_12
  have e_1920_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_13
  simp only [← c117144, ← c117145, k2, ← c117146] at e_1920_13
  have hr := e_1920_13
  simp only [e_1920_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13373)) := by
  have c117147 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117148 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117149 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117150 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117151 := (publicBatchWrapper86_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117152 := (publicBatchWrapper86_copies3661 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_14
  simp only [← c117147, ← c117148, ← c117149] at e_1920_14
  have e_1920_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_15
  simp only [← c117150, ← c117151, k2, ← c117152] at e_1920_15
  have hr := e_1920_15
  simp only [e_1920_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13374)) := by
  have c117153 := (publicBatchWrapper86_copies3661 a h).2.1
  have c117154 := (publicBatchWrapper86_copies3661 a h).2.2.1
  have c117155 := (publicBatchWrapper86_copies3661 a h).2.2.2.1
  have c117156 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.1
  have c117157 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.1
  have c117158 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_16
  simp only [← c117153, ← c117154, ← c117155] at e_1920_16
  have e_1920_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_17
  simp only [← c117156, ← c117157, k2, ← c117158] at e_1920_17
  have hr := e_1920_17
  simp only [e_1920_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1920 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13375)) := by
  have c117159 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.1
  have c117160 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.1
  have c117161 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.1
  have c117162 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117163 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117164 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1920_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_18
  simp only [← c117159, ← c117160, ← c117161] at e_1920_18
  have e_1920_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_19
  simp only [← c117162, ← c117163, k2, ← c117164] at e_1920_19
  have hr := e_1920_19
  simp only [e_1920_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13376)) := by
  have c117165 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117166 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117167 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117168 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117169 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117170 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_0
  simp only [← c117165, ← c117166, ← c117167] at e_1921_0
  have e_1921_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_1
  simp only [← c117168, ← c117169, k2, ← c117170] at e_1921_1
  have hr := e_1921_1
  simp only [e_1921_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13377)) := by
  have c117171 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117172 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117173 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117174 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117175 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117176 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_2
  simp only [← c117171, ← c117172, ← c117173] at e_1921_2
  have e_1921_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_3
  simp only [← c117174, ← c117175, k2, ← c117176] at e_1921_3
  have hr := e_1921_3
  simp only [e_1921_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13378)) := by
  have c117177 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117178 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117179 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117180 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117181 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117182 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_4
  simp only [← c117177, ← c117178, ← c117179] at e_1921_4
  have e_1921_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_5
  simp only [← c117180, ← c117181, k2, ← c117182] at e_1921_5
  have hr := e_1921_5
  simp only [e_1921_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13379)) := by
  have c117183 := (publicBatchWrapper86_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117184 := (publicBatchWrapper86_copies3662 a h).1
  have c117185 := (publicBatchWrapper86_copies3662 a h).2.1
  have c117186 := (publicBatchWrapper86_copies3662 a h).2.2.1
  have c117187 := (publicBatchWrapper86_copies3662 a h).2.2.2.1
  have c117188 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_6
  simp only [← c117183, ← c117184, ← c117185] at e_1921_6
  have e_1921_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_7
  simp only [← c117186, ← c117187, k2, ← c117188] at e_1921_7
  have hr := e_1921_7
  simp only [e_1921_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13380)) := by
  have c117189 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.1
  have c117190 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.1
  have c117191 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.1
  have c117192 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.1
  have c117193 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.1
  have c117194 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_8
  simp only [← c117189, ← c117190, ← c117191] at e_1921_8
  have e_1921_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_9
  simp only [← c117192, ← c117193, k2, ← c117194] at e_1921_9
  have hr := e_1921_9
  simp only [e_1921_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13381)) := by
  have c117195 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117196 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117197 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117198 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117199 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117200 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_10
  simp only [← c117195, ← c117196, ← c117197] at e_1921_10
  have e_1921_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_11
  simp only [← c117198, ← c117199, k2, ← c117200] at e_1921_11
  have hr := e_1921_11
  simp only [e_1921_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13382)) := by
  have c117201 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117202 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117203 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117204 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117205 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117206 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_12
  simp only [← c117201, ← c117202, ← c117203] at e_1921_12
  have e_1921_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_13
  simp only [← c117204, ← c117205, k2, ← c117206] at e_1921_13
  have hr := e_1921_13
  simp only [e_1921_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13383)) := by
  have c117207 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117208 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117209 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117210 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117211 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117212 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_14
  simp only [← c117207, ← c117208, ← c117209] at e_1921_14
  have e_1921_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_15
  simp only [← c117210, ← c117211, k2, ← c117212] at e_1921_15
  have hr := e_1921_15
  simp only [e_1921_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13384)) := by
  have c117213 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117214 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117215 := (publicBatchWrapper86_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117216 := (publicBatchWrapper86_copies3663 a h).1
  have c117217 := (publicBatchWrapper86_copies3663 a h).2.1
  have c117218 := (publicBatchWrapper86_copies3663 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_16
  simp only [← c117213, ← c117214, ← c117215] at e_1921_16
  have e_1921_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_17
  simp only [← c117216, ← c117217, k2, ← c117218] at e_1921_17
  have hr := e_1921_17
  simp only [e_1921_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1921 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13385)) := by
  have c117219 := (publicBatchWrapper86_copies3663 a h).2.2.2.1
  have c117220 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.1
  have c117221 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.1
  have c117222 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.1
  have c117223 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.1
  have c117224 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1921_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_18
  simp only [← c117219, ← c117220, ← c117221] at e_1921_18
  have e_1921_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_19
  simp only [← c117222, ← c117223, k2, ← c117224] at e_1921_19
  have hr := e_1921_19
  simp only [e_1921_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13386)) := by
  have c117225 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.1
  have c117226 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117227 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117228 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117229 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117230 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_0
  simp only [← c117225, ← c117226, ← c117227] at e_1922_0
  have e_1922_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_1
  simp only [← c117228, ← c117229, k2, ← c117230] at e_1922_1
  have hr := e_1922_1
  simp only [e_1922_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13387)) := by
  have c117231 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117232 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117233 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117234 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117235 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117236 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_2
  simp only [← c117231, ← c117232, ← c117233] at e_1922_2
  have e_1922_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_3
  simp only [← c117234, ← c117235, k2, ← c117236] at e_1922_3
  have hr := e_1922_3
  simp only [e_1922_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13388)) := by
  have c117237 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117238 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117239 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117240 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117241 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117242 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_4
  simp only [← c117237, ← c117238, ← c117239] at e_1922_4
  have e_1922_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_5
  simp only [← c117240, ← c117241, k2, ← c117242] at e_1922_5
  have hr := e_1922_5
  simp only [e_1922_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13389)) := by
  have c117243 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117244 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117245 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117246 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117247 := (publicBatchWrapper86_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117248 := (publicBatchWrapper86_copies3664 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_6
  simp only [← c117243, ← c117244, ← c117245] at e_1922_6
  have e_1922_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_7
  simp only [← c117246, ← c117247, k2, ← c117248] at e_1922_7
  have hr := e_1922_7
  simp only [e_1922_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13390)) := by
  have c117249 := (publicBatchWrapper86_copies3664 a h).2.1
  have c117250 := (publicBatchWrapper86_copies3664 a h).2.2.1
  have c117251 := (publicBatchWrapper86_copies3664 a h).2.2.2.1
  have c117252 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.1
  have c117253 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.1
  have c117254 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_8
  simp only [← c117249, ← c117250, ← c117251] at e_1922_8
  have e_1922_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_9
  simp only [← c117252, ← c117253, k2, ← c117254] at e_1922_9
  have hr := e_1922_9
  simp only [e_1922_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13391)) := by
  have c117255 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.1
  have c117256 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.1
  have c117257 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.1
  have c117258 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117259 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117260 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_10
  simp only [← c117255, ← c117256, ← c117257] at e_1922_10
  have e_1922_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_11
  simp only [← c117258, ← c117259, k2, ← c117260] at e_1922_11
  have hr := e_1922_11
  simp only [e_1922_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 55) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13392)) := by
  have c117261 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117262 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117263 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117264 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117265 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117266 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_12
  simp only [← c117261, ← c117262, ← c117263] at e_1922_12
  have e_1922_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_13
  simp only [← c117264, ← c117265, k2, ← c117266] at e_1922_13
  have hr := e_1922_13
  simp only [e_1922_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 63) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13393)) := by
  have c117267 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117268 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117269 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117270 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117271 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117272 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_14
  simp only [← c117267, ← c117268, ← c117269] at e_1922_14
  have e_1922_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_15
  simp only [← c117270, ← c117271, k2, ← c117272] at e_1922_15
  have hr := e_1922_15
  simp only [e_1922_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 71) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13394)) := by
  have c117273 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117274 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117275 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117276 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117277 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117278 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_16
  simp only [← c117273, ← c117274, ← c117275] at e_1922_16
  have e_1922_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_17
  simp only [← c117276, ← c117277, k2, ← c117278] at e_1922_17
  have hr := e_1922_17
  simp only [e_1922_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1922 79) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13395)) := by
  have c117279 := (publicBatchWrapper86_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117280 := (publicBatchWrapper86_copies3665 a h).1
  have c117281 := (publicBatchWrapper86_copies3665 a h).2.1
  have c117282 := (publicBatchWrapper86_copies3665 a h).2.2.1
  have c117283 := (publicBatchWrapper86_copies3665 a h).2.2.2.1
  have c117284 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1922_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_18
  simp only [← c117279, ← c117280, ← c117281] at e_1922_18
  have e_1922_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_19
  simp only [← c117282, ← c117283, k2, ← c117284] at e_1922_19
  have hr := e_1922_19
  simp only [e_1922_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 7) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13396)) := by
  have c117285 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.1
  have c117286 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.1
  have c117287 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.1
  have c117288 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.1
  have c117289 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.1
  have c117290 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_0
  simp only [← c117285, ← c117286, ← c117287] at e_1923_0
  have e_1923_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_1
  simp only [← c117288, ← c117289, k2, ← c117290] at e_1923_1
  have hr := e_1923_1
  simp only [e_1923_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 15) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13397)) := by
  have c117291 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117292 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117293 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117294 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117295 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117296 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_2
  simp only [← c117291, ← c117292, ← c117293] at e_1923_2
  have e_1923_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_3
  simp only [← c117294, ← c117295, k2, ← c117296] at e_1923_3
  have hr := e_1923_3
  simp only [e_1923_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 23) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13398)) := by
  have c117297 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117298 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117299 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117300 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117301 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117302 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_4
  simp only [← c117297, ← c117298, ← c117299] at e_1923_4
  have e_1923_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_5
  simp only [← c117300, ← c117301, k2, ← c117302] at e_1923_5
  have hr := e_1923_5
  simp only [e_1923_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 31) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13399)) := by
  have c117303 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117304 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117305 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117306 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117307 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117308 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_6
  simp only [← c117303, ← c117304, ← c117305] at e_1923_6
  have e_1923_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_7
  simp only [← c117306, ← c117307, k2, ← c117308] at e_1923_7
  have hr := e_1923_7
  simp only [e_1923_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 39) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13400)) := by
  have c117309 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117310 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117311 := (publicBatchWrapper86_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117312 := (publicBatchWrapper86_copies3666 a h).1
  have c117313 := (publicBatchWrapper86_copies3666 a h).2.1
  have c117314 := (publicBatchWrapper86_copies3666 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_8
  simp only [← c117309, ← c117310, ← c117311] at e_1923_8
  have e_1923_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_9
  simp only [← c117312, ← c117313, k2, ← c117314] at e_1923_9
  have hr := e_1923_9
  simp only [e_1923_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 47) = bselect (a (.wire 65 51)) (a (.virt 200)) (a (.virt 13401)) := by
  have c117315 := (publicBatchWrapper86_copies3666 a h).2.2.2.1
  have c117316 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.1
  have c117317 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.1
  have c117318 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.1
  have c117319 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.1
  have c117320 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_10
  simp only [← c117315, ← c117316, ← c117317] at e_1923_10
  have e_1923_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_11
  simp only [← c117318, ← c117319, k2, ← c117320] at e_1923_11
  have hr := e_1923_11
  simp only [e_1923_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13570)) := by
  have c117321 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.1
  have c117322 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117323 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117324 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117325 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117326 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_12
  simp only [← c117321, ← c117322, ← c117323] at e_1923_12
  have e_1923_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_13
  simp only [← c117324, ← c117325, k2, ← c117326] at e_1923_13
  have hr := e_1923_13
  simp only [e_1923_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13571)) := by
  have c117327 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117328 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117329 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117330 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117331 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117332 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_14
  simp only [← c117327, ← c117328, ← c117329] at e_1923_14
  have e_1923_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_15
  simp only [← c117330, ← c117331, k2, ← c117332] at e_1923_15
  have hr := e_1923_15
  simp only [e_1923_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13572)) := by
  have c117333 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117334 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117335 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117336 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117337 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117338 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_16
  simp only [← c117333, ← c117334, ← c117335] at e_1923_16
  have e_1923_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_17
  simp only [← c117336, ← c117337, k2, ← c117338] at e_1923_17
  have hr := e_1923_17
  simp only [e_1923_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1923 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13573)) := by
  have c117339 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117340 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117341 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117342 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117343 := (publicBatchWrapper86_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117344 := (publicBatchWrapper86_copies3667 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1923_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_18
  simp only [← c117339, ← c117340, ← c117341] at e_1923_18
  have e_1923_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_19
  simp only [← c117342, ← c117343, k2, ← c117344] at e_1923_19
  have hr := e_1923_19
  simp only [e_1923_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13574)) := by
  have c117345 := (publicBatchWrapper86_copies3667 a h).2.1
  have c117346 := (publicBatchWrapper86_copies3667 a h).2.2.1
  have c117347 := (publicBatchWrapper86_copies3667 a h).2.2.2.1
  have c117348 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.1
  have c117349 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.1
  have c117350 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_0
  simp only [← c117345, ← c117346, ← c117347] at e_1924_0
  have e_1924_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_1
  simp only [← c117348, ← c117349, k2, ← c117350] at e_1924_1
  have hr := e_1924_1
  simp only [e_1924_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13575)) := by
  have c117351 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.1
  have c117352 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.1
  have c117353 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.1
  have c117354 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117355 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117356 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_2
  simp only [← c117351, ← c117352, ← c117353] at e_1924_2
  have e_1924_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_3
  simp only [← c117354, ← c117355, k2, ← c117356] at e_1924_3
  have hr := e_1924_3
  simp only [e_1924_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13576)) := by
  have c117357 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117358 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117359 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117360 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117361 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117362 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_4
  simp only [← c117357, ← c117358, ← c117359] at e_1924_4
  have e_1924_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_5
  simp only [← c117360, ← c117361, k2, ← c117362] at e_1924_5
  have hr := e_1924_5
  simp only [e_1924_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13577)) := by
  have c117363 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117364 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117365 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117366 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117367 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117368 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_6
  simp only [← c117363, ← c117364, ← c117365] at e_1924_6
  have e_1924_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_7
  simp only [← c117366, ← c117367, k2, ← c117368] at e_1924_7
  have hr := e_1924_7
  simp only [e_1924_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13578)) := by
  have c117369 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117370 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117371 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117372 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117373 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117374 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_8
  simp only [← c117369, ← c117370, ← c117371] at e_1924_8
  have e_1924_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_9
  simp only [← c117372, ← c117373, k2, ← c117374] at e_1924_9
  have hr := e_1924_9
  simp only [e_1924_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13579)) := by
  have c117375 := (publicBatchWrapper86_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117376 := (publicBatchWrapper86_copies3668 a h).1
  have c117377 := (publicBatchWrapper86_copies3668 a h).2.1
  have c117378 := (publicBatchWrapper86_copies3668 a h).2.2.1
  have c117379 := (publicBatchWrapper86_copies3668 a h).2.2.2.1
  have c117380 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_10
  simp only [← c117375, ← c117376, ← c117377] at e_1924_10
  have e_1924_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_11
  simp only [← c117378, ← c117379, k2, ← c117380] at e_1924_11
  have hr := e_1924_11
  simp only [e_1924_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13580)) := by
  have c117381 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.1
  have c117382 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.1
  have c117383 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.1
  have c117384 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.1
  have c117385 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.1
  have c117386 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_12
  simp only [← c117381, ← c117382, ← c117383] at e_1924_12
  have e_1924_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_13
  simp only [← c117384, ← c117385, k2, ← c117386] at e_1924_13
  have hr := e_1924_13
  simp only [e_1924_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13581)) := by
  have c117387 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117388 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117389 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117390 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117391 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117392 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_14
  simp only [← c117387, ← c117388, ← c117389] at e_1924_14
  have e_1924_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_15
  simp only [← c117390, ← c117391, k2, ← c117392] at e_1924_15
  have hr := e_1924_15
  simp only [e_1924_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13582)) := by
  have c117393 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117394 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117395 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117396 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117397 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117398 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_16
  simp only [← c117393, ← c117394, ← c117395] at e_1924_16
  have e_1924_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_17
  simp only [← c117396, ← c117397, k2, ← c117398] at e_1924_17
  have hr := e_1924_17
  simp only [e_1924_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1924 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13583)) := by
  have c117399 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117400 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117401 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117402 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117403 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117404 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1924_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_18
  simp only [← c117399, ← c117400, ← c117401] at e_1924_18
  have e_1924_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_19
  simp only [← c117402, ← c117403, k2, ← c117404] at e_1924_19
  have hr := e_1924_19
  simp only [e_1924_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13584)) := by
  have c117405 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117406 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117407 := (publicBatchWrapper86_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117408 := (publicBatchWrapper86_copies3669 a h).1
  have c117409 := (publicBatchWrapper86_copies3669 a h).2.1
  have c117410 := (publicBatchWrapper86_copies3669 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_0
  simp only [← c117405, ← c117406, ← c117407] at e_1925_0
  have e_1925_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_1
  simp only [← c117408, ← c117409, k2, ← c117410] at e_1925_1
  have hr := e_1925_1
  simp only [e_1925_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13585)) := by
  have c117411 := (publicBatchWrapper86_copies3669 a h).2.2.2.1
  have c117412 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.1
  have c117413 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.1
  have c117414 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.1
  have c117415 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.1
  have c117416 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_2
  simp only [← c117411, ← c117412, ← c117413] at e_1925_2
  have e_1925_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_3
  simp only [← c117414, ← c117415, k2, ← c117416] at e_1925_3
  have hr := e_1925_3
  simp only [e_1925_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13586)) := by
  have c117417 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.1
  have c117418 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117419 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117420 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117421 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117422 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_4
  simp only [← c117417, ← c117418, ← c117419] at e_1925_4
  have e_1925_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_5
  simp only [← c117420, ← c117421, k2, ← c117422] at e_1925_5
  have hr := e_1925_5
  simp only [e_1925_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13587)) := by
  have c117423 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117424 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117425 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117426 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117427 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117428 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_6
  simp only [← c117423, ← c117424, ← c117425] at e_1925_6
  have e_1925_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_7
  simp only [← c117426, ← c117427, k2, ← c117428] at e_1925_7
  have hr := e_1925_7
  simp only [e_1925_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13588)) := by
  have c117429 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117430 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117431 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117432 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117433 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117434 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_8
  simp only [← c117429, ← c117430, ← c117431] at e_1925_8
  have e_1925_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_9
  simp only [← c117432, ← c117433, k2, ← c117434] at e_1925_9
  have hr := e_1925_9
  simp only [e_1925_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13589)) := by
  have c117435 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117436 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117437 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117438 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117439 := (publicBatchWrapper86_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117440 := (publicBatchWrapper86_copies3670 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_10
  simp only [← c117435, ← c117436, ← c117437] at e_1925_10
  have e_1925_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_11
  simp only [← c117438, ← c117439, k2, ← c117440] at e_1925_11
  have hr := e_1925_11
  simp only [e_1925_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13590)) := by
  have c117441 := (publicBatchWrapper86_copies3670 a h).2.1
  have c117442 := (publicBatchWrapper86_copies3670 a h).2.2.1
  have c117443 := (publicBatchWrapper86_copies3670 a h).2.2.2.1
  have c117444 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.1
  have c117445 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.1
  have c117446 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_12
  simp only [← c117441, ← c117442, ← c117443] at e_1925_12
  have e_1925_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_13
  simp only [← c117444, ← c117445, k2, ← c117446] at e_1925_13
  have hr := e_1925_13
  simp only [e_1925_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13591)) := by
  have c117447 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.1
  have c117448 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.1
  have c117449 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.1
  have c117450 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117451 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117452 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_14
  simp only [← c117447, ← c117448, ← c117449] at e_1925_14
  have e_1925_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_15
  simp only [← c117450, ← c117451, k2, ← c117452] at e_1925_15
  have hr := e_1925_15
  simp only [e_1925_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 71) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13592)) := by
  have c117453 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117454 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117455 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117456 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117457 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117458 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_16
  simp only [← c117453, ← c117454, ← c117455] at e_1925_16
  have e_1925_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_17
  simp only [← c117456, ← c117457, k2, ← c117458] at e_1925_17
  have hr := e_1925_17
  simp only [e_1925_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1925 79) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13593)) := by
  have c117459 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117460 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117461 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117462 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117463 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117464 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1925_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_18
  simp only [← c117459, ← c117460, ← c117461] at e_1925_18
  have e_1925_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_19
  simp only [← c117462, ← c117463, k2, ← c117464] at e_1925_19
  have hr := e_1925_19
  simp only [e_1925_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 7) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13594)) := by
  have c117465 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117466 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117467 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117468 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117469 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117470 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_0
  simp only [← c117465, ← c117466, ← c117467] at e_1926_0
  have e_1926_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_1
  simp only [← c117468, ← c117469, k2, ← c117470] at e_1926_1
  have hr := e_1926_1
  simp only [e_1926_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 15) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13595)) := by
  have c117471 := (publicBatchWrapper86_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117472 := (publicBatchWrapper86_copies3671 a h).1
  have c117473 := (publicBatchWrapper86_copies3671 a h).2.1
  have c117474 := (publicBatchWrapper86_copies3671 a h).2.2.1
  have c117475 := (publicBatchWrapper86_copies3671 a h).2.2.2.1
  have c117476 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_2
  simp only [← c117471, ← c117472, ← c117473] at e_1926_2
  have e_1926_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_3
  simp only [← c117474, ← c117475, k2, ← c117476] at e_1926_3
  have hr := e_1926_3
  simp only [e_1926_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 23) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13596)) := by
  have c117477 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.1
  have c117478 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.1
  have c117479 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.1
  have c117480 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.1
  have c117481 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.1
  have c117482 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_4
  simp only [← c117477, ← c117478, ← c117479] at e_1926_4
  have e_1926_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_5
  simp only [← c117480, ← c117481, k2, ← c117482] at e_1926_5
  have hr := e_1926_5
  simp only [e_1926_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 31) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13597)) := by
  have c117483 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117484 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117485 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117486 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117487 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117488 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_6
  simp only [← c117483, ← c117484, ← c117485] at e_1926_6
  have e_1926_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_7
  simp only [← c117486, ← c117487, k2, ← c117488] at e_1926_7
  have hr := e_1926_7
  simp only [e_1926_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 39) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13598)) := by
  have c117489 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117490 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117491 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117492 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117493 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117494 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_8
  simp only [← c117489, ← c117490, ← c117491] at e_1926_8
  have e_1926_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_9
  simp only [← c117492, ← c117493, k2, ← c117494] at e_1926_9
  have hr := e_1926_9
  simp only [e_1926_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 47) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13599)) := by
  have c117495 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117496 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117497 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117498 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117499 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117500 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_10
  simp only [← c117495, ← c117496, ← c117497] at e_1926_10
  have e_1926_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_11
  simp only [← c117498, ← c117499, k2, ← c117500] at e_1926_11
  have hr := e_1926_11
  simp only [e_1926_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 55) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13600)) := by
  have c117501 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117502 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117503 := (publicBatchWrapper86_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117504 := (publicBatchWrapper86_copies3672 a h).1
  have c117505 := (publicBatchWrapper86_copies3672 a h).2.1
  have c117506 := (publicBatchWrapper86_copies3672 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_12
  simp only [← c117501, ← c117502, ← c117503] at e_1926_12
  have e_1926_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_13
  simp only [← c117504, ← c117505, k2, ← c117506] at e_1926_13
  have hr := e_1926_13
  simp only [e_1926_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 63) = bselect (a (.wire 67 15)) (a (.virt 200)) (a (.virt 13601)) := by
  have c117507 := (publicBatchWrapper86_copies3672 a h).2.2.2.1
  have c117508 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.1
  have c117509 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.1
  have c117510 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.1
  have c117511 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.1
  have c117512 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_14
  simp only [← c117507, ← c117508, ← c117509] at e_1926_14
  have e_1926_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_15
  simp only [← c117510, ← c117511, k2, ← c117512] at e_1926_15
  have hr := e_1926_15
  simp only [e_1926_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13770)) := by
  have c117513 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.1
  have c117514 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117515 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117516 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117517 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117518 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_16
  simp only [← c117513, ← c117514, ← c117515] at e_1926_16
  have e_1926_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_17
  simp only [← c117516, ← c117517, k2, ← c117518] at e_1926_17
  have hr := e_1926_17
  simp only [e_1926_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1926 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13771)) := by
  have c117519 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117520 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117521 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117522 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117523 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117524 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1926_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_18
  simp only [← c117519, ← c117520, ← c117521] at e_1926_18
  have e_1926_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_19
  simp only [← c117522, ← c117523, k2, ← c117524] at e_1926_19
  have hr := e_1926_19
  simp only [e_1926_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13772)) := by
  have c117525 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117526 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117527 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117528 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117529 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117530 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_0
  simp only [← c117525, ← c117526, ← c117527] at e_1927_0
  have e_1927_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_1
  simp only [← c117528, ← c117529, k2, ← c117530] at e_1927_1
  have hr := e_1927_1
  simp only [e_1927_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13773)) := by
  have c117531 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117532 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117533 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117534 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117535 := (publicBatchWrapper86_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117536 := (publicBatchWrapper86_copies3673 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_2
  simp only [← c117531, ← c117532, ← c117533] at e_1927_2
  have e_1927_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_3
  simp only [← c117534, ← c117535, k2, ← c117536] at e_1927_3
  have hr := e_1927_3
  simp only [e_1927_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13774)) := by
  have c117537 := (publicBatchWrapper86_copies3673 a h).2.1
  have c117538 := (publicBatchWrapper86_copies3673 a h).2.2.1
  have c117539 := (publicBatchWrapper86_copies3673 a h).2.2.2.1
  have c117540 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.1
  have c117541 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.1
  have c117542 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_4
  simp only [← c117537, ← c117538, ← c117539] at e_1927_4
  have e_1927_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_5
  simp only [← c117540, ← c117541, k2, ← c117542] at e_1927_5
  have hr := e_1927_5
  simp only [e_1927_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13775)) := by
  have c117543 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.1
  have c117544 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.1
  have c117545 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.1
  have c117546 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117547 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117548 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_6
  simp only [← c117543, ← c117544, ← c117545] at e_1927_6
  have e_1927_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_7
  simp only [← c117546, ← c117547, k2, ← c117548] at e_1927_7
  have hr := e_1927_7
  simp only [e_1927_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13776)) := by
  have c117549 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117550 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117551 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117552 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117553 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117554 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_8
  simp only [← c117549, ← c117550, ← c117551] at e_1927_8
  have e_1927_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_9
  simp only [← c117552, ← c117553, k2, ← c117554] at e_1927_9
  have hr := e_1927_9
  simp only [e_1927_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13777)) := by
  have c117555 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117556 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117557 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117558 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117559 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117560 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_10
  simp only [← c117555, ← c117556, ← c117557] at e_1927_10
  have e_1927_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_11
  simp only [← c117558, ← c117559, k2, ← c117560] at e_1927_11
  have hr := e_1927_11
  simp only [e_1927_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13778)) := by
  have c117561 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117562 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117563 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117564 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117565 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117566 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_12
  simp only [← c117561, ← c117562, ← c117563] at e_1927_12
  have e_1927_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_13
  simp only [← c117564, ← c117565, k2, ← c117566] at e_1927_13
  have hr := e_1927_13
  simp only [e_1927_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13779)) := by
  have c117567 := (publicBatchWrapper86_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117568 := (publicBatchWrapper86_copies3674 a h).1
  have c117569 := (publicBatchWrapper86_copies3674 a h).2.1
  have c117570 := (publicBatchWrapper86_copies3674 a h).2.2.1
  have c117571 := (publicBatchWrapper86_copies3674 a h).2.2.2.1
  have c117572 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_14
  simp only [← c117567, ← c117568, ← c117569] at e_1927_14
  have e_1927_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_15
  simp only [← c117570, ← c117571, k2, ← c117572] at e_1927_15
  have hr := e_1927_15
  simp only [e_1927_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13780)) := by
  have c117573 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.1
  have c117574 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.1
  have c117575 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.1
  have c117576 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.1
  have c117577 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.1
  have c117578 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_16
  simp only [← c117573, ← c117574, ← c117575] at e_1927_16
  have e_1927_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_17
  simp only [← c117576, ← c117577, k2, ← c117578] at e_1927_17
  have hr := e_1927_17
  simp only [e_1927_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1927 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13781)) := by
  have c117579 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117580 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117581 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117582 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117583 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117584 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1927_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_18
  simp only [← c117579, ← c117580, ← c117581] at e_1927_18
  have e_1927_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_19
  simp only [← c117582, ← c117583, k2, ← c117584] at e_1927_19
  have hr := e_1927_19
  simp only [e_1927_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13782)) := by
  have c117585 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117586 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117587 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117588 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117589 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117590 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_0
  simp only [← c117585, ← c117586, ← c117587] at e_1928_0
  have e_1928_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_1
  simp only [← c117588, ← c117589, k2, ← c117590] at e_1928_1
  have hr := e_1928_1
  simp only [e_1928_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13783)) := by
  have c117591 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117592 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117593 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117594 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117595 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117596 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_2
  simp only [← c117591, ← c117592, ← c117593] at e_1928_2
  have e_1928_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_3
  simp only [← c117594, ← c117595, k2, ← c117596] at e_1928_3
  have hr := e_1928_3
  simp only [e_1928_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13784)) := by
  have c117597 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117598 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117599 := (publicBatchWrapper86_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117600 := (publicBatchWrapper86_copies3675 a h).1
  have c117601 := (publicBatchWrapper86_copies3675 a h).2.1
  have c117602 := (publicBatchWrapper86_copies3675 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_4
  simp only [← c117597, ← c117598, ← c117599] at e_1928_4
  have e_1928_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_5
  simp only [← c117600, ← c117601, k2, ← c117602] at e_1928_5
  have hr := e_1928_5
  simp only [e_1928_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13785)) := by
  have c117603 := (publicBatchWrapper86_copies3675 a h).2.2.2.1
  have c117604 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.1
  have c117605 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.1
  have c117606 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.1
  have c117607 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.1
  have c117608 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_6
  simp only [← c117603, ← c117604, ← c117605] at e_1928_6
  have e_1928_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_7
  simp only [← c117606, ← c117607, k2, ← c117608] at e_1928_7
  have hr := e_1928_7
  simp only [e_1928_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13786)) := by
  have c117609 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.1
  have c117610 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117611 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117612 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117613 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117614 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_8
  simp only [← c117609, ← c117610, ← c117611] at e_1928_8
  have e_1928_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_9
  simp only [← c117612, ← c117613, k2, ← c117614] at e_1928_9
  have hr := e_1928_9
  simp only [e_1928_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13787)) := by
  have c117615 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117616 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117617 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117618 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117619 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117620 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_10
  simp only [← c117615, ← c117616, ← c117617] at e_1928_10
  have e_1928_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_11
  simp only [← c117618, ← c117619, k2, ← c117620] at e_1928_11
  have hr := e_1928_11
  simp only [e_1928_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13788)) := by
  have c117621 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117622 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117623 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117624 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117625 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117626 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_12
  simp only [← c117621, ← c117622, ← c117623] at e_1928_12
  have e_1928_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_13
  simp only [← c117624, ← c117625, k2, ← c117626] at e_1928_13
  have hr := e_1928_13
  simp only [e_1928_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13789)) := by
  have c117627 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117628 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117629 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117630 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117631 := (publicBatchWrapper86_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117632 := (publicBatchWrapper86_copies3676 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_14
  simp only [← c117627, ← c117628, ← c117629] at e_1928_14
  have e_1928_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_15
  simp only [← c117630, ← c117631, k2, ← c117632] at e_1928_15
  have hr := e_1928_15
  simp only [e_1928_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13790)) := by
  have c117633 := (publicBatchWrapper86_copies3676 a h).2.1
  have c117634 := (publicBatchWrapper86_copies3676 a h).2.2.1
  have c117635 := (publicBatchWrapper86_copies3676 a h).2.2.2.1
  have c117636 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.1
  have c117637 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.1
  have c117638 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_16
  simp only [← c117633, ← c117634, ← c117635] at e_1928_16
  have e_1928_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_17
  simp only [← c117636, ← c117637, k2, ← c117638] at e_1928_17
  have hr := e_1928_17
  simp only [e_1928_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1928 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13791)) := by
  have c117639 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.1
  have c117640 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.1
  have c117641 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.1
  have c117642 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117643 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117644 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1928_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_18
  simp only [← c117639, ← c117640, ← c117641] at e_1928_18
  have e_1928_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_19
  simp only [← c117642, ← c117643, k2, ← c117644] at e_1928_19
  have hr := e_1928_19
  simp only [e_1928_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13792)) := by
  have c117645 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117646 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117647 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117648 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117649 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117650 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_0
  simp only [← c117645, ← c117646, ← c117647] at e_1929_0
  have e_1929_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_1
  simp only [← c117648, ← c117649, k2, ← c117650] at e_1929_1
  have hr := e_1929_1
  simp only [e_1929_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13793)) := by
  have c117651 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117652 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117653 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117654 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117655 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117656 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_2
  simp only [← c117651, ← c117652, ← c117653] at e_1929_2
  have e_1929_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_3
  simp only [← c117654, ← c117655, k2, ← c117656] at e_1929_3
  have hr := e_1929_3
  simp only [e_1929_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13794)) := by
  have c117657 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117658 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117659 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117660 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117661 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117662 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_4
  simp only [← c117657, ← c117658, ← c117659] at e_1929_4
  have e_1929_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_5
  simp only [← c117660, ← c117661, k2, ← c117662] at e_1929_5
  have hr := e_1929_5
  simp only [e_1929_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13795)) := by
  have c117663 := (publicBatchWrapper86_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117664 := (publicBatchWrapper86_copies3677 a h).1
  have c117665 := (publicBatchWrapper86_copies3677 a h).2.1
  have c117666 := (publicBatchWrapper86_copies3677 a h).2.2.1
  have c117667 := (publicBatchWrapper86_copies3677 a h).2.2.2.1
  have c117668 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_6
  simp only [← c117663, ← c117664, ← c117665] at e_1929_6
  have e_1929_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_7
  simp only [← c117666, ← c117667, k2, ← c117668] at e_1929_7
  have hr := e_1929_7
  simp only [e_1929_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13796)) := by
  have c117669 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.1
  have c117670 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.1
  have c117671 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.1
  have c117672 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.1
  have c117673 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.1
  have c117674 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_8
  simp only [← c117669, ← c117670, ← c117671] at e_1929_8
  have e_1929_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_9
  simp only [← c117672, ← c117673, k2, ← c117674] at e_1929_9
  have hr := e_1929_9
  simp only [e_1929_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13797)) := by
  have c117675 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117676 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117677 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117678 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117679 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117680 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_10
  simp only [← c117675, ← c117676, ← c117677] at e_1929_10
  have e_1929_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_11
  simp only [← c117678, ← c117679, k2, ← c117680] at e_1929_11
  have hr := e_1929_11
  simp only [e_1929_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13798)) := by
  have c117681 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117682 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117683 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117684 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117685 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117686 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_12
  simp only [← c117681, ← c117682, ← c117683] at e_1929_12
  have e_1929_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_13
  simp only [← c117684, ← c117685, k2, ← c117686] at e_1929_13
  have hr := e_1929_13
  simp only [e_1929_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13799)) := by
  have c117687 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117688 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117689 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117690 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117691 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117692 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_14
  simp only [← c117687, ← c117688, ← c117689] at e_1929_14
  have e_1929_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_15
  simp only [← c117690, ← c117691, k2, ← c117692] at e_1929_15
  have hr := e_1929_15
  simp only [e_1929_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13800)) := by
  have c117693 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117694 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117695 := (publicBatchWrapper86_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117696 := (publicBatchWrapper86_copies3678 a h).1
  have c117697 := (publicBatchWrapper86_copies3678 a h).2.1
  have c117698 := (publicBatchWrapper86_copies3678 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_16
  simp only [← c117693, ← c117694, ← c117695] at e_1929_16
  have e_1929_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_17
  simp only [← c117696, ← c117697, k2, ← c117698] at e_1929_17
  have hr := e_1929_17
  simp only [e_1929_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1929 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13801)) := by
  have c117699 := (publicBatchWrapper86_copies3678 a h).2.2.2.1
  have c117700 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.1
  have c117701 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.1
  have c117702 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.1
  have c117703 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.1
  have c117704 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1929_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_18
  simp only [← c117699, ← c117700, ← c117701] at e_1929_18
  have e_1929_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_19
  simp only [← c117702, ← c117703, k2, ← c117704] at e_1929_19
  have hr := e_1929_19
  simp only [e_1929_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13970)) := by
  have c117705 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.1
  have c117706 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117707 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117708 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117709 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117710 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_0
  simp only [← c117705, ← c117706, ← c117707] at e_1930_0
  have e_1930_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_1
  simp only [← c117708, ← c117709, k2, ← c117710] at e_1930_1
  have hr := e_1930_1
  simp only [e_1930_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13971)) := by
  have c117711 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117712 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117713 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117714 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117715 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117716 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_2
  simp only [← c117711, ← c117712, ← c117713] at e_1930_2
  have e_1930_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_3
  simp only [← c117714, ← c117715, k2, ← c117716] at e_1930_3
  have hr := e_1930_3
  simp only [e_1930_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13972)) := by
  have c117717 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117718 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117719 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117720 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117721 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117722 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_4
  simp only [← c117717, ← c117718, ← c117719] at e_1930_4
  have e_1930_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_5
  simp only [← c117720, ← c117721, k2, ← c117722] at e_1930_5
  have hr := e_1930_5
  simp only [e_1930_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13973)) := by
  have c117723 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117724 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117725 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117726 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117727 := (publicBatchWrapper86_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117728 := (publicBatchWrapper86_copies3679 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_6
  simp only [← c117723, ← c117724, ← c117725] at e_1930_6
  have e_1930_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_7
  simp only [← c117726, ← c117727, k2, ← c117728] at e_1930_7
  have hr := e_1930_7
  simp only [e_1930_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13974)) := by
  have c117729 := (publicBatchWrapper86_copies3679 a h).2.1
  have c117730 := (publicBatchWrapper86_copies3679 a h).2.2.1
  have c117731 := (publicBatchWrapper86_copies3679 a h).2.2.2.1
  have c117732 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.1
  have c117733 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.1
  have c117734 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_8
  simp only [← c117729, ← c117730, ← c117731] at e_1930_8
  have e_1930_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_9
  simp only [← c117732, ← c117733, k2, ← c117734] at e_1930_9
  have hr := e_1930_9
  simp only [e_1930_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13975)) := by
  have c117735 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.1
  have c117736 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.1
  have c117737 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.1
  have c117738 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117739 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117740 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_10
  simp only [← c117735, ← c117736, ← c117737] at e_1930_10
  have e_1930_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_11
  simp only [← c117738, ← c117739, k2, ← c117740] at e_1930_11
  have hr := e_1930_11
  simp only [e_1930_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13976)) := by
  have c117741 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117742 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117743 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117744 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117745 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117746 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_12
  simp only [← c117741, ← c117742, ← c117743] at e_1930_12
  have e_1930_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_13
  simp only [← c117744, ← c117745, k2, ← c117746] at e_1930_13
  have hr := e_1930_13
  simp only [e_1930_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13977)) := by
  have c117747 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117748 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117749 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117750 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117751 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117752 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_14
  simp only [← c117747, ← c117748, ← c117749] at e_1930_14
  have e_1930_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_15
  simp only [← c117750, ← c117751, k2, ← c117752] at e_1930_15
  have hr := e_1930_15
  simp only [e_1930_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13978)) := by
  have c117753 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117754 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117755 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117756 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117757 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117758 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_16
  simp only [← c117753, ← c117754, ← c117755] at e_1930_16
  have e_1930_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_17
  simp only [← c117756, ← c117757, k2, ← c117758] at e_1930_17
  have hr := e_1930_17
  simp only [e_1930_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1930 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13979)) := by
  have c117759 := (publicBatchWrapper86_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117760 := (publicBatchWrapper86_copies3680 a h).1
  have c117761 := (publicBatchWrapper86_copies3680 a h).2.1
  have c117762 := (publicBatchWrapper86_copies3680 a h).2.2.1
  have c117763 := (publicBatchWrapper86_copies3680 a h).2.2.2.1
  have c117764 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1930_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_18
  simp only [← c117759, ← c117760, ← c117761] at e_1930_18
  have e_1930_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_19
  simp only [← c117762, ← c117763, k2, ← c117764] at e_1930_19
  have hr := e_1930_19
  simp only [e_1930_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13980)) := by
  have c117765 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.1
  have c117766 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.1
  have c117767 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.1
  have c117768 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.1
  have c117769 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.1
  have c117770 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_0
  simp only [← c117765, ← c117766, ← c117767] at e_1931_0
  have e_1931_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_1
  simp only [← c117768, ← c117769, k2, ← c117770] at e_1931_1
  have hr := e_1931_1
  simp only [e_1931_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13981)) := by
  have c117771 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117772 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117773 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117774 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117775 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117776 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_2
  simp only [← c117771, ← c117772, ← c117773] at e_1931_2
  have e_1931_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_3
  simp only [← c117774, ← c117775, k2, ← c117776] at e_1931_3
  have hr := e_1931_3
  simp only [e_1931_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13982)) := by
  have c117777 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117778 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117779 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117780 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117781 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117782 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_4
  simp only [← c117777, ← c117778, ← c117779] at e_1931_4
  have e_1931_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_5
  simp only [← c117780, ← c117781, k2, ← c117782] at e_1931_5
  have hr := e_1931_5
  simp only [e_1931_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13983)) := by
  have c117783 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117784 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117785 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117786 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117787 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117788 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_6
  simp only [← c117783, ← c117784, ← c117785] at e_1931_6
  have e_1931_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_7
  simp only [← c117786, ← c117787, k2, ← c117788] at e_1931_7
  have hr := e_1931_7
  simp only [e_1931_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13984)) := by
  have c117789 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117790 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117791 := (publicBatchWrapper86_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117792 := (publicBatchWrapper86_copies3681 a h).1
  have c117793 := (publicBatchWrapper86_copies3681 a h).2.1
  have c117794 := (publicBatchWrapper86_copies3681 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_8
  simp only [← c117789, ← c117790, ← c117791] at e_1931_8
  have e_1931_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_9
  simp only [← c117792, ← c117793, k2, ← c117794] at e_1931_9
  have hr := e_1931_9
  simp only [e_1931_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13985)) := by
  have c117795 := (publicBatchWrapper86_copies3681 a h).2.2.2.1
  have c117796 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.1
  have c117797 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.1
  have c117798 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.1
  have c117799 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.1
  have c117800 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_10
  simp only [← c117795, ← c117796, ← c117797] at e_1931_10
  have e_1931_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_11
  simp only [← c117798, ← c117799, k2, ← c117800] at e_1931_11
  have hr := e_1931_11
  simp only [e_1931_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13986)) := by
  have c117801 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.1
  have c117802 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117803 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117804 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117805 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117806 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_12
  simp only [← c117801, ← c117802, ← c117803] at e_1931_12
  have e_1931_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_13
  simp only [← c117804, ← c117805, k2, ← c117806] at e_1931_13
  have hr := e_1931_13
  simp only [e_1931_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13987)) := by
  have c117807 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117808 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117809 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117810 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117811 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117812 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_14
  simp only [← c117807, ← c117808, ← c117809] at e_1931_14
  have e_1931_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_15
  simp only [← c117810, ← c117811, k2, ← c117812] at e_1931_15
  have hr := e_1931_15
  simp only [e_1931_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13988)) := by
  have c117813 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117814 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117815 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117816 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117817 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117818 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_16
  simp only [← c117813, ← c117814, ← c117815] at e_1931_16
  have e_1931_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_17
  simp only [← c117816, ← c117817, k2, ← c117818] at e_1931_17
  have hr := e_1931_17
  simp only [e_1931_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1931 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13989)) := by
  have c117819 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117820 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117821 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117822 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117823 := (publicBatchWrapper86_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117824 := (publicBatchWrapper86_copies3682 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1931_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_18
  simp only [← c117819, ← c117820, ← c117821] at e_1931_18
  have e_1931_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_19
  simp only [← c117822, ← c117823, k2, ← c117824] at e_1931_19
  have hr := e_1931_19
  simp only [e_1931_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13990)) := by
  have c117825 := (publicBatchWrapper86_copies3682 a h).2.1
  have c117826 := (publicBatchWrapper86_copies3682 a h).2.2.1
  have c117827 := (publicBatchWrapper86_copies3682 a h).2.2.2.1
  have c117828 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.1
  have c117829 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.1
  have c117830 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_0
  simp only [← c117825, ← c117826, ← c117827] at e_1932_0
  have e_1932_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_1
  simp only [← c117828, ← c117829, k2, ← c117830] at e_1932_1
  have hr := e_1932_1
  simp only [e_1932_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13991)) := by
  have c117831 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.1
  have c117832 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.1
  have c117833 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.1
  have c117834 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117835 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117836 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_2
  simp only [← c117831, ← c117832, ← c117833] at e_1932_2
  have e_1932_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_3
  simp only [← c117834, ← c117835, k2, ← c117836] at e_1932_3
  have hr := e_1932_3
  simp only [e_1932_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13992)) := by
  have c117837 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117838 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117839 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117840 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117841 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117842 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_4
  simp only [← c117837, ← c117838, ← c117839] at e_1932_4
  have e_1932_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_5
  simp only [← c117840, ← c117841, k2, ← c117842] at e_1932_5
  have hr := e_1932_5
  simp only [e_1932_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13993)) := by
  have c117843 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117844 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117845 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117846 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117847 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117848 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_6
  simp only [← c117843, ← c117844, ← c117845] at e_1932_6
  have e_1932_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_7
  simp only [← c117846, ← c117847, k2, ← c117848] at e_1932_7
  have hr := e_1932_7
  simp only [e_1932_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13994)) := by
  have c117849 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117850 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117851 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117852 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117853 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117854 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_8
  simp only [← c117849, ← c117850, ← c117851] at e_1932_8
  have e_1932_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_9
  simp only [← c117852, ← c117853, k2, ← c117854] at e_1932_9
  have hr := e_1932_9
  simp only [e_1932_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13995)) := by
  have c117855 := (publicBatchWrapper86_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117856 := (publicBatchWrapper86_copies3683 a h).1
  have c117857 := (publicBatchWrapper86_copies3683 a h).2.1
  have c117858 := (publicBatchWrapper86_copies3683 a h).2.2.1
  have c117859 := (publicBatchWrapper86_copies3683 a h).2.2.2.1
  have c117860 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_10
  simp only [← c117855, ← c117856, ← c117857] at e_1932_10
  have e_1932_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_11
  simp only [← c117858, ← c117859, k2, ← c117860] at e_1932_11
  have hr := e_1932_11
  simp only [e_1932_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13996)) := by
  have c117861 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.1
  have c117862 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.1
  have c117863 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.1
  have c117864 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.1
  have c117865 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.1
  have c117866 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_12
  simp only [← c117861, ← c117862, ← c117863] at e_1932_12
  have e_1932_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_13
  simp only [← c117864, ← c117865, k2, ← c117866] at e_1932_13
  have hr := e_1932_13
  simp only [e_1932_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13997)) := by
  have c117867 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117868 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117869 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117870 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117871 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117872 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_14
  simp only [← c117867, ← c117868, ← c117869] at e_1932_14
  have e_1932_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_15
  simp only [← c117870, ← c117871, k2, ← c117872] at e_1932_15
  have hr := e_1932_15
  simp only [e_1932_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13998)) := by
  have c117873 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117874 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117875 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117876 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117877 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117878 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_16
  simp only [← c117873, ← c117874, ← c117875] at e_1932_16
  have e_1932_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_17
  simp only [← c117876, ← c117877, k2, ← c117878] at e_1932_17
  have hr := e_1932_17
  simp only [e_1932_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1932 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13999)) := by
  have c117879 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117880 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117881 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117882 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117883 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117884 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1932_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_18
  simp only [← c117879, ← c117880, ← c117881] at e_1932_18
  have e_1932_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_19
  simp only [← c117882, ← c117883, k2, ← c117884] at e_1932_19
  have hr := e_1932_19
  simp only [e_1932_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 14000)) := by
  have c117885 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117886 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117887 := (publicBatchWrapper86_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117888 := (publicBatchWrapper86_copies3684 a h).1
  have c117889 := (publicBatchWrapper86_copies3684 a h).2.1
  have c117890 := (publicBatchWrapper86_copies3684 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_0
  simp only [← c117885, ← c117886, ← c117887] at e_1933_0
  have e_1933_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_1
  simp only [← c117888, ← c117889, k2, ← c117890] at e_1933_1
  have hr := e_1933_1
  simp only [e_1933_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 14001)) := by
  have c117891 := (publicBatchWrapper86_copies3684 a h).2.2.2.1
  have c117892 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.1
  have c117893 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.1
  have c117894 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.1
  have c117895 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.1
  have c117896 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_2
  simp only [← c117891, ← c117892, ← c117893] at e_1933_2
  have e_1933_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_3
  simp only [← c117894, ← c117895, k2, ← c117896] at e_1933_3
  have hr := e_1933_3
  simp only [e_1933_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14170)) := by
  have c117897 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.1
  have c117898 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117899 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117900 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117901 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117902 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_4
  simp only [← c117897, ← c117898, ← c117899] at e_1933_4
  have e_1933_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_5
  simp only [← c117900, ← c117901, k2, ← c117902] at e_1933_5
  have hr := e_1933_5
  simp only [e_1933_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14171)) := by
  have c117903 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117904 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117905 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117906 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117907 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117908 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_6
  simp only [← c117903, ← c117904, ← c117905] at e_1933_6
  have e_1933_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_7
  simp only [← c117906, ← c117907, k2, ← c117908] at e_1933_7
  have hr := e_1933_7
  simp only [e_1933_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14172)) := by
  have c117909 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117910 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117911 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117912 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117913 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117914 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_8
  simp only [← c117909, ← c117910, ← c117911] at e_1933_8
  have e_1933_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_9
  simp only [← c117912, ← c117913, k2, ← c117914] at e_1933_9
  have hr := e_1933_9
  simp only [e_1933_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14173)) := by
  have c117915 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117916 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117917 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117918 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117919 := (publicBatchWrapper86_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117920 := (publicBatchWrapper86_copies3685 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_10
  simp only [← c117915, ← c117916, ← c117917] at e_1933_10
  have e_1933_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_11
  simp only [← c117918, ← c117919, k2, ← c117920] at e_1933_11
  have hr := e_1933_11
  simp only [e_1933_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14174)) := by
  have c117921 := (publicBatchWrapper86_copies3685 a h).2.1
  have c117922 := (publicBatchWrapper86_copies3685 a h).2.2.1
  have c117923 := (publicBatchWrapper86_copies3685 a h).2.2.2.1
  have c117924 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.1
  have c117925 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.1
  have c117926 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_12
  simp only [← c117921, ← c117922, ← c117923] at e_1933_12
  have e_1933_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_13
  simp only [← c117924, ← c117925, k2, ← c117926] at e_1933_13
  have hr := e_1933_13
  simp only [e_1933_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14175)) := by
  have c117927 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.1
  have c117928 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.1
  have c117929 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.1
  have c117930 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117931 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117932 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_14
  simp only [← c117927, ← c117928, ← c117929] at e_1933_14
  have e_1933_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_15
  simp only [← c117930, ← c117931, k2, ← c117932] at e_1933_15
  have hr := e_1933_15
  simp only [e_1933_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14176)) := by
  have c117933 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117934 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117935 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117936 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117937 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117938 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_16
  simp only [← c117933, ← c117934, ← c117935] at e_1933_16
  have e_1933_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_17
  simp only [← c117936, ← c117937, k2, ← c117938] at e_1933_17
  have hr := e_1933_17
  simp only [e_1933_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1933 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14177)) := by
  have c117939 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117940 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117941 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117942 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117943 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117944 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1933_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_18
  simp only [← c117939, ← c117940, ← c117941] at e_1933_18
  have e_1933_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_19
  simp only [← c117942, ← c117943, k2, ← c117944] at e_1933_19
  have hr := e_1933_19
  simp only [e_1933_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14178)) := by
  have c117945 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117946 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117947 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117948 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117949 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117950 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_0
  simp only [← c117945, ← c117946, ← c117947] at e_1934_0
  have e_1934_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_1
  simp only [← c117948, ← c117949, k2, ← c117950] at e_1934_1
  have hr := e_1934_1
  simp only [e_1934_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14179)) := by
  have c117951 := (publicBatchWrapper86_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117952 := (publicBatchWrapper86_copies3686 a h).1
  have c117953 := (publicBatchWrapper86_copies3686 a h).2.1
  have c117954 := (publicBatchWrapper86_copies3686 a h).2.2.1
  have c117955 := (publicBatchWrapper86_copies3686 a h).2.2.2.1
  have c117956 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_2
  simp only [← c117951, ← c117952, ← c117953] at e_1934_2
  have e_1934_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_3
  simp only [← c117954, ← c117955, k2, ← c117956] at e_1934_3
  have hr := e_1934_3
  simp only [e_1934_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14180)) := by
  have c117957 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.1
  have c117958 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.1
  have c117959 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.1
  have c117960 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.1
  have c117961 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.1
  have c117962 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_4
  simp only [← c117957, ← c117958, ← c117959] at e_1934_4
  have e_1934_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_5
  simp only [← c117960, ← c117961, k2, ← c117962] at e_1934_5
  have hr := e_1934_5
  simp only [e_1934_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14181)) := by
  have c117963 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117964 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117965 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117966 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117967 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117968 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_6
  simp only [← c117963, ← c117964, ← c117965] at e_1934_6
  have e_1934_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_7
  simp only [← c117966, ← c117967, k2, ← c117968] at e_1934_7
  have hr := e_1934_7
  simp only [e_1934_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14182)) := by
  have c117969 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117970 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117971 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117972 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117973 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117974 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_8
  simp only [← c117969, ← c117970, ← c117971] at e_1934_8
  have e_1934_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_9
  simp only [← c117972, ← c117973, k2, ← c117974] at e_1934_9
  have hr := e_1934_9
  simp only [e_1934_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14183)) := by
  have c117975 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117976 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117977 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117978 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117979 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117980 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_10
  simp only [← c117975, ← c117976, ← c117977] at e_1934_10
  have e_1934_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_11
  simp only [← c117978, ← c117979, k2, ← c117980] at e_1934_11
  have hr := e_1934_11
  simp only [e_1934_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14184)) := by
  have c117981 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117982 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117983 := (publicBatchWrapper86_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117984 := (publicBatchWrapper86_copies3687 a h).1
  have c117985 := (publicBatchWrapper86_copies3687 a h).2.1
  have c117986 := (publicBatchWrapper86_copies3687 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_12
  simp only [← c117981, ← c117982, ← c117983] at e_1934_12
  have e_1934_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_13
  simp only [← c117984, ← c117985, k2, ← c117986] at e_1934_13
  have hr := e_1934_13
  simp only [e_1934_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14185)) := by
  have c117987 := (publicBatchWrapper86_copies3687 a h).2.2.2.1
  have c117988 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.1
  have c117989 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.1
  have c117990 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.1
  have c117991 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.1
  have c117992 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_14
  simp only [← c117987, ← c117988, ← c117989] at e_1934_14
  have e_1934_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_15
  simp only [← c117990, ← c117991, k2, ← c117992] at e_1934_15
  have hr := e_1934_15
  simp only [e_1934_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14186)) := by
  have c117993 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.1
  have c117994 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117995 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117996 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117997 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117998 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_16
  simp only [← c117993, ← c117994, ← c117995] at e_1934_16
  have e_1934_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_17
  simp only [← c117996, ← c117997, k2, ← c117998] at e_1934_17
  have hr := e_1934_17
  simp only [e_1934_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1934 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14187)) := by
  have c117999 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118000 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118001 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118002 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118003 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118004 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1934_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_18
  simp only [← c117999, ← c118000, ← c118001] at e_1934_18
  have e_1934_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_19
  simp only [← c118002, ← c118003, k2, ← c118004] at e_1934_19
  have hr := e_1934_19
  simp only [e_1934_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14188)) := by
  have c118005 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118006 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118007 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118008 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118009 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118010 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_0
  simp only [← c118005, ← c118006, ← c118007] at e_1935_0
  have e_1935_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_1
  simp only [← c118008, ← c118009, k2, ← c118010] at e_1935_1
  have hr := e_1935_1
  simp only [e_1935_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14189)) := by
  have c118011 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118012 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118013 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118014 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118015 := (publicBatchWrapper86_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118016 := (publicBatchWrapper86_copies3688 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_2
  simp only [← c118011, ← c118012, ← c118013] at e_1935_2
  have e_1935_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_3
  simp only [← c118014, ← c118015, k2, ← c118016] at e_1935_3
  have hr := e_1935_3
  simp only [e_1935_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14190)) := by
  have c118017 := (publicBatchWrapper86_copies3688 a h).2.1
  have c118018 := (publicBatchWrapper86_copies3688 a h).2.2.1
  have c118019 := (publicBatchWrapper86_copies3688 a h).2.2.2.1
  have c118020 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.1
  have c118021 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.1
  have c118022 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_4
  simp only [← c118017, ← c118018, ← c118019] at e_1935_4
  have e_1935_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_5
  simp only [← c118020, ← c118021, k2, ← c118022] at e_1935_5
  have hr := e_1935_5
  simp only [e_1935_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14191)) := by
  have c118023 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.1
  have c118024 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.1
  have c118025 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.1
  have c118026 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118027 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118028 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_6
  simp only [← c118023, ← c118024, ← c118025] at e_1935_6
  have e_1935_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_7
  simp only [← c118026, ← c118027, k2, ← c118028] at e_1935_7
  have hr := e_1935_7
  simp only [e_1935_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14192)) := by
  have c118029 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118030 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118031 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118032 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118033 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118034 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_8
  simp only [← c118029, ← c118030, ← c118031] at e_1935_8
  have e_1935_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_9
  simp only [← c118032, ← c118033, k2, ← c118034] at e_1935_9
  have hr := e_1935_9
  simp only [e_1935_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14193)) := by
  have c118035 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118036 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118037 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118038 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118039 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118040 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_10
  simp only [← c118035, ← c118036, ← c118037] at e_1935_10
  have e_1935_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_11
  simp only [← c118038, ← c118039, k2, ← c118040] at e_1935_11
  have hr := e_1935_11
  simp only [e_1935_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14194)) := by
  have c118041 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118042 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118043 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118044 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118045 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118046 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_12
  simp only [← c118041, ← c118042, ← c118043] at e_1935_12
  have e_1935_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_13
  simp only [← c118044, ← c118045, k2, ← c118046] at e_1935_13
  have hr := e_1935_13
  simp only [e_1935_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14195)) := by
  have c118047 := (publicBatchWrapper86_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118048 := (publicBatchWrapper86_copies3689 a h).1
  have c118049 := (publicBatchWrapper86_copies3689 a h).2.1
  have c118050 := (publicBatchWrapper86_copies3689 a h).2.2.1
  have c118051 := (publicBatchWrapper86_copies3689 a h).2.2.2.1
  have c118052 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_14
  simp only [← c118047, ← c118048, ← c118049] at e_1935_14
  have e_1935_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_15
  simp only [← c118050, ← c118051, k2, ← c118052] at e_1935_15
  have hr := e_1935_15
  simp only [e_1935_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14196)) := by
  have c118053 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.1
  have c118054 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.1
  have c118055 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.1
  have c118056 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.1
  have c118057 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.1
  have c118058 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_16
  simp only [← c118053, ← c118054, ← c118055] at e_1935_16
  have e_1935_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_17
  simp only [← c118056, ← c118057, k2, ← c118058] at e_1935_17
  have hr := e_1935_17
  simp only [e_1935_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1935 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14197)) := by
  have c118059 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118060 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118061 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118062 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118063 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118064 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1935_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_18
  simp only [← c118059, ← c118060, ← c118061] at e_1935_18
  have e_1935_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_19
  simp only [← c118062, ← c118063, k2, ← c118064] at e_1935_19
  have hr := e_1935_19
  simp only [e_1935_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14198)) := by
  have c118065 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118066 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118067 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118068 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118069 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118070 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_0
  simp only [← c118065, ← c118066, ← c118067] at e_1936_0
  have e_1936_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_1
  simp only [← c118068, ← c118069, k2, ← c118070] at e_1936_1
  have hr := e_1936_1
  simp only [e_1936_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14199)) := by
  have c118071 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118072 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118073 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118074 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118075 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118076 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_2
  simp only [← c118071, ← c118072, ← c118073] at e_1936_2
  have e_1936_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_3
  simp only [← c118074, ← c118075, k2, ← c118076] at e_1936_3
  have hr := e_1936_3
  simp only [e_1936_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14200)) := by
  have c118077 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118078 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118079 := (publicBatchWrapper86_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118080 := (publicBatchWrapper86_copies3690 a h).1
  have c118081 := (publicBatchWrapper86_copies3690 a h).2.1
  have c118082 := (publicBatchWrapper86_copies3690 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_4
  simp only [← c118077, ← c118078, ← c118079] at e_1936_4
  have e_1936_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_5
  simp only [← c118080, ← c118081, k2, ← c118082] at e_1936_5
  have hr := e_1936_5
  simp only [e_1936_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14201)) := by
  have c118083 := (publicBatchWrapper86_copies3690 a h).2.2.2.1
  have c118084 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.1
  have c118085 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.1
  have c118086 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.1
  have c118087 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.1
  have c118088 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_6
  simp only [← c118083, ← c118084, ← c118085] at e_1936_6
  have e_1936_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_7
  simp only [← c118086, ← c118087, k2, ← c118088] at e_1936_7
  have hr := e_1936_7
  simp only [e_1936_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14370)) := by
  have c118089 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.1
  have c118090 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118091 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118092 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118093 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118094 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_8
  simp only [← c118089, ← c118090, ← c118091] at e_1936_8
  have e_1936_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_9
  simp only [← c118092, ← c118093, k2, ← c118094] at e_1936_9
  have hr := e_1936_9
  simp only [e_1936_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14371)) := by
  have c118095 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118096 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118097 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118098 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118099 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118100 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_10
  simp only [← c118095, ← c118096, ← c118097] at e_1936_10
  have e_1936_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_11
  simp only [← c118098, ← c118099, k2, ← c118100] at e_1936_11
  have hr := e_1936_11
  simp only [e_1936_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14372)) := by
  have c118101 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118102 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118103 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118104 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118105 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118106 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_12
  simp only [← c118101, ← c118102, ← c118103] at e_1936_12
  have e_1936_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_13
  simp only [← c118104, ← c118105, k2, ← c118106] at e_1936_13
  have hr := e_1936_13
  simp only [e_1936_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14373)) := by
  have c118107 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118108 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118109 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118110 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118111 := (publicBatchWrapper86_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118112 := (publicBatchWrapper86_copies3691 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_14
  simp only [← c118107, ← c118108, ← c118109] at e_1936_14
  have e_1936_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_15
  simp only [← c118110, ← c118111, k2, ← c118112] at e_1936_15
  have hr := e_1936_15
  simp only [e_1936_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14374)) := by
  have c118113 := (publicBatchWrapper86_copies3691 a h).2.1
  have c118114 := (publicBatchWrapper86_copies3691 a h).2.2.1
  have c118115 := (publicBatchWrapper86_copies3691 a h).2.2.2.1
  have c118116 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.1
  have c118117 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.1
  have c118118 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_16
  simp only [← c118113, ← c118114, ← c118115] at e_1936_16
  have e_1936_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_17
  simp only [← c118116, ← c118117, k2, ← c118118] at e_1936_17
  have hr := e_1936_17
  simp only [e_1936_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1936 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14375)) := by
  have c118119 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.1
  have c118120 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.1
  have c118121 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.1
  have c118122 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118123 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118124 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1936_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_18
  simp only [← c118119, ← c118120, ← c118121] at e_1936_18
  have e_1936_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_19
  simp only [← c118122, ← c118123, k2, ← c118124] at e_1936_19
  have hr := e_1936_19
  simp only [e_1936_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14376)) := by
  have c118125 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118126 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118127 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118128 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118129 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118130 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_0
  simp only [← c118125, ← c118126, ← c118127] at e_1937_0
  have e_1937_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_1
  simp only [← c118128, ← c118129, k2, ← c118130] at e_1937_1
  have hr := e_1937_1
  simp only [e_1937_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14377)) := by
  have c118131 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118132 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118133 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118134 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118135 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118136 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_2
  simp only [← c118131, ← c118132, ← c118133] at e_1937_2
  have e_1937_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_3
  simp only [← c118134, ← c118135, k2, ← c118136] at e_1937_3
  have hr := e_1937_3
  simp only [e_1937_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14378)) := by
  have c118137 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118138 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118139 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118140 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118141 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118142 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_4
  simp only [← c118137, ← c118138, ← c118139] at e_1937_4
  have e_1937_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_5
  simp only [← c118140, ← c118141, k2, ← c118142] at e_1937_5
  have hr := e_1937_5
  simp only [e_1937_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14379)) := by
  have c118143 := (publicBatchWrapper86_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118144 := (publicBatchWrapper86_copies3692 a h).1
  have c118145 := (publicBatchWrapper86_copies3692 a h).2.1
  have c118146 := (publicBatchWrapper86_copies3692 a h).2.2.1
  have c118147 := (publicBatchWrapper86_copies3692 a h).2.2.2.1
  have c118148 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_6
  simp only [← c118143, ← c118144, ← c118145] at e_1937_6
  have e_1937_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_7
  simp only [← c118146, ← c118147, k2, ← c118148] at e_1937_7
  have hr := e_1937_7
  simp only [e_1937_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14380)) := by
  have c118149 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.1
  have c118150 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.1
  have c118151 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.1
  have c118152 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.1
  have c118153 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.1
  have c118154 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_8
  simp only [← c118149, ← c118150, ← c118151] at e_1937_8
  have e_1937_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_9
  simp only [← c118152, ← c118153, k2, ← c118154] at e_1937_9
  have hr := e_1937_9
  simp only [e_1937_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14381)) := by
  have c118155 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118156 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118157 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118158 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118159 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118160 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_10
  simp only [← c118155, ← c118156, ← c118157] at e_1937_10
  have e_1937_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_11
  simp only [← c118158, ← c118159, k2, ← c118160] at e_1937_11
  have hr := e_1937_11
  simp only [e_1937_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14382)) := by
  have c118161 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118162 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118163 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118164 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118165 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118166 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_12
  simp only [← c118161, ← c118162, ← c118163] at e_1937_12
  have e_1937_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_13
  simp only [← c118164, ← c118165, k2, ← c118166] at e_1937_13
  have hr := e_1937_13
  simp only [e_1937_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14383)) := by
  have c118167 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118168 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118169 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118170 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118171 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118172 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_14
  simp only [← c118167, ← c118168, ← c118169] at e_1937_14
  have e_1937_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_15
  simp only [← c118170, ← c118171, k2, ← c118172] at e_1937_15
  have hr := e_1937_15
  simp only [e_1937_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14384)) := by
  have c118173 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118174 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118175 := (publicBatchWrapper86_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118176 := (publicBatchWrapper86_copies3693 a h).1
  have c118177 := (publicBatchWrapper86_copies3693 a h).2.1
  have c118178 := (publicBatchWrapper86_copies3693 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_16
  simp only [← c118173, ← c118174, ← c118175] at e_1937_16
  have e_1937_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_17
  simp only [← c118176, ← c118177, k2, ← c118178] at e_1937_17
  have hr := e_1937_17
  simp only [e_1937_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1937 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14385)) := by
  have c118179 := (publicBatchWrapper86_copies3693 a h).2.2.2.1
  have c118180 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.1
  have c118181 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.1
  have c118182 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.1
  have c118183 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.1
  have c118184 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1937_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_18
  simp only [← c118179, ← c118180, ← c118181] at e_1937_18
  have e_1937_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_19
  simp only [← c118182, ← c118183, k2, ← c118184] at e_1937_19
  have hr := e_1937_19
  simp only [e_1937_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14386)) := by
  have c118185 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.1
  have c118186 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118187 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118188 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118189 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118190 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_0
  simp only [← c118185, ← c118186, ← c118187] at e_1938_0
  have e_1938_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_1
  simp only [← c118188, ← c118189, k2, ← c118190] at e_1938_1
  have hr := e_1938_1
  simp only [e_1938_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14387)) := by
  have c118191 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118192 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118193 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118194 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118195 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118196 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_2
  simp only [← c118191, ← c118192, ← c118193] at e_1938_2
  have e_1938_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_3
  simp only [← c118194, ← c118195, k2, ← c118196] at e_1938_3
  have hr := e_1938_3
  simp only [e_1938_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14388)) := by
  have c118197 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118198 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118199 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118200 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118201 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118202 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_4
  simp only [← c118197, ← c118198, ← c118199] at e_1938_4
  have e_1938_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_5
  simp only [← c118200, ← c118201, k2, ← c118202] at e_1938_5
  have hr := e_1938_5
  simp only [e_1938_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14389)) := by
  have c118203 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118204 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118205 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118206 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118207 := (publicBatchWrapper86_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118208 := (publicBatchWrapper86_copies3694 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_6
  simp only [← c118203, ← c118204, ← c118205] at e_1938_6
  have e_1938_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_7
  simp only [← c118206, ← c118207, k2, ← c118208] at e_1938_7
  have hr := e_1938_7
  simp only [e_1938_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14390)) := by
  have c118209 := (publicBatchWrapper86_copies3694 a h).2.1
  have c118210 := (publicBatchWrapper86_copies3694 a h).2.2.1
  have c118211 := (publicBatchWrapper86_copies3694 a h).2.2.2.1
  have c118212 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.1
  have c118213 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.1
  have c118214 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_8
  simp only [← c118209, ← c118210, ← c118211] at e_1938_8
  have e_1938_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_9
  simp only [← c118212, ← c118213, k2, ← c118214] at e_1938_9
  have hr := e_1938_9
  simp only [e_1938_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14391)) := by
  have c118215 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.1
  have c118216 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.1
  have c118217 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.1
  have c118218 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118219 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118220 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_10
  simp only [← c118215, ← c118216, ← c118217] at e_1938_10
  have e_1938_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_11
  simp only [← c118218, ← c118219, k2, ← c118220] at e_1938_11
  have hr := e_1938_11
  simp only [e_1938_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14392)) := by
  have c118221 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118222 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118223 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118224 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118225 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118226 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_12
  simp only [← c118221, ← c118222, ← c118223] at e_1938_12
  have e_1938_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_13
  simp only [← c118224, ← c118225, k2, ← c118226] at e_1938_13
  have hr := e_1938_13
  simp only [e_1938_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14393)) := by
  have c118227 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118228 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118229 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118230 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118231 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118232 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_14
  simp only [← c118227, ← c118228, ← c118229] at e_1938_14
  have e_1938_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_15
  simp only [← c118230, ← c118231, k2, ← c118232] at e_1938_15
  have hr := e_1938_15
  simp only [e_1938_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14394)) := by
  have c118233 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118234 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118235 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118236 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118237 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118238 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_16
  simp only [← c118233, ← c118234, ← c118235] at e_1938_16
  have e_1938_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_17
  simp only [← c118236, ← c118237, k2, ← c118238] at e_1938_17
  have hr := e_1938_17
  simp only [e_1938_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1938 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14395)) := by
  have c118239 := (publicBatchWrapper86_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118240 := (publicBatchWrapper86_copies3695 a h).1
  have c118241 := (publicBatchWrapper86_copies3695 a h).2.1
  have c118242 := (publicBatchWrapper86_copies3695 a h).2.2.1
  have c118243 := (publicBatchWrapper86_copies3695 a h).2.2.2.1
  have c118244 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1938_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_18
  simp only [← c118239, ← c118240, ← c118241] at e_1938_18
  have e_1938_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_19
  simp only [← c118242, ← c118243, k2, ← c118244] at e_1938_19
  have hr := e_1938_19
  simp only [e_1938_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14396)) := by
  have c118245 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.1
  have c118246 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.1
  have c118247 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.1
  have c118248 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.1
  have c118249 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.1
  have c118250 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_0
  simp only [← c118245, ← c118246, ← c118247] at e_1939_0
  have e_1939_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_1
  simp only [← c118248, ← c118249, k2, ← c118250] at e_1939_1
  have hr := e_1939_1
  simp only [e_1939_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14397)) := by
  have c118251 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118252 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118253 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118254 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118255 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118256 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_2
  simp only [← c118251, ← c118252, ← c118253] at e_1939_2
  have e_1939_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_3
  simp only [← c118254, ← c118255, k2, ← c118256] at e_1939_3
  have hr := e_1939_3
  simp only [e_1939_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14398)) := by
  have c118257 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118258 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118259 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118260 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118261 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118262 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_4
  simp only [← c118257, ← c118258, ← c118259] at e_1939_4
  have e_1939_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_5
  simp only [← c118260, ← c118261, k2, ← c118262] at e_1939_5
  have hr := e_1939_5
  simp only [e_1939_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14399)) := by
  have c118263 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118264 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118265 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118266 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118267 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118268 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_6
  simp only [← c118263, ← c118264, ← c118265] at e_1939_6
  have e_1939_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_7
  simp only [← c118266, ← c118267, k2, ← c118268] at e_1939_7
  have hr := e_1939_7
  simp only [e_1939_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14400)) := by
  have c118269 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118270 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118271 := (publicBatchWrapper86_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118272 := (publicBatchWrapper86_copies3696 a h).1
  have c118273 := (publicBatchWrapper86_copies3696 a h).2.1
  have c118274 := (publicBatchWrapper86_copies3696 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_8
  simp only [← c118269, ← c118270, ← c118271] at e_1939_8
  have e_1939_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_9
  simp only [← c118272, ← c118273, k2, ← c118274] at e_1939_9
  have hr := e_1939_9
  simp only [e_1939_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14401)) := by
  have c118275 := (publicBatchWrapper86_copies3696 a h).2.2.2.1
  have c118276 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.1
  have c118277 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.1
  have c118278 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.1
  have c118279 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.1
  have c118280 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_10
  simp only [← c118275, ← c118276, ← c118277] at e_1939_10
  have e_1939_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_11
  simp only [← c118278, ← c118279, k2, ← c118280] at e_1939_11
  have hr := e_1939_11
  simp only [e_1939_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14570)) := by
  have c118281 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.1
  have c118282 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118283 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118284 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118285 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118286 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_12
  simp only [← c118281, ← c118282, ← c118283] at e_1939_12
  have e_1939_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_13
  simp only [← c118284, ← c118285, k2, ← c118286] at e_1939_13
  have hr := e_1939_13
  simp only [e_1939_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14571)) := by
  have c118287 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118288 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118289 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118290 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118291 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118292 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_14
  simp only [← c118287, ← c118288, ← c118289] at e_1939_14
  have e_1939_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_15
  simp only [← c118290, ← c118291, k2, ← c118292] at e_1939_15
  have hr := e_1939_15
  simp only [e_1939_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14572)) := by
  have c118293 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118294 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118295 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118296 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118297 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118298 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_16
  simp only [← c118293, ← c118294, ← c118295] at e_1939_16
  have e_1939_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_17
  simp only [← c118296, ← c118297, k2, ← c118298] at e_1939_17
  have hr := e_1939_17
  simp only [e_1939_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1939 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14573)) := by
  have c118299 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118300 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118301 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118302 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118303 := (publicBatchWrapper86_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118304 := (publicBatchWrapper86_copies3697 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1939_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_18
  simp only [← c118299, ← c118300, ← c118301] at e_1939_18
  have e_1939_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_19
  simp only [← c118302, ← c118303, k2, ← c118304] at e_1939_19
  have hr := e_1939_19
  simp only [e_1939_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14574)) := by
  have c118305 := (publicBatchWrapper86_copies3697 a h).2.1
  have c118306 := (publicBatchWrapper86_copies3697 a h).2.2.1
  have c118307 := (publicBatchWrapper86_copies3697 a h).2.2.2.1
  have c118308 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.1
  have c118309 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.1
  have c118310 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_0
  simp only [← c118305, ← c118306, ← c118307] at e_1940_0
  have e_1940_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_1
  simp only [← c118308, ← c118309, k2, ← c118310] at e_1940_1
  have hr := e_1940_1
  simp only [e_1940_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14575)) := by
  have c118311 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.1
  have c118312 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.1
  have c118313 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.1
  have c118314 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118315 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118316 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_2
  simp only [← c118311, ← c118312, ← c118313] at e_1940_2
  have e_1940_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_3
  simp only [← c118314, ← c118315, k2, ← c118316] at e_1940_3
  have hr := e_1940_3
  simp only [e_1940_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14576)) := by
  have c118317 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118318 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118319 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118320 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118321 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118322 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_4
  simp only [← c118317, ← c118318, ← c118319] at e_1940_4
  have e_1940_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_5
  simp only [← c118320, ← c118321, k2, ← c118322] at e_1940_5
  have hr := e_1940_5
  simp only [e_1940_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14577)) := by
  have c118323 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118324 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118325 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118326 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118327 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118328 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_6
  simp only [← c118323, ← c118324, ← c118325] at e_1940_6
  have e_1940_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_7
  simp only [← c118326, ← c118327, k2, ← c118328] at e_1940_7
  have hr := e_1940_7
  simp only [e_1940_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14578)) := by
  have c118329 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118330 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118331 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118332 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118333 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118334 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_8
  simp only [← c118329, ← c118330, ← c118331] at e_1940_8
  have e_1940_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_9
  simp only [← c118332, ← c118333, k2, ← c118334] at e_1940_9
  have hr := e_1940_9
  simp only [e_1940_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14579)) := by
  have c118335 := (publicBatchWrapper86_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118336 := (publicBatchWrapper86_copies3698 a h).1
  have c118337 := (publicBatchWrapper86_copies3698 a h).2.1
  have c118338 := (publicBatchWrapper86_copies3698 a h).2.2.1
  have c118339 := (publicBatchWrapper86_copies3698 a h).2.2.2.1
  have c118340 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_10
  simp only [← c118335, ← c118336, ← c118337] at e_1940_10
  have e_1940_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_11
  simp only [← c118338, ← c118339, k2, ← c118340] at e_1940_11
  have hr := e_1940_11
  simp only [e_1940_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14580)) := by
  have c118341 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.1
  have c118342 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.1
  have c118343 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.1
  have c118344 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.1
  have c118345 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.1
  have c118346 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_12
  simp only [← c118341, ← c118342, ← c118343] at e_1940_12
  have e_1940_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_13
  simp only [← c118344, ← c118345, k2, ← c118346] at e_1940_13
  have hr := e_1940_13
  simp only [e_1940_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14581)) := by
  have c118347 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118348 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118349 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118350 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118351 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118352 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_14
  simp only [← c118347, ← c118348, ← c118349] at e_1940_14
  have e_1940_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_15
  simp only [← c118350, ← c118351, k2, ← c118352] at e_1940_15
  have hr := e_1940_15
  simp only [e_1940_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14582)) := by
  have c118353 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118354 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118355 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118356 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118357 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118358 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_16
  simp only [← c118353, ← c118354, ← c118355] at e_1940_16
  have e_1940_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_17
  simp only [← c118356, ← c118357, k2, ← c118358] at e_1940_17
  have hr := e_1940_17
  simp only [e_1940_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1940 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14583)) := by
  have c118359 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118360 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118361 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118362 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118363 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118364 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1940_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_18
  simp only [← c118359, ← c118360, ← c118361] at e_1940_18
  have e_1940_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_19
  simp only [← c118362, ← c118363, k2, ← c118364] at e_1940_19
  have hr := e_1940_19
  simp only [e_1940_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14584)) := by
  have c118365 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118366 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118367 := (publicBatchWrapper86_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118368 := (publicBatchWrapper86_copies3699 a h).1
  have c118369 := (publicBatchWrapper86_copies3699 a h).2.1
  have c118370 := (publicBatchWrapper86_copies3699 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_0
  simp only [← c118365, ← c118366, ← c118367] at e_1941_0
  have e_1941_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_1
  simp only [← c118368, ← c118369, k2, ← c118370] at e_1941_1
  have hr := e_1941_1
  simp only [e_1941_0] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
