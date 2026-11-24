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

theorem publicBatchWrapper89_f5222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2726)) := by
  have c35838 := (publicBatchWrapper89_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35839 := (publicBatchWrapper89_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35840 := (publicBatchWrapper89_copies1120 a h).1
  have c35841 := (publicBatchWrapper89_copies1120 a h).2.1
  have c35842 := (publicBatchWrapper89_copies1120 a h).2.2.1
  have c35843 := (publicBatchWrapper89_copies1120 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_2 := arithEq_of_rows h (row := 564) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_2
  simp only [← c35838, ← c35839, ← c35840] at e_564_2
  have e_564_3 := arithEq_of_rows h (row := 564) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_3
  simp only [← c35841, ← c35842, k2, ← c35843] at e_564_3
  have hr := e_564_3
  simp only [e_564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2727)) := by
  have c35844 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.1
  have c35845 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.1
  have c35846 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.1
  have c35847 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.1
  have c35848 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.1
  have c35849 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_4 := arithEq_of_rows h (row := 564) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_4
  simp only [← c35844, ← c35845, ← c35846] at e_564_4
  have e_564_5 := arithEq_of_rows h (row := 564) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_5
  simp only [← c35847, ← c35848, k2, ← c35849] at e_564_5
  have hr := e_564_5
  simp only [e_564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2728)) := by
  have c35850 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35851 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35852 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35853 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35854 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35855 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_6 := arithEq_of_rows h (row := 564) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_6
  simp only [← c35850, ← c35851, ← c35852] at e_564_6
  have e_564_7 := arithEq_of_rows h (row := 564) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_7
  simp only [← c35853, ← c35854, k2, ← c35855] at e_564_7
  have hr := e_564_7
  simp only [e_564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2729)) := by
  have c35856 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35857 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35858 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35859 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35860 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35861 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_8 := arithEq_of_rows h (row := 564) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_8
  simp only [← c35856, ← c35857, ← c35858] at e_564_8
  have e_564_9 := arithEq_of_rows h (row := 564) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_9
  simp only [← c35859, ← c35860, k2, ← c35861] at e_564_9
  have hr := e_564_9
  simp only [e_564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2730)) := by
  have c35862 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35863 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35864 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35865 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35866 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35867 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_10 := arithEq_of_rows h (row := 564) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_10
  simp only [← c35862, ← c35863, ← c35864] at e_564_10
  have e_564_11 := arithEq_of_rows h (row := 564) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_11
  simp only [← c35865, ← c35866, k2, ← c35867] at e_564_11
  have hr := e_564_11
  simp only [e_564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2731)) := by
  have c35868 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35869 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35870 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35871 := (publicBatchWrapper89_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35872 := (publicBatchWrapper89_copies1121 a h).1
  have c35873 := (publicBatchWrapper89_copies1121 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_12 := arithEq_of_rows h (row := 564) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_12
  simp only [← c35868, ← c35869, ← c35870] at e_564_12
  have e_564_13 := arithEq_of_rows h (row := 564) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_13
  simp only [← c35871, ← c35872, k2, ← c35873] at e_564_13
  have hr := e_564_13
  simp only [e_564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2732)) := by
  have c35874 := (publicBatchWrapper89_copies1121 a h).2.2.1
  have c35875 := (publicBatchWrapper89_copies1121 a h).2.2.2.1
  have c35876 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.1
  have c35877 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.1
  have c35878 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.1
  have c35879 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_14 := arithEq_of_rows h (row := 564) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_14
  simp only [← c35874, ← c35875, ← c35876] at e_564_14
  have e_564_15 := arithEq_of_rows h (row := 564) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_15
  simp only [← c35877, ← c35878, k2, ← c35879] at e_564_15
  have hr := e_564_15
  simp only [e_564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2733)) := by
  have c35880 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.1
  have c35881 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.1
  have c35882 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35883 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35884 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35885 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_16 := arithEq_of_rows h (row := 564) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_16
  simp only [← c35880, ← c35881, ← c35882] at e_564_16
  have e_564_17 := arithEq_of_rows h (row := 564) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_17
  simp only [← c35883, ← c35884, k2, ← c35885] at e_564_17
  have hr := e_564_17
  simp only [e_564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 564 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2734)) := by
  have c35886 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35887 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35888 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35889 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35890 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35891 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_564_18 := arithEq_of_rows h (row := 564) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_18
  simp only [← c35886, ← c35887, ← c35888] at e_564_18
  have e_564_19 := arithEq_of_rows h (row := 564) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_19
  simp only [← c35889, ← c35890, k2, ← c35891] at e_564_19
  have hr := e_564_19
  simp only [e_564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2735)) := by
  have c35892 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35893 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35894 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35895 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35896 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35897 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_0 := arithEq_of_rows h (row := 565) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_0
  simp only [← c35892, ← c35893, ← c35894] at e_565_0
  have e_565_1 := arithEq_of_rows h (row := 565) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_1
  simp only [← c35895, ← c35896, k2, ← c35897] at e_565_1
  have hr := e_565_1
  simp only [e_565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2736)) := by
  have c35898 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35899 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35900 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35901 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35902 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35903 := (publicBatchWrapper89_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_2 := arithEq_of_rows h (row := 565) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_2
  simp only [← c35898, ← c35899, ← c35900] at e_565_2
  have e_565_3 := arithEq_of_rows h (row := 565) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_3
  simp only [← c35901, ← c35902, k2, ← c35903] at e_565_3
  have hr := e_565_3
  simp only [e_565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2737)) := by
  have c35904 := (publicBatchWrapper89_copies1122 a h).1
  have c35905 := (publicBatchWrapper89_copies1122 a h).2.1
  have c35906 := (publicBatchWrapper89_copies1122 a h).2.2.1
  have c35907 := (publicBatchWrapper89_copies1122 a h).2.2.2.1
  have c35908 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.1
  have c35909 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_4 := arithEq_of_rows h (row := 565) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_4
  simp only [← c35904, ← c35905, ← c35906] at e_565_4
  have e_565_5 := arithEq_of_rows h (row := 565) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_5
  simp only [← c35907, ← c35908, k2, ← c35909] at e_565_5
  have hr := e_565_5
  simp only [e_565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2738)) := by
  have c35910 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.1
  have c35911 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.1
  have c35912 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.1
  have c35913 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.1
  have c35914 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35915 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_6 := arithEq_of_rows h (row := 565) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_6
  simp only [← c35910, ← c35911, ← c35912] at e_565_6
  have e_565_7 := arithEq_of_rows h (row := 565) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_7
  simp only [← c35913, ← c35914, k2, ← c35915] at e_565_7
  have hr := e_565_7
  simp only [e_565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2739)) := by
  have c35916 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35917 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35918 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35919 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35920 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35921 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_8 := arithEq_of_rows h (row := 565) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_8
  simp only [← c35916, ← c35917, ← c35918] at e_565_8
  have e_565_9 := arithEq_of_rows h (row := 565) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_9
  simp only [← c35919, ← c35920, k2, ← c35921] at e_565_9
  have hr := e_565_9
  simp only [e_565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2740)) := by
  have c35922 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35923 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35924 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35925 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35926 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35927 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_10 := arithEq_of_rows h (row := 565) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_10
  simp only [← c35922, ← c35923, ← c35924] at e_565_10
  have e_565_11 := arithEq_of_rows h (row := 565) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_11
  simp only [← c35925, ← c35926, k2, ← c35927] at e_565_11
  have hr := e_565_11
  simp only [e_565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2741)) := by
  have c35928 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35929 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35930 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35931 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35932 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35933 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_12 := arithEq_of_rows h (row := 565) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_12
  simp only [← c35928, ← c35929, ← c35930] at e_565_12
  have e_565_13 := arithEq_of_rows h (row := 565) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_13
  simp only [← c35931, ← c35932, k2, ← c35933] at e_565_13
  have hr := e_565_13
  simp only [e_565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2742)) := by
  have c35934 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35935 := (publicBatchWrapper89_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35936 := (publicBatchWrapper89_copies1123 a h).1
  have c35937 := (publicBatchWrapper89_copies1123 a h).2.1
  have c35938 := (publicBatchWrapper89_copies1123 a h).2.2.1
  have c35939 := (publicBatchWrapper89_copies1123 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_14 := arithEq_of_rows h (row := 565) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_14
  simp only [← c35934, ← c35935, ← c35936] at e_565_14
  have e_565_15 := arithEq_of_rows h (row := 565) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_15
  simp only [← c35937, ← c35938, k2, ← c35939] at e_565_15
  have hr := e_565_15
  simp only [e_565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2743)) := by
  have c35940 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.1
  have c35941 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.1
  have c35942 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.1
  have c35943 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.1
  have c35944 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.1
  have c35945 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_16 := arithEq_of_rows h (row := 565) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_16
  simp only [← c35940, ← c35941, ← c35942] at e_565_16
  have e_565_17 := arithEq_of_rows h (row := 565) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_17
  simp only [← c35943, ← c35944, k2, ← c35945] at e_565_17
  have hr := e_565_17
  simp only [e_565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 565 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2744)) := by
  have c35946 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35947 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35948 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35949 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35950 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35951 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_565_18 := arithEq_of_rows h (row := 565) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_18
  simp only [← c35946, ← c35947, ← c35948] at e_565_18
  have e_565_19 := arithEq_of_rows h (row := 565) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_19
  simp only [← c35949, ← c35950, k2, ← c35951] at e_565_19
  have hr := e_565_19
  simp only [e_565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2745)) := by
  have c35952 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35953 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35954 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35955 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35956 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35957 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_0 := arithEq_of_rows h (row := 566) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_0
  simp only [← c35952, ← c35953, ← c35954] at e_566_0
  have e_566_1 := arithEq_of_rows h (row := 566) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_1
  simp only [← c35955, ← c35956, k2, ← c35957] at e_566_1
  have hr := e_566_1
  simp only [e_566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2746)) := by
  have c35958 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35959 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35960 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35961 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35962 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35963 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_2 := arithEq_of_rows h (row := 566) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_2
  simp only [← c35958, ← c35959, ← c35960] at e_566_2
  have e_566_3 := arithEq_of_rows h (row := 566) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_3
  simp only [← c35961, ← c35962, k2, ← c35963] at e_566_3
  have hr := e_566_3
  simp only [e_566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2747)) := by
  have c35964 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35965 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35966 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35967 := (publicBatchWrapper89_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35968 := (publicBatchWrapper89_copies1124 a h).1
  have c35969 := (publicBatchWrapper89_copies1124 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_4 := arithEq_of_rows h (row := 566) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_4
  simp only [← c35964, ← c35965, ← c35966] at e_566_4
  have e_566_5 := arithEq_of_rows h (row := 566) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_5
  simp only [← c35967, ← c35968, k2, ← c35969] at e_566_5
  have hr := e_566_5
  simp only [e_566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2748)) := by
  have c35970 := (publicBatchWrapper89_copies1124 a h).2.2.1
  have c35971 := (publicBatchWrapper89_copies1124 a h).2.2.2.1
  have c35972 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.1
  have c35973 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.1
  have c35974 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.1
  have c35975 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_6 := arithEq_of_rows h (row := 566) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_6
  simp only [← c35970, ← c35971, ← c35972] at e_566_6
  have e_566_7 := arithEq_of_rows h (row := 566) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_7
  simp only [← c35973, ← c35974, k2, ← c35975] at e_566_7
  have hr := e_566_7
  simp only [e_566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2749)) := by
  have c35976 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.1
  have c35977 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.1
  have c35978 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35979 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35980 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35981 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_8 := arithEq_of_rows h (row := 566) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_8
  simp only [← c35976, ← c35977, ← c35978] at e_566_8
  have e_566_9 := arithEq_of_rows h (row := 566) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_9
  simp only [← c35979, ← c35980, k2, ← c35981] at e_566_9
  have hr := e_566_9
  simp only [e_566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2750)) := by
  have c35982 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35983 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35984 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35985 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35986 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35987 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_10 := arithEq_of_rows h (row := 566) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_10
  simp only [← c35982, ← c35983, ← c35984] at e_566_10
  have e_566_11 := arithEq_of_rows h (row := 566) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_11
  simp only [← c35985, ← c35986, k2, ← c35987] at e_566_11
  have hr := e_566_11
  simp only [e_566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2751)) := by
  have c35988 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35989 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35990 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35991 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35992 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35993 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_12 := arithEq_of_rows h (row := 566) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_12
  simp only [← c35988, ← c35989, ← c35990] at e_566_12
  have e_566_13 := arithEq_of_rows h (row := 566) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_13
  simp only [← c35991, ← c35992, k2, ← c35993] at e_566_13
  have hr := e_566_13
  simp only [e_566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2752)) := by
  have c35994 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35995 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35996 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35997 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35998 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35999 := (publicBatchWrapper89_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_14 := arithEq_of_rows h (row := 566) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_14
  simp only [← c35994, ← c35995, ← c35996] at e_566_14
  have e_566_15 := arithEq_of_rows h (row := 566) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_15
  simp only [← c35997, ← c35998, k2, ← c35999] at e_566_15
  have hr := e_566_15
  simp only [e_566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2753)) := by
  have c36000 := (publicBatchWrapper89_copies1125 a h).1
  have c36001 := (publicBatchWrapper89_copies1125 a h).2.1
  have c36002 := (publicBatchWrapper89_copies1125 a h).2.2.1
  have c36003 := (publicBatchWrapper89_copies1125 a h).2.2.2.1
  have c36004 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.1
  have c36005 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_16 := arithEq_of_rows h (row := 566) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_16
  simp only [← c36000, ← c36001, ← c36002] at e_566_16
  have e_566_17 := arithEq_of_rows h (row := 566) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_17
  simp only [← c36003, ← c36004, k2, ← c36005] at e_566_17
  have hr := e_566_17
  simp only [e_566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 566 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2754)) := by
  have c36006 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.1
  have c36007 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.1
  have c36008 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.1
  have c36009 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.1
  have c36010 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36011 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_566_18 := arithEq_of_rows h (row := 566) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_18
  simp only [← c36006, ← c36007, ← c36008] at e_566_18
  have e_566_19 := arithEq_of_rows h (row := 566) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_19
  simp only [← c36009, ← c36010, k2, ← c36011] at e_566_19
  have hr := e_566_19
  simp only [e_566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2755)) := by
  have c36012 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36013 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36014 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36015 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36016 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36017 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_0 := arithEq_of_rows h (row := 567) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_0
  simp only [← c36012, ← c36013, ← c36014] at e_567_0
  have e_567_1 := arithEq_of_rows h (row := 567) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_1
  simp only [← c36015, ← c36016, k2, ← c36017] at e_567_1
  have hr := e_567_1
  simp only [e_567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2756)) := by
  have c36018 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36019 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36020 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36021 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36022 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36023 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_2 := arithEq_of_rows h (row := 567) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_2
  simp only [← c36018, ← c36019, ← c36020] at e_567_2
  have e_567_3 := arithEq_of_rows h (row := 567) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_3
  simp only [← c36021, ← c36022, k2, ← c36023] at e_567_3
  have hr := e_567_3
  simp only [e_567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2757)) := by
  have c36024 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36025 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36026 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36027 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36028 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36029 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_4 := arithEq_of_rows h (row := 567) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_4
  simp only [← c36024, ← c36025, ← c36026] at e_567_4
  have e_567_5 := arithEq_of_rows h (row := 567) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_5
  simp only [← c36027, ← c36028, k2, ← c36029] at e_567_5
  have hr := e_567_5
  simp only [e_567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2758)) := by
  have c36030 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36031 := (publicBatchWrapper89_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36032 := (publicBatchWrapper89_copies1126 a h).1
  have c36033 := (publicBatchWrapper89_copies1126 a h).2.1
  have c36034 := (publicBatchWrapper89_copies1126 a h).2.2.1
  have c36035 := (publicBatchWrapper89_copies1126 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_6 := arithEq_of_rows h (row := 567) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_6
  simp only [← c36030, ← c36031, ← c36032] at e_567_6
  have e_567_7 := arithEq_of_rows h (row := 567) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_7
  simp only [← c36033, ← c36034, k2, ← c36035] at e_567_7
  have hr := e_567_7
  simp only [e_567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2759)) := by
  have c36036 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.1
  have c36037 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.1
  have c36038 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.1
  have c36039 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.1
  have c36040 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.1
  have c36041 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_8 := arithEq_of_rows h (row := 567) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_8
  simp only [← c36036, ← c36037, ← c36038] at e_567_8
  have e_567_9 := arithEq_of_rows h (row := 567) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_9
  simp only [← c36039, ← c36040, k2, ← c36041] at e_567_9
  have hr := e_567_9
  simp only [e_567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2760)) := by
  have c36042 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36043 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36044 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36045 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36046 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36047 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_10 := arithEq_of_rows h (row := 567) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_10
  simp only [← c36042, ← c36043, ← c36044] at e_567_10
  have e_567_11 := arithEq_of_rows h (row := 567) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_11
  simp only [← c36045, ← c36046, k2, ← c36047] at e_567_11
  have hr := e_567_11
  simp only [e_567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2761)) := by
  have c36048 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36049 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36050 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36051 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36052 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36053 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_12 := arithEq_of_rows h (row := 567) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_12
  simp only [← c36048, ← c36049, ← c36050] at e_567_12
  have e_567_13 := arithEq_of_rows h (row := 567) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_13
  simp only [← c36051, ← c36052, k2, ← c36053] at e_567_13
  have hr := e_567_13
  simp only [e_567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2762)) := by
  have c36054 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36055 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36056 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36057 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36058 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36059 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_14 := arithEq_of_rows h (row := 567) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_14
  simp only [← c36054, ← c36055, ← c36056] at e_567_14
  have e_567_15 := arithEq_of_rows h (row := 567) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_15
  simp only [← c36057, ← c36058, k2, ← c36059] at e_567_15
  have hr := e_567_15
  simp only [e_567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2763)) := by
  have c36060 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36061 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36062 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36063 := (publicBatchWrapper89_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36064 := (publicBatchWrapper89_copies1127 a h).1
  have c36065 := (publicBatchWrapper89_copies1127 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_16 := arithEq_of_rows h (row := 567) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_16
  simp only [← c36060, ← c36061, ← c36062] at e_567_16
  have e_567_17 := arithEq_of_rows h (row := 567) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_17
  simp only [← c36063, ← c36064, k2, ← c36065] at e_567_17
  have hr := e_567_17
  simp only [e_567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 567 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2764)) := by
  have c36066 := (publicBatchWrapper89_copies1127 a h).2.2.1
  have c36067 := (publicBatchWrapper89_copies1127 a h).2.2.2.1
  have c36068 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.1
  have c36069 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.1
  have c36070 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.1
  have c36071 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_567_18 := arithEq_of_rows h (row := 567) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_18
  simp only [← c36066, ← c36067, ← c36068] at e_567_18
  have e_567_19 := arithEq_of_rows h (row := 567) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_19
  simp only [← c36069, ← c36070, k2, ← c36071] at e_567_19
  have hr := e_567_19
  simp only [e_567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2765)) := by
  have c36072 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.1
  have c36073 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.1
  have c36074 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36075 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36076 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36077 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_0 := arithEq_of_rows h (row := 568) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_0
  simp only [← c36072, ← c36073, ← c36074] at e_568_0
  have e_568_1 := arithEq_of_rows h (row := 568) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_1
  simp only [← c36075, ← c36076, k2, ← c36077] at e_568_1
  have hr := e_568_1
  simp only [e_568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2766)) := by
  have c36078 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36079 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36080 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36081 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36082 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36083 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_2 := arithEq_of_rows h (row := 568) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_2
  simp only [← c36078, ← c36079, ← c36080] at e_568_2
  have e_568_3 := arithEq_of_rows h (row := 568) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_3
  simp only [← c36081, ← c36082, k2, ← c36083] at e_568_3
  have hr := e_568_3
  simp only [e_568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2767)) := by
  have c36084 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36085 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36086 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36087 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36088 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36089 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_4 := arithEq_of_rows h (row := 568) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_4
  simp only [← c36084, ← c36085, ← c36086] at e_568_4
  have e_568_5 := arithEq_of_rows h (row := 568) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_5
  simp only [← c36087, ← c36088, k2, ← c36089] at e_568_5
  have hr := e_568_5
  simp only [e_568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2768)) := by
  have c36090 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36091 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36092 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36093 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36094 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36095 := (publicBatchWrapper89_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_6 := arithEq_of_rows h (row := 568) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_6
  simp only [← c36090, ← c36091, ← c36092] at e_568_6
  have e_568_7 := arithEq_of_rows h (row := 568) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_7
  simp only [← c36093, ← c36094, k2, ← c36095] at e_568_7
  have hr := e_568_7
  simp only [e_568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2769)) := by
  have c36096 := (publicBatchWrapper89_copies1128 a h).1
  have c36097 := (publicBatchWrapper89_copies1128 a h).2.1
  have c36098 := (publicBatchWrapper89_copies1128 a h).2.2.1
  have c36099 := (publicBatchWrapper89_copies1128 a h).2.2.2.1
  have c36100 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.1
  have c36101 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_8 := arithEq_of_rows h (row := 568) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_8
  simp only [← c36096, ← c36097, ← c36098] at e_568_8
  have e_568_9 := arithEq_of_rows h (row := 568) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_9
  simp only [← c36099, ← c36100, k2, ← c36101] at e_568_9
  have hr := e_568_9
  simp only [e_568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2810)) := by
  have c36102 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.1
  have c36103 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.1
  have c36104 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.1
  have c36105 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.1
  have c36106 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36107 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_10 := arithEq_of_rows h (row := 568) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_10
  simp only [← c36102, ← c36103, ← c36104] at e_568_10
  have e_568_11 := arithEq_of_rows h (row := 568) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_11
  simp only [← c36105, ← c36106, k2, ← c36107] at e_568_11
  have hr := e_568_11
  simp only [e_568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2811)) := by
  have c36108 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36109 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36110 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36111 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36112 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36113 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_12 := arithEq_of_rows h (row := 568) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_12
  simp only [← c36108, ← c36109, ← c36110] at e_568_12
  have e_568_13 := arithEq_of_rows h (row := 568) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_13
  simp only [← c36111, ← c36112, k2, ← c36113] at e_568_13
  have hr := e_568_13
  simp only [e_568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2812)) := by
  have c36114 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36115 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36116 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36117 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36118 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36119 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_14 := arithEq_of_rows h (row := 568) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_14
  simp only [← c36114, ← c36115, ← c36116] at e_568_14
  have e_568_15 := arithEq_of_rows h (row := 568) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_15
  simp only [← c36117, ← c36118, k2, ← c36119] at e_568_15
  have hr := e_568_15
  simp only [e_568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2813)) := by
  have c36120 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36121 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36122 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36123 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36124 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36125 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_16 := arithEq_of_rows h (row := 568) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_16
  simp only [← c36120, ← c36121, ← c36122] at e_568_16
  have e_568_17 := arithEq_of_rows h (row := 568) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_17
  simp only [← c36123, ← c36124, k2, ← c36125] at e_568_17
  have hr := e_568_17
  simp only [e_568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 568 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2814)) := by
  have c36126 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36127 := (publicBatchWrapper89_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36128 := (publicBatchWrapper89_copies1129 a h).1
  have c36129 := (publicBatchWrapper89_copies1129 a h).2.1
  have c36130 := (publicBatchWrapper89_copies1129 a h).2.2.1
  have c36131 := (publicBatchWrapper89_copies1129 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_568_18 := arithEq_of_rows h (row := 568) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_18
  simp only [← c36126, ← c36127, ← c36128] at e_568_18
  have e_568_19 := arithEq_of_rows h (row := 568) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_19
  simp only [← c36129, ← c36130, k2, ← c36131] at e_568_19
  have hr := e_568_19
  simp only [e_568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2815)) := by
  have c36132 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.1
  have c36133 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.1
  have c36134 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.1
  have c36135 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.1
  have c36136 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.1
  have c36137 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_0 := arithEq_of_rows h (row := 569) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_0
  simp only [← c36132, ← c36133, ← c36134] at e_569_0
  have e_569_1 := arithEq_of_rows h (row := 569) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_1
  simp only [← c36135, ← c36136, k2, ← c36137] at e_569_1
  have hr := e_569_1
  simp only [e_569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2816)) := by
  have c36138 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36139 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36140 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36141 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36142 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36143 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_2 := arithEq_of_rows h (row := 569) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_2
  simp only [← c36138, ← c36139, ← c36140] at e_569_2
  have e_569_3 := arithEq_of_rows h (row := 569) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_3
  simp only [← c36141, ← c36142, k2, ← c36143] at e_569_3
  have hr := e_569_3
  simp only [e_569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2817)) := by
  have c36144 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36145 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36146 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36147 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36148 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36149 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_4 := arithEq_of_rows h (row := 569) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_4
  simp only [← c36144, ← c36145, ← c36146] at e_569_4
  have e_569_5 := arithEq_of_rows h (row := 569) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_5
  simp only [← c36147, ← c36148, k2, ← c36149] at e_569_5
  have hr := e_569_5
  simp only [e_569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2818)) := by
  have c36150 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36151 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36152 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36153 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36154 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36155 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_6 := arithEq_of_rows h (row := 569) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_6
  simp only [← c36150, ← c36151, ← c36152] at e_569_6
  have e_569_7 := arithEq_of_rows h (row := 569) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_7
  simp only [← c36153, ← c36154, k2, ← c36155] at e_569_7
  have hr := e_569_7
  simp only [e_569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2819)) := by
  have c36156 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36157 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36158 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36159 := (publicBatchWrapper89_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36160 := (publicBatchWrapper89_copies1130 a h).1
  have c36161 := (publicBatchWrapper89_copies1130 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_8 := arithEq_of_rows h (row := 569) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_8
  simp only [← c36156, ← c36157, ← c36158] at e_569_8
  have e_569_9 := arithEq_of_rows h (row := 569) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_9
  simp only [← c36159, ← c36160, k2, ← c36161] at e_569_9
  have hr := e_569_9
  simp only [e_569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2820)) := by
  have c36162 := (publicBatchWrapper89_copies1130 a h).2.2.1
  have c36163 := (publicBatchWrapper89_copies1130 a h).2.2.2.1
  have c36164 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.1
  have c36165 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.1
  have c36166 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.1
  have c36167 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_10 := arithEq_of_rows h (row := 569) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_10
  simp only [← c36162, ← c36163, ← c36164] at e_569_10
  have e_569_11 := arithEq_of_rows h (row := 569) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_11
  simp only [← c36165, ← c36166, k2, ← c36167] at e_569_11
  have hr := e_569_11
  simp only [e_569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2821)) := by
  have c36168 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.1
  have c36169 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.1
  have c36170 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36171 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36172 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36173 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_12 := arithEq_of_rows h (row := 569) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_12
  simp only [← c36168, ← c36169, ← c36170] at e_569_12
  have e_569_13 := arithEq_of_rows h (row := 569) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_13
  simp only [← c36171, ← c36172, k2, ← c36173] at e_569_13
  have hr := e_569_13
  simp only [e_569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2822)) := by
  have c36174 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36175 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36176 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36177 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36178 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36179 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_14 := arithEq_of_rows h (row := 569) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_14
  simp only [← c36174, ← c36175, ← c36176] at e_569_14
  have e_569_15 := arithEq_of_rows h (row := 569) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_15
  simp only [← c36177, ← c36178, k2, ← c36179] at e_569_15
  have hr := e_569_15
  simp only [e_569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2823)) := by
  have c36180 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36181 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36182 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36183 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36184 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36185 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_16 := arithEq_of_rows h (row := 569) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_16
  simp only [← c36180, ← c36181, ← c36182] at e_569_16
  have e_569_17 := arithEq_of_rows h (row := 569) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_17
  simp only [← c36183, ← c36184, k2, ← c36185] at e_569_17
  have hr := e_569_17
  simp only [e_569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 569 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2824)) := by
  have c36186 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36187 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36188 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36189 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36190 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36191 := (publicBatchWrapper89_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_569_18 := arithEq_of_rows h (row := 569) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_18
  simp only [← c36186, ← c36187, ← c36188] at e_569_18
  have e_569_19 := arithEq_of_rows h (row := 569) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_19
  simp only [← c36189, ← c36190, k2, ← c36191] at e_569_19
  have hr := e_569_19
  simp only [e_569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2825)) := by
  have c36192 := (publicBatchWrapper89_copies1131 a h).1
  have c36193 := (publicBatchWrapper89_copies1131 a h).2.1
  have c36194 := (publicBatchWrapper89_copies1131 a h).2.2.1
  have c36195 := (publicBatchWrapper89_copies1131 a h).2.2.2.1
  have c36196 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.1
  have c36197 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_0 := arithEq_of_rows h (row := 570) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_0
  simp only [← c36192, ← c36193, ← c36194] at e_570_0
  have e_570_1 := arithEq_of_rows h (row := 570) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_1
  simp only [← c36195, ← c36196, k2, ← c36197] at e_570_1
  have hr := e_570_1
  simp only [e_570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2826)) := by
  have c36198 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.1
  have c36199 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.1
  have c36200 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.1
  have c36201 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.1
  have c36202 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36203 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_2 := arithEq_of_rows h (row := 570) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_2
  simp only [← c36198, ← c36199, ← c36200] at e_570_2
  have e_570_3 := arithEq_of_rows h (row := 570) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_3
  simp only [← c36201, ← c36202, k2, ← c36203] at e_570_3
  have hr := e_570_3
  simp only [e_570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2827)) := by
  have c36204 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36205 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36206 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36207 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36208 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36209 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_4 := arithEq_of_rows h (row := 570) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_4
  simp only [← c36204, ← c36205, ← c36206] at e_570_4
  have e_570_5 := arithEq_of_rows h (row := 570) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_5
  simp only [← c36207, ← c36208, k2, ← c36209] at e_570_5
  have hr := e_570_5
  simp only [e_570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2828)) := by
  have c36210 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36211 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36212 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36213 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36214 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36215 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_6 := arithEq_of_rows h (row := 570) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_6
  simp only [← c36210, ← c36211, ← c36212] at e_570_6
  have e_570_7 := arithEq_of_rows h (row := 570) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_7
  simp only [← c36213, ← c36214, k2, ← c36215] at e_570_7
  have hr := e_570_7
  simp only [e_570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2829)) := by
  have c36216 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36217 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36218 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36219 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36220 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36221 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_8 := arithEq_of_rows h (row := 570) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_8
  simp only [← c36216, ← c36217, ← c36218] at e_570_8
  have e_570_9 := arithEq_of_rows h (row := 570) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_9
  simp only [← c36219, ← c36220, k2, ← c36221] at e_570_9
  have hr := e_570_9
  simp only [e_570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2830)) := by
  have c36222 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36223 := (publicBatchWrapper89_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36224 := (publicBatchWrapper89_copies1132 a h).1
  have c36225 := (publicBatchWrapper89_copies1132 a h).2.1
  have c36226 := (publicBatchWrapper89_copies1132 a h).2.2.1
  have c36227 := (publicBatchWrapper89_copies1132 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_10 := arithEq_of_rows h (row := 570) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_10
  simp only [← c36222, ← c36223, ← c36224] at e_570_10
  have e_570_11 := arithEq_of_rows h (row := 570) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_11
  simp only [← c36225, ← c36226, k2, ← c36227] at e_570_11
  have hr := e_570_11
  simp only [e_570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2831)) := by
  have c36228 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.1
  have c36229 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.1
  have c36230 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.1
  have c36231 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.1
  have c36232 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.1
  have c36233 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_12 := arithEq_of_rows h (row := 570) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_12
  simp only [← c36228, ← c36229, ← c36230] at e_570_12
  have e_570_13 := arithEq_of_rows h (row := 570) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_13
  simp only [← c36231, ← c36232, k2, ← c36233] at e_570_13
  have hr := e_570_13
  simp only [e_570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2832)) := by
  have c36234 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36235 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36236 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36237 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36238 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36239 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_14 := arithEq_of_rows h (row := 570) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_14
  simp only [← c36234, ← c36235, ← c36236] at e_570_14
  have e_570_15 := arithEq_of_rows h (row := 570) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_15
  simp only [← c36237, ← c36238, k2, ← c36239] at e_570_15
  have hr := e_570_15
  simp only [e_570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2833)) := by
  have c36240 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36241 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36242 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36243 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36244 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36245 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_16 := arithEq_of_rows h (row := 570) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_16
  simp only [← c36240, ← c36241, ← c36242] at e_570_16
  have e_570_17 := arithEq_of_rows h (row := 570) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_17
  simp only [← c36243, ← c36244, k2, ← c36245] at e_570_17
  have hr := e_570_17
  simp only [e_570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 570 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2834)) := by
  have c36246 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36247 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36248 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36249 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36250 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36251 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_570_18 := arithEq_of_rows h (row := 570) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_18
  simp only [← c36246, ← c36247, ← c36248] at e_570_18
  have e_570_19 := arithEq_of_rows h (row := 570) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_19
  simp only [← c36249, ← c36250, k2, ← c36251] at e_570_19
  have hr := e_570_19
  simp only [e_570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2835)) := by
  have c36252 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36253 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36254 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36255 := (publicBatchWrapper89_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36256 := (publicBatchWrapper89_copies1133 a h).1
  have c36257 := (publicBatchWrapper89_copies1133 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_0 := arithEq_of_rows h (row := 571) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_0
  simp only [← c36252, ← c36253, ← c36254] at e_571_0
  have e_571_1 := arithEq_of_rows h (row := 571) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_1
  simp only [← c36255, ← c36256, k2, ← c36257] at e_571_1
  have hr := e_571_1
  simp only [e_571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2836)) := by
  have c36258 := (publicBatchWrapper89_copies1133 a h).2.2.1
  have c36259 := (publicBatchWrapper89_copies1133 a h).2.2.2.1
  have c36260 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.1
  have c36261 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.1
  have c36262 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.1
  have c36263 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_2 := arithEq_of_rows h (row := 571) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_2
  simp only [← c36258, ← c36259, ← c36260] at e_571_2
  have e_571_3 := arithEq_of_rows h (row := 571) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_3
  simp only [← c36261, ← c36262, k2, ← c36263] at e_571_3
  have hr := e_571_3
  simp only [e_571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2837)) := by
  have c36264 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.1
  have c36265 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.1
  have c36266 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36267 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36268 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36269 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_4 := arithEq_of_rows h (row := 571) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_4
  simp only [← c36264, ← c36265, ← c36266] at e_571_4
  have e_571_5 := arithEq_of_rows h (row := 571) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_5
  simp only [← c36267, ← c36268, k2, ← c36269] at e_571_5
  have hr := e_571_5
  simp only [e_571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2838)) := by
  have c36270 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36271 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36272 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36273 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36274 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36275 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_6 := arithEq_of_rows h (row := 571) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_6
  simp only [← c36270, ← c36271, ← c36272] at e_571_6
  have e_571_7 := arithEq_of_rows h (row := 571) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_7
  simp only [← c36273, ← c36274, k2, ← c36275] at e_571_7
  have hr := e_571_7
  simp only [e_571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2839)) := by
  have c36276 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36277 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36278 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36279 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36280 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36281 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_8 := arithEq_of_rows h (row := 571) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_8
  simp only [← c36276, ← c36277, ← c36278] at e_571_8
  have e_571_9 := arithEq_of_rows h (row := 571) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_9
  simp only [← c36279, ← c36280, k2, ← c36281] at e_571_9
  have hr := e_571_9
  simp only [e_571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2840)) := by
  have c36282 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36283 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36284 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36285 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36286 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36287 := (publicBatchWrapper89_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_10 := arithEq_of_rows h (row := 571) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_10
  simp only [← c36282, ← c36283, ← c36284] at e_571_10
  have e_571_11 := arithEq_of_rows h (row := 571) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_11
  simp only [← c36285, ← c36286, k2, ← c36287] at e_571_11
  have hr := e_571_11
  simp only [e_571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2841)) := by
  have c36288 := (publicBatchWrapper89_copies1134 a h).1
  have c36289 := (publicBatchWrapper89_copies1134 a h).2.1
  have c36290 := (publicBatchWrapper89_copies1134 a h).2.2.1
  have c36291 := (publicBatchWrapper89_copies1134 a h).2.2.2.1
  have c36292 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.1
  have c36293 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_12 := arithEq_of_rows h (row := 571) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_12
  simp only [← c36288, ← c36289, ← c36290] at e_571_12
  have e_571_13 := arithEq_of_rows h (row := 571) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_13
  simp only [← c36291, ← c36292, k2, ← c36293] at e_571_13
  have hr := e_571_13
  simp only [e_571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2842)) := by
  have c36294 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.1
  have c36295 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.1
  have c36296 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.1
  have c36297 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.1
  have c36298 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36299 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_14 := arithEq_of_rows h (row := 571) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_14
  simp only [← c36294, ← c36295, ← c36296] at e_571_14
  have e_571_15 := arithEq_of_rows h (row := 571) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_15
  simp only [← c36297, ← c36298, k2, ← c36299] at e_571_15
  have hr := e_571_15
  simp only [e_571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2843)) := by
  have c36300 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36301 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36302 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36303 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36304 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36305 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_16 := arithEq_of_rows h (row := 571) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_16
  simp only [← c36300, ← c36301, ← c36302] at e_571_16
  have e_571_17 := arithEq_of_rows h (row := 571) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_17
  simp only [← c36303, ← c36304, k2, ← c36305] at e_571_17
  have hr := e_571_17
  simp only [e_571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 571 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2844)) := by
  have c36306 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36307 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36308 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36309 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36310 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36311 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_571_18 := arithEq_of_rows h (row := 571) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_18
  simp only [← c36306, ← c36307, ← c36308] at e_571_18
  have e_571_19 := arithEq_of_rows h (row := 571) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_19
  simp only [← c36309, ← c36310, k2, ← c36311] at e_571_19
  have hr := e_571_19
  simp only [e_571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2845)) := by
  have c36312 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36313 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36314 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36315 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36316 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36317 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_0 := arithEq_of_rows h (row := 572) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_0
  simp only [← c36312, ← c36313, ← c36314] at e_572_0
  have e_572_1 := arithEq_of_rows h (row := 572) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_1
  simp only [← c36315, ← c36316, k2, ← c36317] at e_572_1
  have hr := e_572_1
  simp only [e_572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2846)) := by
  have c36318 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36319 := (publicBatchWrapper89_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36320 := (publicBatchWrapper89_copies1135 a h).1
  have c36321 := (publicBatchWrapper89_copies1135 a h).2.1
  have c36322 := (publicBatchWrapper89_copies1135 a h).2.2.1
  have c36323 := (publicBatchWrapper89_copies1135 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_2 := arithEq_of_rows h (row := 572) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_2
  simp only [← c36318, ← c36319, ← c36320] at e_572_2
  have e_572_3 := arithEq_of_rows h (row := 572) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_3
  simp only [← c36321, ← c36322, k2, ← c36323] at e_572_3
  have hr := e_572_3
  simp only [e_572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2847)) := by
  have c36324 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.1
  have c36325 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.1
  have c36326 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.1
  have c36327 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.1
  have c36328 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.1
  have c36329 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_4 := arithEq_of_rows h (row := 572) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_4
  simp only [← c36324, ← c36325, ← c36326] at e_572_4
  have e_572_5 := arithEq_of_rows h (row := 572) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_5
  simp only [← c36327, ← c36328, k2, ← c36329] at e_572_5
  have hr := e_572_5
  simp only [e_572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2848)) := by
  have c36330 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36331 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36332 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36333 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36334 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36335 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_6 := arithEq_of_rows h (row := 572) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_6
  simp only [← c36330, ← c36331, ← c36332] at e_572_6
  have e_572_7 := arithEq_of_rows h (row := 572) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_7
  simp only [← c36333, ← c36334, k2, ← c36335] at e_572_7
  have hr := e_572_7
  simp only [e_572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2849)) := by
  have c36336 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36337 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36338 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36339 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36340 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36341 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_8 := arithEq_of_rows h (row := 572) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_8
  simp only [← c36336, ← c36337, ← c36338] at e_572_8
  have e_572_9 := arithEq_of_rows h (row := 572) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_9
  simp only [← c36339, ← c36340, k2, ← c36341] at e_572_9
  have hr := e_572_9
  simp only [e_572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2850)) := by
  have c36342 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36343 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36344 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36345 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36346 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36347 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_10 := arithEq_of_rows h (row := 572) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_10
  simp only [← c36342, ← c36343, ← c36344] at e_572_10
  have e_572_11 := arithEq_of_rows h (row := 572) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_11
  simp only [← c36345, ← c36346, k2, ← c36347] at e_572_11
  have hr := e_572_11
  simp only [e_572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2851)) := by
  have c36348 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36349 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36350 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36351 := (publicBatchWrapper89_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36352 := (publicBatchWrapper89_copies1136 a h).1
  have c36353 := (publicBatchWrapper89_copies1136 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_12 := arithEq_of_rows h (row := 572) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_12
  simp only [← c36348, ← c36349, ← c36350] at e_572_12
  have e_572_13 := arithEq_of_rows h (row := 572) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_13
  simp only [← c36351, ← c36352, k2, ← c36353] at e_572_13
  have hr := e_572_13
  simp only [e_572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2852)) := by
  have c36354 := (publicBatchWrapper89_copies1136 a h).2.2.1
  have c36355 := (publicBatchWrapper89_copies1136 a h).2.2.2.1
  have c36356 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.1
  have c36357 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.1
  have c36358 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.1
  have c36359 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_14 := arithEq_of_rows h (row := 572) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_14
  simp only [← c36354, ← c36355, ← c36356] at e_572_14
  have e_572_15 := arithEq_of_rows h (row := 572) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_15
  simp only [← c36357, ← c36358, k2, ← c36359] at e_572_15
  have hr := e_572_15
  simp only [e_572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2853)) := by
  have c36360 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.1
  have c36361 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.1
  have c36362 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36363 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36364 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36365 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_16 := arithEq_of_rows h (row := 572) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_16
  simp only [← c36360, ← c36361, ← c36362] at e_572_16
  have e_572_17 := arithEq_of_rows h (row := 572) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_17
  simp only [← c36363, ← c36364, k2, ← c36365] at e_572_17
  have hr := e_572_17
  simp only [e_572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 572 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2854)) := by
  have c36366 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36367 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36368 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36369 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36370 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36371 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_572_18 := arithEq_of_rows h (row := 572) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_18
  simp only [← c36366, ← c36367, ← c36368] at e_572_18
  have e_572_19 := arithEq_of_rows h (row := 572) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_19
  simp only [← c36369, ← c36370, k2, ← c36371] at e_572_19
  have hr := e_572_19
  simp only [e_572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2855)) := by
  have c36372 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36373 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36374 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36375 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36376 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36377 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_0 := arithEq_of_rows h (row := 573) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_0
  simp only [← c36372, ← c36373, ← c36374] at e_573_0
  have e_573_1 := arithEq_of_rows h (row := 573) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_1
  simp only [← c36375, ← c36376, k2, ← c36377] at e_573_1
  have hr := e_573_1
  simp only [e_573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2856)) := by
  have c36378 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36379 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36380 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36381 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36382 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36383 := (publicBatchWrapper89_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_2 := arithEq_of_rows h (row := 573) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_2
  simp only [← c36378, ← c36379, ← c36380] at e_573_2
  have e_573_3 := arithEq_of_rows h (row := 573) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_3
  simp only [← c36381, ← c36382, k2, ← c36383] at e_573_3
  have hr := e_573_3
  simp only [e_573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2857)) := by
  have c36384 := (publicBatchWrapper89_copies1137 a h).1
  have c36385 := (publicBatchWrapper89_copies1137 a h).2.1
  have c36386 := (publicBatchWrapper89_copies1137 a h).2.2.1
  have c36387 := (publicBatchWrapper89_copies1137 a h).2.2.2.1
  have c36388 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.1
  have c36389 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_4 := arithEq_of_rows h (row := 573) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_4
  simp only [← c36384, ← c36385, ← c36386] at e_573_4
  have e_573_5 := arithEq_of_rows h (row := 573) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_5
  simp only [← c36387, ← c36388, k2, ← c36389] at e_573_5
  have hr := e_573_5
  simp only [e_573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2858)) := by
  have c36390 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.1
  have c36391 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.1
  have c36392 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.1
  have c36393 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.1
  have c36394 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36395 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_6 := arithEq_of_rows h (row := 573) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_6
  simp only [← c36390, ← c36391, ← c36392] at e_573_6
  have e_573_7 := arithEq_of_rows h (row := 573) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_7
  simp only [← c36393, ← c36394, k2, ← c36395] at e_573_7
  have hr := e_573_7
  simp only [e_573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2859)) := by
  have c36396 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36397 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36398 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36399 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36400 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36401 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_8 := arithEq_of_rows h (row := 573) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_8
  simp only [← c36396, ← c36397, ← c36398] at e_573_8
  have e_573_9 := arithEq_of_rows h (row := 573) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_9
  simp only [← c36399, ← c36400, k2, ← c36401] at e_573_9
  have hr := e_573_9
  simp only [e_573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2860)) := by
  have c36402 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36403 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36404 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36405 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36406 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36407 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_10 := arithEq_of_rows h (row := 573) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_10
  simp only [← c36402, ← c36403, ← c36404] at e_573_10
  have e_573_11 := arithEq_of_rows h (row := 573) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_11
  simp only [← c36405, ← c36406, k2, ← c36407] at e_573_11
  have hr := e_573_11
  simp only [e_573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2861)) := by
  have c36408 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36409 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36410 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36411 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36412 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36413 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_12 := arithEq_of_rows h (row := 573) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_12
  simp only [← c36408, ← c36409, ← c36410] at e_573_12
  have e_573_13 := arithEq_of_rows h (row := 573) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_13
  simp only [← c36411, ← c36412, k2, ← c36413] at e_573_13
  have hr := e_573_13
  simp only [e_573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2862)) := by
  have c36414 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36415 := (publicBatchWrapper89_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36416 := (publicBatchWrapper89_copies1138 a h).1
  have c36417 := (publicBatchWrapper89_copies1138 a h).2.1
  have c36418 := (publicBatchWrapper89_copies1138 a h).2.2.1
  have c36419 := (publicBatchWrapper89_copies1138 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_14 := arithEq_of_rows h (row := 573) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_14
  simp only [← c36414, ← c36415, ← c36416] at e_573_14
  have e_573_15 := arithEq_of_rows h (row := 573) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_15
  simp only [← c36417, ← c36418, k2, ← c36419] at e_573_15
  have hr := e_573_15
  simp only [e_573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2863)) := by
  have c36420 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.1
  have c36421 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.1
  have c36422 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.1
  have c36423 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.1
  have c36424 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.1
  have c36425 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_16 := arithEq_of_rows h (row := 573) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_16
  simp only [← c36420, ← c36421, ← c36422] at e_573_16
  have e_573_17 := arithEq_of_rows h (row := 573) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_17
  simp only [← c36423, ← c36424, k2, ← c36425] at e_573_17
  have hr := e_573_17
  simp only [e_573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 573 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2864)) := by
  have c36426 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36427 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36428 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36429 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36430 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36431 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_573_18 := arithEq_of_rows h (row := 573) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_18
  simp only [← c36426, ← c36427, ← c36428] at e_573_18
  have e_573_19 := arithEq_of_rows h (row := 573) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_19
  simp only [← c36429, ← c36430, k2, ← c36431] at e_573_19
  have hr := e_573_19
  simp only [e_573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2865)) := by
  have c36432 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36433 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36434 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36435 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36436 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36437 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_0 := arithEq_of_rows h (row := 574) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_0
  simp only [← c36432, ← c36433, ← c36434] at e_574_0
  have e_574_1 := arithEq_of_rows h (row := 574) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_1
  simp only [← c36435, ← c36436, k2, ← c36437] at e_574_1
  have hr := e_574_1
  simp only [e_574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2866)) := by
  have c36438 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36439 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36440 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36441 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36442 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36443 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_2 := arithEq_of_rows h (row := 574) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_2
  simp only [← c36438, ← c36439, ← c36440] at e_574_2
  have e_574_3 := arithEq_of_rows h (row := 574) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_3
  simp only [← c36441, ← c36442, k2, ← c36443] at e_574_3
  have hr := e_574_3
  simp only [e_574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2867)) := by
  have c36444 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36445 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36446 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36447 := (publicBatchWrapper89_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36448 := (publicBatchWrapper89_copies1139 a h).1
  have c36449 := (publicBatchWrapper89_copies1139 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_4 := arithEq_of_rows h (row := 574) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_4
  simp only [← c36444, ← c36445, ← c36446] at e_574_4
  have e_574_5 := arithEq_of_rows h (row := 574) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_5
  simp only [← c36447, ← c36448, k2, ← c36449] at e_574_5
  have hr := e_574_5
  simp only [e_574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2868)) := by
  have c36450 := (publicBatchWrapper89_copies1139 a h).2.2.1
  have c36451 := (publicBatchWrapper89_copies1139 a h).2.2.2.1
  have c36452 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.1
  have c36453 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.1
  have c36454 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.1
  have c36455 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_6 := arithEq_of_rows h (row := 574) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_6
  simp only [← c36450, ← c36451, ← c36452] at e_574_6
  have e_574_7 := arithEq_of_rows h (row := 574) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_7
  simp only [← c36453, ← c36454, k2, ← c36455] at e_574_7
  have hr := e_574_7
  simp only [e_574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2869)) := by
  have c36456 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.1
  have c36457 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.1
  have c36458 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36459 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36460 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36461 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_8 := arithEq_of_rows h (row := 574) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_8
  simp only [← c36456, ← c36457, ← c36458] at e_574_8
  have e_574_9 := arithEq_of_rows h (row := 574) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_9
  simp only [← c36459, ← c36460, k2, ← c36461] at e_574_9
  have hr := e_574_9
  simp only [e_574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2870)) := by
  have c36462 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36463 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36464 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36465 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36466 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36467 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_10 := arithEq_of_rows h (row := 574) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_10
  simp only [← c36462, ← c36463, ← c36464] at e_574_10
  have e_574_11 := arithEq_of_rows h (row := 574) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_11
  simp only [← c36465, ← c36466, k2, ← c36467] at e_574_11
  have hr := e_574_11
  simp only [e_574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2871)) := by
  have c36468 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36469 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36470 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36471 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36472 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36473 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_12 := arithEq_of_rows h (row := 574) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_12
  simp only [← c36468, ← c36469, ← c36470] at e_574_12
  have e_574_13 := arithEq_of_rows h (row := 574) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_13
  simp only [← c36471, ← c36472, k2, ← c36473] at e_574_13
  have hr := e_574_13
  simp only [e_574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2872)) := by
  have c36474 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36475 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36476 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36477 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36478 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36479 := (publicBatchWrapper89_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_14 := arithEq_of_rows h (row := 574) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_14
  simp only [← c36474, ← c36475, ← c36476] at e_574_14
  have e_574_15 := arithEq_of_rows h (row := 574) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_15
  simp only [← c36477, ← c36478, k2, ← c36479] at e_574_15
  have hr := e_574_15
  simp only [e_574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2873)) := by
  have c36480 := (publicBatchWrapper89_copies1140 a h).1
  have c36481 := (publicBatchWrapper89_copies1140 a h).2.1
  have c36482 := (publicBatchWrapper89_copies1140 a h).2.2.1
  have c36483 := (publicBatchWrapper89_copies1140 a h).2.2.2.1
  have c36484 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.1
  have c36485 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_16 := arithEq_of_rows h (row := 574) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_16
  simp only [← c36480, ← c36481, ← c36482] at e_574_16
  have e_574_17 := arithEq_of_rows h (row := 574) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_17
  simp only [← c36483, ← c36484, k2, ← c36485] at e_574_17
  have hr := e_574_17
  simp only [e_574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 574 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2874)) := by
  have c36486 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.1
  have c36487 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.1
  have c36488 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.1
  have c36489 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.1
  have c36490 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36491 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_574_18 := arithEq_of_rows h (row := 574) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_18
  simp only [← c36486, ← c36487, ← c36488] at e_574_18
  have e_574_19 := arithEq_of_rows h (row := 574) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_19
  simp only [← c36489, ← c36490, k2, ← c36491] at e_574_19
  have hr := e_574_19
  simp only [e_574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2875)) := by
  have c36492 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36493 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36494 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36495 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36496 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36497 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_0 := arithEq_of_rows h (row := 575) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_0
  simp only [← c36492, ← c36493, ← c36494] at e_575_0
  have e_575_1 := arithEq_of_rows h (row := 575) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_1
  simp only [← c36495, ← c36496, k2, ← c36497] at e_575_1
  have hr := e_575_1
  simp only [e_575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2876)) := by
  have c36498 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36499 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36500 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36501 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36502 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36503 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_2 := arithEq_of_rows h (row := 575) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_2
  simp only [← c36498, ← c36499, ← c36500] at e_575_2
  have e_575_3 := arithEq_of_rows h (row := 575) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_3
  simp only [← c36501, ← c36502, k2, ← c36503] at e_575_3
  have hr := e_575_3
  simp only [e_575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2877)) := by
  have c36504 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36505 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36506 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36507 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36508 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36509 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_4 := arithEq_of_rows h (row := 575) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_4
  simp only [← c36504, ← c36505, ← c36506] at e_575_4
  have e_575_5 := arithEq_of_rows h (row := 575) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_5
  simp only [← c36507, ← c36508, k2, ← c36509] at e_575_5
  have hr := e_575_5
  simp only [e_575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2878)) := by
  have c36510 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36511 := (publicBatchWrapper89_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36512 := (publicBatchWrapper89_copies1141 a h).1
  have c36513 := (publicBatchWrapper89_copies1141 a h).2.1
  have c36514 := (publicBatchWrapper89_copies1141 a h).2.2.1
  have c36515 := (publicBatchWrapper89_copies1141 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_6 := arithEq_of_rows h (row := 575) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_6
  simp only [← c36510, ← c36511, ← c36512] at e_575_6
  have e_575_7 := arithEq_of_rows h (row := 575) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_7
  simp only [← c36513, ← c36514, k2, ← c36515] at e_575_7
  have hr := e_575_7
  simp only [e_575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2879)) := by
  have c36516 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.1
  have c36517 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.1
  have c36518 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.1
  have c36519 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.1
  have c36520 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.1
  have c36521 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_8 := arithEq_of_rows h (row := 575) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_8
  simp only [← c36516, ← c36517, ← c36518] at e_575_8
  have e_575_9 := arithEq_of_rows h (row := 575) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_9
  simp only [← c36519, ← c36520, k2, ← c36521] at e_575_9
  have hr := e_575_9
  simp only [e_575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2880)) := by
  have c36522 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36523 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36524 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36525 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36526 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36527 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_10 := arithEq_of_rows h (row := 575) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_10
  simp only [← c36522, ← c36523, ← c36524] at e_575_10
  have e_575_11 := arithEq_of_rows h (row := 575) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_11
  simp only [← c36525, ← c36526, k2, ← c36527] at e_575_11
  have hr := e_575_11
  simp only [e_575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2881)) := by
  have c36528 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36529 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36530 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36531 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36532 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36533 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_12 := arithEq_of_rows h (row := 575) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_12
  simp only [← c36528, ← c36529, ← c36530] at e_575_12
  have e_575_13 := arithEq_of_rows h (row := 575) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_13
  simp only [← c36531, ← c36532, k2, ← c36533] at e_575_13
  have hr := e_575_13
  simp only [e_575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2882)) := by
  have c36534 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36535 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36536 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36537 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36538 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36539 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_14 := arithEq_of_rows h (row := 575) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_14
  simp only [← c36534, ← c36535, ← c36536] at e_575_14
  have e_575_15 := arithEq_of_rows h (row := 575) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_15
  simp only [← c36537, ← c36538, k2, ← c36539] at e_575_15
  have hr := e_575_15
  simp only [e_575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2883)) := by
  have c36540 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36541 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36542 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36543 := (publicBatchWrapper89_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36544 := (publicBatchWrapper89_copies1142 a h).1
  have c36545 := (publicBatchWrapper89_copies1142 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_16 := arithEq_of_rows h (row := 575) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_16
  simp only [← c36540, ← c36541, ← c36542] at e_575_16
  have e_575_17 := arithEq_of_rows h (row := 575) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_17
  simp only [← c36543, ← c36544, k2, ← c36545] at e_575_17
  have hr := e_575_17
  simp only [e_575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 575 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2884)) := by
  have c36546 := (publicBatchWrapper89_copies1142 a h).2.2.1
  have c36547 := (publicBatchWrapper89_copies1142 a h).2.2.2.1
  have c36548 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.1
  have c36549 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.1
  have c36550 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.1
  have c36551 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_575_18 := arithEq_of_rows h (row := 575) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_18
  simp only [← c36546, ← c36547, ← c36548] at e_575_18
  have e_575_19 := arithEq_of_rows h (row := 575) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_19
  simp only [← c36549, ← c36550, k2, ← c36551] at e_575_19
  have hr := e_575_19
  simp only [e_575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2885)) := by
  have c36552 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.1
  have c36553 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.1
  have c36554 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36555 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36556 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36557 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_0 := arithEq_of_rows h (row := 576) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_0
  simp only [← c36552, ← c36553, ← c36554] at e_576_0
  have e_576_1 := arithEq_of_rows h (row := 576) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_1
  simp only [← c36555, ← c36556, k2, ← c36557] at e_576_1
  have hr := e_576_1
  simp only [e_576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2886)) := by
  have c36558 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36559 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36560 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36561 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36562 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36563 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_2 := arithEq_of_rows h (row := 576) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_2
  simp only [← c36558, ← c36559, ← c36560] at e_576_2
  have e_576_3 := arithEq_of_rows h (row := 576) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_3
  simp only [← c36561, ← c36562, k2, ← c36563] at e_576_3
  have hr := e_576_3
  simp only [e_576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2887)) := by
  have c36564 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36565 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36566 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36567 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36568 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36569 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_4 := arithEq_of_rows h (row := 576) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_4
  simp only [← c36564, ← c36565, ← c36566] at e_576_4
  have e_576_5 := arithEq_of_rows h (row := 576) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_5
  simp only [← c36567, ← c36568, k2, ← c36569] at e_576_5
  have hr := e_576_5
  simp only [e_576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2888)) := by
  have c36570 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36571 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36572 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36573 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36574 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36575 := (publicBatchWrapper89_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_6 := arithEq_of_rows h (row := 576) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_6
  simp only [← c36570, ← c36571, ← c36572] at e_576_6
  have e_576_7 := arithEq_of_rows h (row := 576) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_7
  simp only [← c36573, ← c36574, k2, ← c36575] at e_576_7
  have hr := e_576_7
  simp only [e_576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2889)) := by
  have c36576 := (publicBatchWrapper89_copies1143 a h).1
  have c36577 := (publicBatchWrapper89_copies1143 a h).2.1
  have c36578 := (publicBatchWrapper89_copies1143 a h).2.2.1
  have c36579 := (publicBatchWrapper89_copies1143 a h).2.2.2.1
  have c36580 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.1
  have c36581 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_8 := arithEq_of_rows h (row := 576) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_8
  simp only [← c36576, ← c36577, ← c36578] at e_576_8
  have e_576_9 := arithEq_of_rows h (row := 576) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_9
  simp only [← c36579, ← c36580, k2, ← c36581] at e_576_9
  have hr := e_576_9
  simp only [e_576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2890)) := by
  have c36582 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.1
  have c36583 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.1
  have c36584 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.1
  have c36585 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.1
  have c36586 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36587 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_10 := arithEq_of_rows h (row := 576) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_10
  simp only [← c36582, ← c36583, ← c36584] at e_576_10
  have e_576_11 := arithEq_of_rows h (row := 576) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_11
  simp only [← c36585, ← c36586, k2, ← c36587] at e_576_11
  have hr := e_576_11
  simp only [e_576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2891)) := by
  have c36588 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36589 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36590 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36591 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36592 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36593 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_12 := arithEq_of_rows h (row := 576) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_12
  simp only [← c36588, ← c36589, ← c36590] at e_576_12
  have e_576_13 := arithEq_of_rows h (row := 576) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_13
  simp only [← c36591, ← c36592, k2, ← c36593] at e_576_13
  have hr := e_576_13
  simp only [e_576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2892)) := by
  have c36594 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36595 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36596 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36597 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36598 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36599 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_14 := arithEq_of_rows h (row := 576) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_14
  simp only [← c36594, ← c36595, ← c36596] at e_576_14
  have e_576_15 := arithEq_of_rows h (row := 576) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_15
  simp only [← c36597, ← c36598, k2, ← c36599] at e_576_15
  have hr := e_576_15
  simp only [e_576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2893)) := by
  have c36600 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36601 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36602 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36603 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36604 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36605 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_16 := arithEq_of_rows h (row := 576) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_16
  simp only [← c36600, ← c36601, ← c36602] at e_576_16
  have e_576_17 := arithEq_of_rows h (row := 576) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_17
  simp only [← c36603, ← c36604, k2, ← c36605] at e_576_17
  have hr := e_576_17
  simp only [e_576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 576 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2894)) := by
  have c36606 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36607 := (publicBatchWrapper89_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36608 := (publicBatchWrapper89_copies1144 a h).1
  have c36609 := (publicBatchWrapper89_copies1144 a h).2.1
  have c36610 := (publicBatchWrapper89_copies1144 a h).2.2.1
  have c36611 := (publicBatchWrapper89_copies1144 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_576_18 := arithEq_of_rows h (row := 576) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_18
  simp only [← c36606, ← c36607, ← c36608] at e_576_18
  have e_576_19 := arithEq_of_rows h (row := 576) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_19
  simp only [← c36609, ← c36610, k2, ← c36611] at e_576_19
  have hr := e_576_19
  simp only [e_576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2895)) := by
  have c36612 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.1
  have c36613 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.1
  have c36614 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.1
  have c36615 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.1
  have c36616 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.1
  have c36617 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_0 := arithEq_of_rows h (row := 577) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_0
  simp only [← c36612, ← c36613, ← c36614] at e_577_0
  have e_577_1 := arithEq_of_rows h (row := 577) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_1
  simp only [← c36615, ← c36616, k2, ← c36617] at e_577_1
  have hr := e_577_1
  simp only [e_577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2896)) := by
  have c36618 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36619 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36620 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36621 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36622 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36623 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_2 := arithEq_of_rows h (row := 577) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_2
  simp only [← c36618, ← c36619, ← c36620] at e_577_2
  have e_577_3 := arithEq_of_rows h (row := 577) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_3
  simp only [← c36621, ← c36622, k2, ← c36623] at e_577_3
  have hr := e_577_3
  simp only [e_577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2897)) := by
  have c36624 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36625 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36626 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36627 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36628 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36629 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_4 := arithEq_of_rows h (row := 577) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_4
  simp only [← c36624, ← c36625, ← c36626] at e_577_4
  have e_577_5 := arithEq_of_rows h (row := 577) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_5
  simp only [← c36627, ← c36628, k2, ← c36629] at e_577_5
  have hr := e_577_5
  simp only [e_577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2898)) := by
  have c36630 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36631 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36632 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36633 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36634 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36635 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_6 := arithEq_of_rows h (row := 577) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_6
  simp only [← c36630, ← c36631, ← c36632] at e_577_6
  have e_577_7 := arithEq_of_rows h (row := 577) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_7
  simp only [← c36633, ← c36634, k2, ← c36635] at e_577_7
  have hr := e_577_7
  simp only [e_577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2899)) := by
  have c36636 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36637 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36638 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36639 := (publicBatchWrapper89_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36640 := (publicBatchWrapper89_copies1145 a h).1
  have c36641 := (publicBatchWrapper89_copies1145 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_8 := arithEq_of_rows h (row := 577) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_8
  simp only [← c36636, ← c36637, ← c36638] at e_577_8
  have e_577_9 := arithEq_of_rows h (row := 577) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_9
  simp only [← c36639, ← c36640, k2, ← c36641] at e_577_9
  have hr := e_577_9
  simp only [e_577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2900)) := by
  have c36642 := (publicBatchWrapper89_copies1145 a h).2.2.1
  have c36643 := (publicBatchWrapper89_copies1145 a h).2.2.2.1
  have c36644 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.1
  have c36645 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.1
  have c36646 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.1
  have c36647 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_10 := arithEq_of_rows h (row := 577) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_10
  simp only [← c36642, ← c36643, ← c36644] at e_577_10
  have e_577_11 := arithEq_of_rows h (row := 577) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_11
  simp only [← c36645, ← c36646, k2, ← c36647] at e_577_11
  have hr := e_577_11
  simp only [e_577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2901)) := by
  have c36648 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.1
  have c36649 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.1
  have c36650 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36651 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36652 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36653 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_12 := arithEq_of_rows h (row := 577) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_12
  simp only [← c36648, ← c36649, ← c36650] at e_577_12
  have e_577_13 := arithEq_of_rows h (row := 577) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_13
  simp only [← c36651, ← c36652, k2, ← c36653] at e_577_13
  have hr := e_577_13
  simp only [e_577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2902)) := by
  have c36654 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36655 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36656 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36657 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36658 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36659 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_14 := arithEq_of_rows h (row := 577) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_14
  simp only [← c36654, ← c36655, ← c36656] at e_577_14
  have e_577_15 := arithEq_of_rows h (row := 577) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_15
  simp only [← c36657, ← c36658, k2, ← c36659] at e_577_15
  have hr := e_577_15
  simp only [e_577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2903)) := by
  have c36660 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36661 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36662 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36663 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36664 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36665 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_16 := arithEq_of_rows h (row := 577) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_16
  simp only [← c36660, ← c36661, ← c36662] at e_577_16
  have e_577_17 := arithEq_of_rows h (row := 577) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_17
  simp only [← c36663, ← c36664, k2, ← c36665] at e_577_17
  have hr := e_577_17
  simp only [e_577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 577 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2904)) := by
  have c36666 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36667 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36668 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36669 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36670 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36671 := (publicBatchWrapper89_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_577_18 := arithEq_of_rows h (row := 577) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_18
  simp only [← c36666, ← c36667, ← c36668] at e_577_18
  have e_577_19 := arithEq_of_rows h (row := 577) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_19
  simp only [← c36669, ← c36670, k2, ← c36671] at e_577_19
  have hr := e_577_19
  simp only [e_577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2905)) := by
  have c36672 := (publicBatchWrapper89_copies1146 a h).1
  have c36673 := (publicBatchWrapper89_copies1146 a h).2.1
  have c36674 := (publicBatchWrapper89_copies1146 a h).2.2.1
  have c36675 := (publicBatchWrapper89_copies1146 a h).2.2.2.1
  have c36676 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.1
  have c36677 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_0 := arithEq_of_rows h (row := 578) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_0
  simp only [← c36672, ← c36673, ← c36674] at e_578_0
  have e_578_1 := arithEq_of_rows h (row := 578) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_1
  simp only [← c36675, ← c36676, k2, ← c36677] at e_578_1
  have hr := e_578_1
  simp only [e_578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2906)) := by
  have c36678 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.1
  have c36679 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.1
  have c36680 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.1
  have c36681 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.1
  have c36682 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36683 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_2 := arithEq_of_rows h (row := 578) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_2
  simp only [← c36678, ← c36679, ← c36680] at e_578_2
  have e_578_3 := arithEq_of_rows h (row := 578) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_3
  simp only [← c36681, ← c36682, k2, ← c36683] at e_578_3
  have hr := e_578_3
  simp only [e_578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2907)) := by
  have c36684 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36685 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36686 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36687 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36688 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36689 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_4 := arithEq_of_rows h (row := 578) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_4
  simp only [← c36684, ← c36685, ← c36686] at e_578_4
  have e_578_5 := arithEq_of_rows h (row := 578) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_5
  simp only [← c36687, ← c36688, k2, ← c36689] at e_578_5
  have hr := e_578_5
  simp only [e_578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2908)) := by
  have c36690 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36691 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36692 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36693 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36694 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36695 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_6 := arithEq_of_rows h (row := 578) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_6
  simp only [← c36690, ← c36691, ← c36692] at e_578_6
  have e_578_7 := arithEq_of_rows h (row := 578) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_7
  simp only [← c36693, ← c36694, k2, ← c36695] at e_578_7
  have hr := e_578_7
  simp only [e_578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2909)) := by
  have c36696 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36697 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36698 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36699 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36700 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36701 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_8 := arithEq_of_rows h (row := 578) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_8
  simp only [← c36696, ← c36697, ← c36698] at e_578_8
  have e_578_9 := arithEq_of_rows h (row := 578) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_9
  simp only [← c36699, ← c36700, k2, ← c36701] at e_578_9
  have hr := e_578_9
  simp only [e_578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2910)) := by
  have c36702 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36703 := (publicBatchWrapper89_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36704 := (publicBatchWrapper89_copies1147 a h).1
  have c36705 := (publicBatchWrapper89_copies1147 a h).2.1
  have c36706 := (publicBatchWrapper89_copies1147 a h).2.2.1
  have c36707 := (publicBatchWrapper89_copies1147 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_10 := arithEq_of_rows h (row := 578) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_10
  simp only [← c36702, ← c36703, ← c36704] at e_578_10
  have e_578_11 := arithEq_of_rows h (row := 578) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_11
  simp only [← c36705, ← c36706, k2, ← c36707] at e_578_11
  have hr := e_578_11
  simp only [e_578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2911)) := by
  have c36708 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.1
  have c36709 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.1
  have c36710 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.1
  have c36711 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.1
  have c36712 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.1
  have c36713 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_12 := arithEq_of_rows h (row := 578) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_12
  simp only [← c36708, ← c36709, ← c36710] at e_578_12
  have e_578_13 := arithEq_of_rows h (row := 578) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_13
  simp only [← c36711, ← c36712, k2, ← c36713] at e_578_13
  have hr := e_578_13
  simp only [e_578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2912)) := by
  have c36714 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36715 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36716 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36717 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36718 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36719 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_14 := arithEq_of_rows h (row := 578) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_14
  simp only [← c36714, ← c36715, ← c36716] at e_578_14
  have e_578_15 := arithEq_of_rows h (row := 578) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_15
  simp only [← c36717, ← c36718, k2, ← c36719] at e_578_15
  have hr := e_578_15
  simp only [e_578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2913)) := by
  have c36720 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36721 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36722 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36723 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36724 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36725 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_16 := arithEq_of_rows h (row := 578) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_16
  simp only [← c36720, ← c36721, ← c36722] at e_578_16
  have e_578_17 := arithEq_of_rows h (row := 578) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_17
  simp only [← c36723, ← c36724, k2, ← c36725] at e_578_17
  have hr := e_578_17
  simp only [e_578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 578 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2914)) := by
  have c36726 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36727 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36728 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36729 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36730 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36731 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_578_18 := arithEq_of_rows h (row := 578) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_18
  simp only [← c36726, ← c36727, ← c36728] at e_578_18
  have e_578_19 := arithEq_of_rows h (row := 578) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_19
  simp only [← c36729, ← c36730, k2, ← c36731] at e_578_19
  have hr := e_578_19
  simp only [e_578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2915)) := by
  have c36732 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36733 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36734 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36735 := (publicBatchWrapper89_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36736 := (publicBatchWrapper89_copies1148 a h).1
  have c36737 := (publicBatchWrapper89_copies1148 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_0 := arithEq_of_rows h (row := 579) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_0
  simp only [← c36732, ← c36733, ← c36734] at e_579_0
  have e_579_1 := arithEq_of_rows h (row := 579) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_1
  simp only [← c36735, ← c36736, k2, ← c36737] at e_579_1
  have hr := e_579_1
  simp only [e_579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2916)) := by
  have c36738 := (publicBatchWrapper89_copies1148 a h).2.2.1
  have c36739 := (publicBatchWrapper89_copies1148 a h).2.2.2.1
  have c36740 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.1
  have c36741 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.1
  have c36742 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.1
  have c36743 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_2 := arithEq_of_rows h (row := 579) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_2
  simp only [← c36738, ← c36739, ← c36740] at e_579_2
  have e_579_3 := arithEq_of_rows h (row := 579) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_3
  simp only [← c36741, ← c36742, k2, ← c36743] at e_579_3
  have hr := e_579_3
  simp only [e_579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2917)) := by
  have c36744 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.1
  have c36745 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.1
  have c36746 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36747 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36748 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36749 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_4 := arithEq_of_rows h (row := 579) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_4
  simp only [← c36744, ← c36745, ← c36746] at e_579_4
  have e_579_5 := arithEq_of_rows h (row := 579) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_5
  simp only [← c36747, ← c36748, k2, ← c36749] at e_579_5
  have hr := e_579_5
  simp only [e_579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2918)) := by
  have c36750 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36751 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36752 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36753 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36754 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36755 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_6 := arithEq_of_rows h (row := 579) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_6
  simp only [← c36750, ← c36751, ← c36752] at e_579_6
  have e_579_7 := arithEq_of_rows h (row := 579) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_7
  simp only [← c36753, ← c36754, k2, ← c36755] at e_579_7
  have hr := e_579_7
  simp only [e_579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2919)) := by
  have c36756 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36757 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36758 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36759 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36760 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36761 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_8 := arithEq_of_rows h (row := 579) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_8
  simp only [← c36756, ← c36757, ← c36758] at e_579_8
  have e_579_9 := arithEq_of_rows h (row := 579) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_9
  simp only [← c36759, ← c36760, k2, ← c36761] at e_579_9
  have hr := e_579_9
  simp only [e_579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2920)) := by
  have c36762 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36763 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36764 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36765 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36766 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36767 := (publicBatchWrapper89_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_10 := arithEq_of_rows h (row := 579) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_10
  simp only [← c36762, ← c36763, ← c36764] at e_579_10
  have e_579_11 := arithEq_of_rows h (row := 579) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_11
  simp only [← c36765, ← c36766, k2, ← c36767] at e_579_11
  have hr := e_579_11
  simp only [e_579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2921)) := by
  have c36768 := (publicBatchWrapper89_copies1149 a h).1
  have c36769 := (publicBatchWrapper89_copies1149 a h).2.1
  have c36770 := (publicBatchWrapper89_copies1149 a h).2.2.1
  have c36771 := (publicBatchWrapper89_copies1149 a h).2.2.2.1
  have c36772 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.1
  have c36773 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_12 := arithEq_of_rows h (row := 579) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_12
  simp only [← c36768, ← c36769, ← c36770] at e_579_12
  have e_579_13 := arithEq_of_rows h (row := 579) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_13
  simp only [← c36771, ← c36772, k2, ← c36773] at e_579_13
  have hr := e_579_13
  simp only [e_579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2922)) := by
  have c36774 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.1
  have c36775 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.1
  have c36776 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.1
  have c36777 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.1
  have c36778 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36779 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_14 := arithEq_of_rows h (row := 579) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_14
  simp only [← c36774, ← c36775, ← c36776] at e_579_14
  have e_579_15 := arithEq_of_rows h (row := 579) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_15
  simp only [← c36777, ← c36778, k2, ← c36779] at e_579_15
  have hr := e_579_15
  simp only [e_579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2923)) := by
  have c36780 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36781 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36782 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36783 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36784 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36785 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_16 := arithEq_of_rows h (row := 579) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_16
  simp only [← c36780, ← c36781, ← c36782] at e_579_16
  have e_579_17 := arithEq_of_rows h (row := 579) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_17
  simp only [← c36783, ← c36784, k2, ← c36785] at e_579_17
  have hr := e_579_17
  simp only [e_579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 579 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2924)) := by
  have c36786 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36787 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36788 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36789 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36790 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36791 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_579_18 := arithEq_of_rows h (row := 579) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_18
  simp only [← c36786, ← c36787, ← c36788] at e_579_18
  have e_579_19 := arithEq_of_rows h (row := 579) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_19
  simp only [← c36789, ← c36790, k2, ← c36791] at e_579_19
  have hr := e_579_19
  simp only [e_579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2925)) := by
  have c36792 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36793 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36794 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36795 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36796 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36797 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_0 := arithEq_of_rows h (row := 580) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_0
  simp only [← c36792, ← c36793, ← c36794] at e_580_0
  have e_580_1 := arithEq_of_rows h (row := 580) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_1
  simp only [← c36795, ← c36796, k2, ← c36797] at e_580_1
  have hr := e_580_1
  simp only [e_580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2926)) := by
  have c36798 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36799 := (publicBatchWrapper89_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36800 := (publicBatchWrapper89_copies1150 a h).1
  have c36801 := (publicBatchWrapper89_copies1150 a h).2.1
  have c36802 := (publicBatchWrapper89_copies1150 a h).2.2.1
  have c36803 := (publicBatchWrapper89_copies1150 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_2 := arithEq_of_rows h (row := 580) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_2
  simp only [← c36798, ← c36799, ← c36800] at e_580_2
  have e_580_3 := arithEq_of_rows h (row := 580) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_3
  simp only [← c36801, ← c36802, k2, ← c36803] at e_580_3
  have hr := e_580_3
  simp only [e_580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2927)) := by
  have c36804 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.1
  have c36805 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.1
  have c36806 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.1
  have c36807 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.1
  have c36808 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.1
  have c36809 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_4 := arithEq_of_rows h (row := 580) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_4
  simp only [← c36804, ← c36805, ← c36806] at e_580_4
  have e_580_5 := arithEq_of_rows h (row := 580) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_5
  simp only [← c36807, ← c36808, k2, ← c36809] at e_580_5
  have hr := e_580_5
  simp only [e_580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2928)) := by
  have c36810 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36811 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36812 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36813 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36814 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36815 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_6 := arithEq_of_rows h (row := 580) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_6
  simp only [← c36810, ← c36811, ← c36812] at e_580_6
  have e_580_7 := arithEq_of_rows h (row := 580) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_7
  simp only [← c36813, ← c36814, k2, ← c36815] at e_580_7
  have hr := e_580_7
  simp only [e_580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2929)) := by
  have c36816 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36817 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36818 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36819 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36820 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36821 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_8 := arithEq_of_rows h (row := 580) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_8
  simp only [← c36816, ← c36817, ← c36818] at e_580_8
  have e_580_9 := arithEq_of_rows h (row := 580) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_9
  simp only [← c36819, ← c36820, k2, ← c36821] at e_580_9
  have hr := e_580_9
  simp only [e_580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2930)) := by
  have c36822 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36823 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36824 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36825 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36826 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36827 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_10 := arithEq_of_rows h (row := 580) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_10
  simp only [← c36822, ← c36823, ← c36824] at e_580_10
  have e_580_11 := arithEq_of_rows h (row := 580) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_11
  simp only [← c36825, ← c36826, k2, ← c36827] at e_580_11
  have hr := e_580_11
  simp only [e_580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2931)) := by
  have c36828 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36829 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36830 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36831 := (publicBatchWrapper89_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36832 := (publicBatchWrapper89_copies1151 a h).1
  have c36833 := (publicBatchWrapper89_copies1151 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_12 := arithEq_of_rows h (row := 580) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_12
  simp only [← c36828, ← c36829, ← c36830] at e_580_12
  have e_580_13 := arithEq_of_rows h (row := 580) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_13
  simp only [← c36831, ← c36832, k2, ← c36833] at e_580_13
  have hr := e_580_13
  simp only [e_580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2932)) := by
  have c36834 := (publicBatchWrapper89_copies1151 a h).2.2.1
  have c36835 := (publicBatchWrapper89_copies1151 a h).2.2.2.1
  have c36836 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.1
  have c36837 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.1
  have c36838 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.1
  have c36839 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_14 := arithEq_of_rows h (row := 580) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_14
  simp only [← c36834, ← c36835, ← c36836] at e_580_14
  have e_580_15 := arithEq_of_rows h (row := 580) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_15
  simp only [← c36837, ← c36838, k2, ← c36839] at e_580_15
  have hr := e_580_15
  simp only [e_580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2933)) := by
  have c36840 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.1
  have c36841 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.1
  have c36842 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36843 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36844 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36845 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_16 := arithEq_of_rows h (row := 580) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_16
  simp only [← c36840, ← c36841, ← c36842] at e_580_16
  have e_580_17 := arithEq_of_rows h (row := 580) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_17
  simp only [← c36843, ← c36844, k2, ← c36845] at e_580_17
  have hr := e_580_17
  simp only [e_580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 580 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2934)) := by
  have c36846 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36847 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36848 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36849 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36850 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36851 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_580_18 := arithEq_of_rows h (row := 580) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_18
  simp only [← c36846, ← c36847, ← c36848] at e_580_18
  have e_580_19 := arithEq_of_rows h (row := 580) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_19
  simp only [← c36849, ← c36850, k2, ← c36851] at e_580_19
  have hr := e_580_19
  simp only [e_580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2935)) := by
  have c36852 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36853 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36854 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36855 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36856 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36857 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_0 := arithEq_of_rows h (row := 581) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_0
  simp only [← c36852, ← c36853, ← c36854] at e_581_0
  have e_581_1 := arithEq_of_rows h (row := 581) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_1
  simp only [← c36855, ← c36856, k2, ← c36857] at e_581_1
  have hr := e_581_1
  simp only [e_581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2936)) := by
  have c36858 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36859 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36860 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36861 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36862 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36863 := (publicBatchWrapper89_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_2 := arithEq_of_rows h (row := 581) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_2
  simp only [← c36858, ← c36859, ← c36860] at e_581_2
  have e_581_3 := arithEq_of_rows h (row := 581) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_3
  simp only [← c36861, ← c36862, k2, ← c36863] at e_581_3
  have hr := e_581_3
  simp only [e_581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2937)) := by
  have c36864 := (publicBatchWrapper89_copies1152 a h).1
  have c36865 := (publicBatchWrapper89_copies1152 a h).2.1
  have c36866 := (publicBatchWrapper89_copies1152 a h).2.2.1
  have c36867 := (publicBatchWrapper89_copies1152 a h).2.2.2.1
  have c36868 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.1
  have c36869 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_4 := arithEq_of_rows h (row := 581) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_4
  simp only [← c36864, ← c36865, ← c36866] at e_581_4
  have e_581_5 := arithEq_of_rows h (row := 581) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_5
  simp only [← c36867, ← c36868, k2, ← c36869] at e_581_5
  have hr := e_581_5
  simp only [e_581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2938)) := by
  have c36870 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.1
  have c36871 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.1
  have c36872 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.1
  have c36873 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.1
  have c36874 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36875 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_6 := arithEq_of_rows h (row := 581) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_6
  simp only [← c36870, ← c36871, ← c36872] at e_581_6
  have e_581_7 := arithEq_of_rows h (row := 581) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_7
  simp only [← c36873, ← c36874, k2, ← c36875] at e_581_7
  have hr := e_581_7
  simp only [e_581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2939)) := by
  have c36876 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36877 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36878 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36879 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36880 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36881 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_8 := arithEq_of_rows h (row := 581) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_8
  simp only [← c36876, ← c36877, ← c36878] at e_581_8
  have e_581_9 := arithEq_of_rows h (row := 581) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_9
  simp only [← c36879, ← c36880, k2, ← c36881] at e_581_9
  have hr := e_581_9
  simp only [e_581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2940)) := by
  have c36882 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36883 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36884 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36885 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36886 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36887 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_10 := arithEq_of_rows h (row := 581) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_10
  simp only [← c36882, ← c36883, ← c36884] at e_581_10
  have e_581_11 := arithEq_of_rows h (row := 581) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_11
  simp only [← c36885, ← c36886, k2, ← c36887] at e_581_11
  have hr := e_581_11
  simp only [e_581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2941)) := by
  have c36888 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36889 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36890 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36891 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36892 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36893 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_12 := arithEq_of_rows h (row := 581) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_12
  simp only [← c36888, ← c36889, ← c36890] at e_581_12
  have e_581_13 := arithEq_of_rows h (row := 581) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_13
  simp only [← c36891, ← c36892, k2, ← c36893] at e_581_13
  have hr := e_581_13
  simp only [e_581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2942)) := by
  have c36894 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36895 := (publicBatchWrapper89_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36896 := (publicBatchWrapper89_copies1153 a h).1
  have c36897 := (publicBatchWrapper89_copies1153 a h).2.1
  have c36898 := (publicBatchWrapper89_copies1153 a h).2.2.1
  have c36899 := (publicBatchWrapper89_copies1153 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_14 := arithEq_of_rows h (row := 581) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_14
  simp only [← c36894, ← c36895, ← c36896] at e_581_14
  have e_581_15 := arithEq_of_rows h (row := 581) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_15
  simp only [← c36897, ← c36898, k2, ← c36899] at e_581_15
  have hr := e_581_15
  simp only [e_581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2943)) := by
  have c36900 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.1
  have c36901 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.1
  have c36902 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.1
  have c36903 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.1
  have c36904 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.1
  have c36905 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_16 := arithEq_of_rows h (row := 581) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_16
  simp only [← c36900, ← c36901, ← c36902] at e_581_16
  have e_581_17 := arithEq_of_rows h (row := 581) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_17
  simp only [← c36903, ← c36904, k2, ← c36905] at e_581_17
  have hr := e_581_17
  simp only [e_581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 581 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2944)) := by
  have c36906 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36907 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36908 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36909 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36910 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36911 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_581_18 := arithEq_of_rows h (row := 581) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_18
  simp only [← c36906, ← c36907, ← c36908] at e_581_18
  have e_581_19 := arithEq_of_rows h (row := 581) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_19
  simp only [← c36909, ← c36910, k2, ← c36911] at e_581_19
  have hr := e_581_19
  simp only [e_581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2945)) := by
  have c36912 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36913 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36914 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36915 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36916 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36917 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_0 := arithEq_of_rows h (row := 582) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_0
  simp only [← c36912, ← c36913, ← c36914] at e_582_0
  have e_582_1 := arithEq_of_rows h (row := 582) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_1
  simp only [← c36915, ← c36916, k2, ← c36917] at e_582_1
  have hr := e_582_1
  simp only [e_582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2946)) := by
  have c36918 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36919 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36920 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36921 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36922 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36923 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_2 := arithEq_of_rows h (row := 582) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_2
  simp only [← c36918, ← c36919, ← c36920] at e_582_2
  have e_582_3 := arithEq_of_rows h (row := 582) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_3
  simp only [← c36921, ← c36922, k2, ← c36923] at e_582_3
  have hr := e_582_3
  simp only [e_582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2947)) := by
  have c36924 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36925 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36926 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36927 := (publicBatchWrapper89_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36928 := (publicBatchWrapper89_copies1154 a h).1
  have c36929 := (publicBatchWrapper89_copies1154 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_4 := arithEq_of_rows h (row := 582) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_4
  simp only [← c36924, ← c36925, ← c36926] at e_582_4
  have e_582_5 := arithEq_of_rows h (row := 582) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_5
  simp only [← c36927, ← c36928, k2, ← c36929] at e_582_5
  have hr := e_582_5
  simp only [e_582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2948)) := by
  have c36930 := (publicBatchWrapper89_copies1154 a h).2.2.1
  have c36931 := (publicBatchWrapper89_copies1154 a h).2.2.2.1
  have c36932 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.1
  have c36933 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.1
  have c36934 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.1
  have c36935 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_6 := arithEq_of_rows h (row := 582) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_6
  simp only [← c36930, ← c36931, ← c36932] at e_582_6
  have e_582_7 := arithEq_of_rows h (row := 582) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_7
  simp only [← c36933, ← c36934, k2, ← c36935] at e_582_7
  have hr := e_582_7
  simp only [e_582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2949)) := by
  have c36936 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.1
  have c36937 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.1
  have c36938 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36939 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36940 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36941 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_8 := arithEq_of_rows h (row := 582) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_8
  simp only [← c36936, ← c36937, ← c36938] at e_582_8
  have e_582_9 := arithEq_of_rows h (row := 582) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_9
  simp only [← c36939, ← c36940, k2, ← c36941] at e_582_9
  have hr := e_582_9
  simp only [e_582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2950)) := by
  have c36942 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36943 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36944 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36945 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36946 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36947 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_10 := arithEq_of_rows h (row := 582) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_10
  simp only [← c36942, ← c36943, ← c36944] at e_582_10
  have e_582_11 := arithEq_of_rows h (row := 582) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_11
  simp only [← c36945, ← c36946, k2, ← c36947] at e_582_11
  have hr := e_582_11
  simp only [e_582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2951)) := by
  have c36948 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36949 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36950 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36951 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36952 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36953 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_12 := arithEq_of_rows h (row := 582) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_12
  simp only [← c36948, ← c36949, ← c36950] at e_582_12
  have e_582_13 := arithEq_of_rows h (row := 582) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_13
  simp only [← c36951, ← c36952, k2, ← c36953] at e_582_13
  have hr := e_582_13
  simp only [e_582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2952)) := by
  have c36954 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36955 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36956 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36957 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36958 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36959 := (publicBatchWrapper89_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_14 := arithEq_of_rows h (row := 582) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_14
  simp only [← c36954, ← c36955, ← c36956] at e_582_14
  have e_582_15 := arithEq_of_rows h (row := 582) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_15
  simp only [← c36957, ← c36958, k2, ← c36959] at e_582_15
  have hr := e_582_15
  simp only [e_582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2953)) := by
  have c36960 := (publicBatchWrapper89_copies1155 a h).1
  have c36961 := (publicBatchWrapper89_copies1155 a h).2.1
  have c36962 := (publicBatchWrapper89_copies1155 a h).2.2.1
  have c36963 := (publicBatchWrapper89_copies1155 a h).2.2.2.1
  have c36964 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.1
  have c36965 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_16 := arithEq_of_rows h (row := 582) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_16
  simp only [← c36960, ← c36961, ← c36962] at e_582_16
  have e_582_17 := arithEq_of_rows h (row := 582) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_17
  simp only [← c36963, ← c36964, k2, ← c36965] at e_582_17
  have hr := e_582_17
  simp only [e_582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 582 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2954)) := by
  have c36966 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.1
  have c36967 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.1
  have c36968 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.1
  have c36969 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.1
  have c36970 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36971 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_582_18 := arithEq_of_rows h (row := 582) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_18
  simp only [← c36966, ← c36967, ← c36968] at e_582_18
  have e_582_19 := arithEq_of_rows h (row := 582) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_19
  simp only [← c36969, ← c36970, k2, ← c36971] at e_582_19
  have hr := e_582_19
  simp only [e_582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2955)) := by
  have c36972 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36973 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36974 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36975 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36976 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36977 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_0 := arithEq_of_rows h (row := 583) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_0
  simp only [← c36972, ← c36973, ← c36974] at e_583_0
  have e_583_1 := arithEq_of_rows h (row := 583) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_1
  simp only [← c36975, ← c36976, k2, ← c36977] at e_583_1
  have hr := e_583_1
  simp only [e_583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2956)) := by
  have c36978 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36979 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36980 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36981 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36982 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36983 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_2 := arithEq_of_rows h (row := 583) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_2
  simp only [← c36978, ← c36979, ← c36980] at e_583_2
  have e_583_3 := arithEq_of_rows h (row := 583) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_3
  simp only [← c36981, ← c36982, k2, ← c36983] at e_583_3
  have hr := e_583_3
  simp only [e_583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2957)) := by
  have c36984 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36985 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36986 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36987 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36988 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36989 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_4 := arithEq_of_rows h (row := 583) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_4
  simp only [← c36984, ← c36985, ← c36986] at e_583_4
  have e_583_5 := arithEq_of_rows h (row := 583) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_5
  simp only [← c36987, ← c36988, k2, ← c36989] at e_583_5
  have hr := e_583_5
  simp only [e_583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2958)) := by
  have c36990 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36991 := (publicBatchWrapper89_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36992 := (publicBatchWrapper89_copies1156 a h).1
  have c36993 := (publicBatchWrapper89_copies1156 a h).2.1
  have c36994 := (publicBatchWrapper89_copies1156 a h).2.2.1
  have c36995 := (publicBatchWrapper89_copies1156 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_6 := arithEq_of_rows h (row := 583) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_6
  simp only [← c36990, ← c36991, ← c36992] at e_583_6
  have e_583_7 := arithEq_of_rows h (row := 583) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_7
  simp only [← c36993, ← c36994, k2, ← c36995] at e_583_7
  have hr := e_583_7
  simp only [e_583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2959)) := by
  have c36996 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.1
  have c36997 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.1
  have c36998 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.1
  have c36999 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.1
  have c37000 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.1
  have c37001 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_8 := arithEq_of_rows h (row := 583) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_8
  simp only [← c36996, ← c36997, ← c36998] at e_583_8
  have e_583_9 := arithEq_of_rows h (row := 583) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_9
  simp only [← c36999, ← c37000, k2, ← c37001] at e_583_9
  have hr := e_583_9
  simp only [e_583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2960)) := by
  have c37002 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37003 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37004 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37005 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37006 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37007 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_10 := arithEq_of_rows h (row := 583) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_10
  simp only [← c37002, ← c37003, ← c37004] at e_583_10
  have e_583_11 := arithEq_of_rows h (row := 583) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_11
  simp only [← c37005, ← c37006, k2, ← c37007] at e_583_11
  have hr := e_583_11
  simp only [e_583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2961)) := by
  have c37008 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37009 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37010 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37011 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37012 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37013 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_12 := arithEq_of_rows h (row := 583) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_12
  simp only [← c37008, ← c37009, ← c37010] at e_583_12
  have e_583_13 := arithEq_of_rows h (row := 583) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_13
  simp only [← c37011, ← c37012, k2, ← c37013] at e_583_13
  have hr := e_583_13
  simp only [e_583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2962)) := by
  have c37014 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37015 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37016 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37017 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37018 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37019 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_14 := arithEq_of_rows h (row := 583) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_14
  simp only [← c37014, ← c37015, ← c37016] at e_583_14
  have e_583_15 := arithEq_of_rows h (row := 583) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_15
  simp only [← c37017, ← c37018, k2, ← c37019] at e_583_15
  have hr := e_583_15
  simp only [e_583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2963)) := by
  have c37020 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37021 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37022 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37023 := (publicBatchWrapper89_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37024 := (publicBatchWrapper89_copies1157 a h).1
  have c37025 := (publicBatchWrapper89_copies1157 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_16 := arithEq_of_rows h (row := 583) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_16
  simp only [← c37020, ← c37021, ← c37022] at e_583_16
  have e_583_17 := arithEq_of_rows h (row := 583) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_17
  simp only [← c37023, ← c37024, k2, ← c37025] at e_583_17
  have hr := e_583_17
  simp only [e_583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 583 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2964)) := by
  have c37026 := (publicBatchWrapper89_copies1157 a h).2.2.1
  have c37027 := (publicBatchWrapper89_copies1157 a h).2.2.2.1
  have c37028 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.1
  have c37029 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.1
  have c37030 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.1
  have c37031 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_583_18 := arithEq_of_rows h (row := 583) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_18
  simp only [← c37026, ← c37027, ← c37028] at e_583_18
  have e_583_19 := arithEq_of_rows h (row := 583) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_19
  simp only [← c37029, ← c37030, k2, ← c37031] at e_583_19
  have hr := e_583_19
  simp only [e_583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2965)) := by
  have c37032 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.1
  have c37033 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.1
  have c37034 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37035 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37036 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37037 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_0 := arithEq_of_rows h (row := 584) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_0
  simp only [← c37032, ← c37033, ← c37034] at e_584_0
  have e_584_1 := arithEq_of_rows h (row := 584) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_1
  simp only [← c37035, ← c37036, k2, ← c37037] at e_584_1
  have hr := e_584_1
  simp only [e_584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2966)) := by
  have c37038 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37039 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37040 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37041 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37042 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37043 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_2 := arithEq_of_rows h (row := 584) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_2
  simp only [← c37038, ← c37039, ← c37040] at e_584_2
  have e_584_3 := arithEq_of_rows h (row := 584) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_3
  simp only [← c37041, ← c37042, k2, ← c37043] at e_584_3
  have hr := e_584_3
  simp only [e_584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2967)) := by
  have c37044 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37045 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37046 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37047 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37048 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37049 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_4 := arithEq_of_rows h (row := 584) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_4
  simp only [← c37044, ← c37045, ← c37046] at e_584_4
  have e_584_5 := arithEq_of_rows h (row := 584) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_5
  simp only [← c37047, ← c37048, k2, ← c37049] at e_584_5
  have hr := e_584_5
  simp only [e_584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2968)) := by
  have c37050 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37051 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37052 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37053 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37054 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37055 := (publicBatchWrapper89_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_6 := arithEq_of_rows h (row := 584) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_6
  simp only [← c37050, ← c37051, ← c37052] at e_584_6
  have e_584_7 := arithEq_of_rows h (row := 584) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_7
  simp only [← c37053, ← c37054, k2, ← c37055] at e_584_7
  have hr := e_584_7
  simp only [e_584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2969)) := by
  have c37056 := (publicBatchWrapper89_copies1158 a h).1
  have c37057 := (publicBatchWrapper89_copies1158 a h).2.1
  have c37058 := (publicBatchWrapper89_copies1158 a h).2.2.1
  have c37059 := (publicBatchWrapper89_copies1158 a h).2.2.2.1
  have c37060 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.1
  have c37061 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_8 := arithEq_of_rows h (row := 584) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_8
  simp only [← c37056, ← c37057, ← c37058] at e_584_8
  have e_584_9 := arithEq_of_rows h (row := 584) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_9
  simp only [← c37059, ← c37060, k2, ← c37061] at e_584_9
  have hr := e_584_9
  simp only [e_584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3010)) := by
  have c37062 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.1
  have c37063 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.1
  have c37064 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.1
  have c37065 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.1
  have c37066 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37067 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_10 := arithEq_of_rows h (row := 584) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_10
  simp only [← c37062, ← c37063, ← c37064] at e_584_10
  have e_584_11 := arithEq_of_rows h (row := 584) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_11
  simp only [← c37065, ← c37066, k2, ← c37067] at e_584_11
  have hr := e_584_11
  simp only [e_584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3011)) := by
  have c37068 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37069 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37070 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37071 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37072 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37073 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_12 := arithEq_of_rows h (row := 584) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_12
  simp only [← c37068, ← c37069, ← c37070] at e_584_12
  have e_584_13 := arithEq_of_rows h (row := 584) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_13
  simp only [← c37071, ← c37072, k2, ← c37073] at e_584_13
  have hr := e_584_13
  simp only [e_584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3012)) := by
  have c37074 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37075 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37076 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37077 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37078 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37079 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_14 := arithEq_of_rows h (row := 584) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_14
  simp only [← c37074, ← c37075, ← c37076] at e_584_14
  have e_584_15 := arithEq_of_rows h (row := 584) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_15
  simp only [← c37077, ← c37078, k2, ← c37079] at e_584_15
  have hr := e_584_15
  simp only [e_584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3013)) := by
  have c37080 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37081 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37082 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37083 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37084 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37085 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_16 := arithEq_of_rows h (row := 584) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_16
  simp only [← c37080, ← c37081, ← c37082] at e_584_16
  have e_584_17 := arithEq_of_rows h (row := 584) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_17
  simp only [← c37083, ← c37084, k2, ← c37085] at e_584_17
  have hr := e_584_17
  simp only [e_584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 584 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3014)) := by
  have c37086 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37087 := (publicBatchWrapper89_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37088 := (publicBatchWrapper89_copies1159 a h).1
  have c37089 := (publicBatchWrapper89_copies1159 a h).2.1
  have c37090 := (publicBatchWrapper89_copies1159 a h).2.2.1
  have c37091 := (publicBatchWrapper89_copies1159 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_584_18 := arithEq_of_rows h (row := 584) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_18
  simp only [← c37086, ← c37087, ← c37088] at e_584_18
  have e_584_19 := arithEq_of_rows h (row := 584) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_19
  simp only [← c37089, ← c37090, k2, ← c37091] at e_584_19
  have hr := e_584_19
  simp only [e_584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3015)) := by
  have c37092 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.1
  have c37093 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.1
  have c37094 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.1
  have c37095 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.1
  have c37096 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.1
  have c37097 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_0 := arithEq_of_rows h (row := 585) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_0
  simp only [← c37092, ← c37093, ← c37094] at e_585_0
  have e_585_1 := arithEq_of_rows h (row := 585) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_1
  simp only [← c37095, ← c37096, k2, ← c37097] at e_585_1
  have hr := e_585_1
  simp only [e_585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3016)) := by
  have c37098 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37099 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37100 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37101 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37102 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37103 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_2 := arithEq_of_rows h (row := 585) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_2
  simp only [← c37098, ← c37099, ← c37100] at e_585_2
  have e_585_3 := arithEq_of_rows h (row := 585) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_3
  simp only [← c37101, ← c37102, k2, ← c37103] at e_585_3
  have hr := e_585_3
  simp only [e_585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3017)) := by
  have c37104 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37105 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37106 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37107 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37108 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37109 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_4 := arithEq_of_rows h (row := 585) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_4
  simp only [← c37104, ← c37105, ← c37106] at e_585_4
  have e_585_5 := arithEq_of_rows h (row := 585) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_5
  simp only [← c37107, ← c37108, k2, ← c37109] at e_585_5
  have hr := e_585_5
  simp only [e_585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3018)) := by
  have c37110 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37111 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37112 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37113 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37114 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37115 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_6 := arithEq_of_rows h (row := 585) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_6
  simp only [← c37110, ← c37111, ← c37112] at e_585_6
  have e_585_7 := arithEq_of_rows h (row := 585) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_7
  simp only [← c37113, ← c37114, k2, ← c37115] at e_585_7
  have hr := e_585_7
  simp only [e_585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3019)) := by
  have c37116 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37117 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37118 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37119 := (publicBatchWrapper89_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37120 := (publicBatchWrapper89_copies1160 a h).1
  have c37121 := (publicBatchWrapper89_copies1160 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_8 := arithEq_of_rows h (row := 585) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_8
  simp only [← c37116, ← c37117, ← c37118] at e_585_8
  have e_585_9 := arithEq_of_rows h (row := 585) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_9
  simp only [← c37119, ← c37120, k2, ← c37121] at e_585_9
  have hr := e_585_9
  simp only [e_585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3020)) := by
  have c37122 := (publicBatchWrapper89_copies1160 a h).2.2.1
  have c37123 := (publicBatchWrapper89_copies1160 a h).2.2.2.1
  have c37124 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.1
  have c37125 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.1
  have c37126 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.1
  have c37127 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_10 := arithEq_of_rows h (row := 585) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_10
  simp only [← c37122, ← c37123, ← c37124] at e_585_10
  have e_585_11 := arithEq_of_rows h (row := 585) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_11
  simp only [← c37125, ← c37126, k2, ← c37127] at e_585_11
  have hr := e_585_11
  simp only [e_585_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3021)) := by
  have c37128 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.1
  have c37129 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.1
  have c37130 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37131 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37132 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37133 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_12 := arithEq_of_rows h (row := 585) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_12
  simp only [← c37128, ← c37129, ← c37130] at e_585_12
  have e_585_13 := arithEq_of_rows h (row := 585) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_13
  simp only [← c37131, ← c37132, k2, ← c37133] at e_585_13
  have hr := e_585_13
  simp only [e_585_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3022)) := by
  have c37134 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37135 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37136 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37137 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37138 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37139 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_14 := arithEq_of_rows h (row := 585) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_14
  simp only [← c37134, ← c37135, ← c37136] at e_585_14
  have e_585_15 := arithEq_of_rows h (row := 585) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_15
  simp only [← c37137, ← c37138, k2, ← c37139] at e_585_15
  have hr := e_585_15
  simp only [e_585_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3023)) := by
  have c37140 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37141 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37142 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37143 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37144 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37145 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_16 := arithEq_of_rows h (row := 585) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_16
  simp only [← c37140, ← c37141, ← c37142] at e_585_16
  have e_585_17 := arithEq_of_rows h (row := 585) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_17
  simp only [← c37143, ← c37144, k2, ← c37145] at e_585_17
  have hr := e_585_17
  simp only [e_585_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 585 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3024)) := by
  have c37146 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37147 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37148 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37149 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37150 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37151 := (publicBatchWrapper89_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_585_18 := arithEq_of_rows h (row := 585) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_18
  simp only [← c37146, ← c37147, ← c37148] at e_585_18
  have e_585_19 := arithEq_of_rows h (row := 585) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_19
  simp only [← c37149, ← c37150, k2, ← c37151] at e_585_19
  have hr := e_585_19
  simp only [e_585_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3025)) := by
  have c37152 := (publicBatchWrapper89_copies1161 a h).1
  have c37153 := (publicBatchWrapper89_copies1161 a h).2.1
  have c37154 := (publicBatchWrapper89_copies1161 a h).2.2.1
  have c37155 := (publicBatchWrapper89_copies1161 a h).2.2.2.1
  have c37156 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.1
  have c37157 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_0 := arithEq_of_rows h (row := 586) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_0
  simp only [← c37152, ← c37153, ← c37154] at e_586_0
  have e_586_1 := arithEq_of_rows h (row := 586) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_1
  simp only [← c37155, ← c37156, k2, ← c37157] at e_586_1
  have hr := e_586_1
  simp only [e_586_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3026)) := by
  have c37158 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.1
  have c37159 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.1
  have c37160 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.1
  have c37161 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.1
  have c37162 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37163 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_2 := arithEq_of_rows h (row := 586) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_2
  simp only [← c37158, ← c37159, ← c37160] at e_586_2
  have e_586_3 := arithEq_of_rows h (row := 586) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_3
  simp only [← c37161, ← c37162, k2, ← c37163] at e_586_3
  have hr := e_586_3
  simp only [e_586_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3027)) := by
  have c37164 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37165 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37166 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37167 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37168 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37169 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_4 := arithEq_of_rows h (row := 586) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_4
  simp only [← c37164, ← c37165, ← c37166] at e_586_4
  have e_586_5 := arithEq_of_rows h (row := 586) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_5
  simp only [← c37167, ← c37168, k2, ← c37169] at e_586_5
  have hr := e_586_5
  simp only [e_586_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3028)) := by
  have c37170 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37171 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37172 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37173 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37174 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37175 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_6 := arithEq_of_rows h (row := 586) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_6
  simp only [← c37170, ← c37171, ← c37172] at e_586_6
  have e_586_7 := arithEq_of_rows h (row := 586) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_7
  simp only [← c37173, ← c37174, k2, ← c37175] at e_586_7
  have hr := e_586_7
  simp only [e_586_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3029)) := by
  have c37176 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37177 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37178 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37179 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37180 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37181 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_8 := arithEq_of_rows h (row := 586) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_8
  simp only [← c37176, ← c37177, ← c37178] at e_586_8
  have e_586_9 := arithEq_of_rows h (row := 586) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_9
  simp only [← c37179, ← c37180, k2, ← c37181] at e_586_9
  have hr := e_586_9
  simp only [e_586_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3030)) := by
  have c37182 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37183 := (publicBatchWrapper89_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37184 := (publicBatchWrapper89_copies1162 a h).1
  have c37185 := (publicBatchWrapper89_copies1162 a h).2.1
  have c37186 := (publicBatchWrapper89_copies1162 a h).2.2.1
  have c37187 := (publicBatchWrapper89_copies1162 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_10 := arithEq_of_rows h (row := 586) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_10
  simp only [← c37182, ← c37183, ← c37184] at e_586_10
  have e_586_11 := arithEq_of_rows h (row := 586) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_11
  simp only [← c37185, ← c37186, k2, ← c37187] at e_586_11
  have hr := e_586_11
  simp only [e_586_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3031)) := by
  have c37188 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.1
  have c37189 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.1
  have c37190 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.1
  have c37191 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.1
  have c37192 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.1
  have c37193 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_12 := arithEq_of_rows h (row := 586) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_12
  simp only [← c37188, ← c37189, ← c37190] at e_586_12
  have e_586_13 := arithEq_of_rows h (row := 586) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_13
  simp only [← c37191, ← c37192, k2, ← c37193] at e_586_13
  have hr := e_586_13
  simp only [e_586_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3032)) := by
  have c37194 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37195 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37196 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37197 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37198 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37199 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_14 := arithEq_of_rows h (row := 586) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_14
  simp only [← c37194, ← c37195, ← c37196] at e_586_14
  have e_586_15 := arithEq_of_rows h (row := 586) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_15
  simp only [← c37197, ← c37198, k2, ← c37199] at e_586_15
  have hr := e_586_15
  simp only [e_586_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3033)) := by
  have c37200 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37201 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37202 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37203 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37204 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37205 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_16 := arithEq_of_rows h (row := 586) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_16
  simp only [← c37200, ← c37201, ← c37202] at e_586_16
  have e_586_17 := arithEq_of_rows h (row := 586) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_17
  simp only [← c37203, ← c37204, k2, ← c37205] at e_586_17
  have hr := e_586_17
  simp only [e_586_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 586 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3034)) := by
  have c37206 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37207 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37208 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37209 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37210 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37211 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_586_18 := arithEq_of_rows h (row := 586) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_18
  simp only [← c37206, ← c37207, ← c37208] at e_586_18
  have e_586_19 := arithEq_of_rows h (row := 586) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_19
  simp only [← c37209, ← c37210, k2, ← c37211] at e_586_19
  have hr := e_586_19
  simp only [e_586_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3035)) := by
  have c37212 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37213 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37214 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37215 := (publicBatchWrapper89_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37216 := (publicBatchWrapper89_copies1163 a h).1
  have c37217 := (publicBatchWrapper89_copies1163 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_0 := arithEq_of_rows h (row := 587) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_0
  simp only [← c37212, ← c37213, ← c37214] at e_587_0
  have e_587_1 := arithEq_of_rows h (row := 587) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_1
  simp only [← c37215, ← c37216, k2, ← c37217] at e_587_1
  have hr := e_587_1
  simp only [e_587_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3036)) := by
  have c37218 := (publicBatchWrapper89_copies1163 a h).2.2.1
  have c37219 := (publicBatchWrapper89_copies1163 a h).2.2.2.1
  have c37220 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.1
  have c37221 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.1
  have c37222 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.1
  have c37223 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_2 := arithEq_of_rows h (row := 587) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_2
  simp only [← c37218, ← c37219, ← c37220] at e_587_2
  have e_587_3 := arithEq_of_rows h (row := 587) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_3
  simp only [← c37221, ← c37222, k2, ← c37223] at e_587_3
  have hr := e_587_3
  simp only [e_587_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3037)) := by
  have c37224 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.1
  have c37225 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.1
  have c37226 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37227 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37228 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37229 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_4 := arithEq_of_rows h (row := 587) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_4
  simp only [← c37224, ← c37225, ← c37226] at e_587_4
  have e_587_5 := arithEq_of_rows h (row := 587) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_5
  simp only [← c37227, ← c37228, k2, ← c37229] at e_587_5
  have hr := e_587_5
  simp only [e_587_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3038)) := by
  have c37230 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37231 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37232 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37233 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37234 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37235 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_6 := arithEq_of_rows h (row := 587) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_6
  simp only [← c37230, ← c37231, ← c37232] at e_587_6
  have e_587_7 := arithEq_of_rows h (row := 587) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_7
  simp only [← c37233, ← c37234, k2, ← c37235] at e_587_7
  have hr := e_587_7
  simp only [e_587_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3039)) := by
  have c37236 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37237 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37238 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37239 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37240 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37241 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_8 := arithEq_of_rows h (row := 587) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_8
  simp only [← c37236, ← c37237, ← c37238] at e_587_8
  have e_587_9 := arithEq_of_rows h (row := 587) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_9
  simp only [← c37239, ← c37240, k2, ← c37241] at e_587_9
  have hr := e_587_9
  simp only [e_587_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3040)) := by
  have c37242 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37243 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37244 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37245 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37246 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37247 := (publicBatchWrapper89_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_10 := arithEq_of_rows h (row := 587) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_10
  simp only [← c37242, ← c37243, ← c37244] at e_587_10
  have e_587_11 := arithEq_of_rows h (row := 587) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_11
  simp only [← c37245, ← c37246, k2, ← c37247] at e_587_11
  have hr := e_587_11
  simp only [e_587_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3041)) := by
  have c37248 := (publicBatchWrapper89_copies1164 a h).1
  have c37249 := (publicBatchWrapper89_copies1164 a h).2.1
  have c37250 := (publicBatchWrapper89_copies1164 a h).2.2.1
  have c37251 := (publicBatchWrapper89_copies1164 a h).2.2.2.1
  have c37252 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.1
  have c37253 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_12 := arithEq_of_rows h (row := 587) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_12
  simp only [← c37248, ← c37249, ← c37250] at e_587_12
  have e_587_13 := arithEq_of_rows h (row := 587) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_13
  simp only [← c37251, ← c37252, k2, ← c37253] at e_587_13
  have hr := e_587_13
  simp only [e_587_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3042)) := by
  have c37254 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.1
  have c37255 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.1
  have c37256 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.1
  have c37257 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.1
  have c37258 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37259 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_14 := arithEq_of_rows h (row := 587) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_14
  simp only [← c37254, ← c37255, ← c37256] at e_587_14
  have e_587_15 := arithEq_of_rows h (row := 587) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_15
  simp only [← c37257, ← c37258, k2, ← c37259] at e_587_15
  have hr := e_587_15
  simp only [e_587_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3043)) := by
  have c37260 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37261 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37262 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37263 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37264 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37265 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_16 := arithEq_of_rows h (row := 587) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_16
  simp only [← c37260, ← c37261, ← c37262] at e_587_16
  have e_587_17 := arithEq_of_rows h (row := 587) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_17
  simp only [← c37263, ← c37264, k2, ← c37265] at e_587_17
  have hr := e_587_17
  simp only [e_587_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 587 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3044)) := by
  have c37266 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37267 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37268 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37269 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37270 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37271 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_587_18 := arithEq_of_rows h (row := 587) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_18
  simp only [← c37266, ← c37267, ← c37268] at e_587_18
  have e_587_19 := arithEq_of_rows h (row := 587) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_19
  simp only [← c37269, ← c37270, k2, ← c37271] at e_587_19
  have hr := e_587_19
  simp only [e_587_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3045)) := by
  have c37272 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37273 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37274 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37275 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37276 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37277 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_0 := arithEq_of_rows h (row := 588) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_0
  simp only [← c37272, ← c37273, ← c37274] at e_588_0
  have e_588_1 := arithEq_of_rows h (row := 588) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_1
  simp only [← c37275, ← c37276, k2, ← c37277] at e_588_1
  have hr := e_588_1
  simp only [e_588_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3046)) := by
  have c37278 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37279 := (publicBatchWrapper89_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37280 := (publicBatchWrapper89_copies1165 a h).1
  have c37281 := (publicBatchWrapper89_copies1165 a h).2.1
  have c37282 := (publicBatchWrapper89_copies1165 a h).2.2.1
  have c37283 := (publicBatchWrapper89_copies1165 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_2 := arithEq_of_rows h (row := 588) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_2
  simp only [← c37278, ← c37279, ← c37280] at e_588_2
  have e_588_3 := arithEq_of_rows h (row := 588) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_3
  simp only [← c37281, ← c37282, k2, ← c37283] at e_588_3
  have hr := e_588_3
  simp only [e_588_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3047)) := by
  have c37284 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.1
  have c37285 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.1
  have c37286 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.1
  have c37287 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.1
  have c37288 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.1
  have c37289 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_4 := arithEq_of_rows h (row := 588) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_4
  simp only [← c37284, ← c37285, ← c37286] at e_588_4
  have e_588_5 := arithEq_of_rows h (row := 588) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_5
  simp only [← c37287, ← c37288, k2, ← c37289] at e_588_5
  have hr := e_588_5
  simp only [e_588_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3048)) := by
  have c37290 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37291 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37292 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37293 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37294 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37295 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_6 := arithEq_of_rows h (row := 588) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_6
  simp only [← c37290, ← c37291, ← c37292] at e_588_6
  have e_588_7 := arithEq_of_rows h (row := 588) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_7
  simp only [← c37293, ← c37294, k2, ← c37295] at e_588_7
  have hr := e_588_7
  simp only [e_588_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3049)) := by
  have c37296 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37297 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37298 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37299 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37300 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37301 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_8 := arithEq_of_rows h (row := 588) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_8
  simp only [← c37296, ← c37297, ← c37298] at e_588_8
  have e_588_9 := arithEq_of_rows h (row := 588) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_9
  simp only [← c37299, ← c37300, k2, ← c37301] at e_588_9
  have hr := e_588_9
  simp only [e_588_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3050)) := by
  have c37302 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37303 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37304 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37305 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37306 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37307 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_10 := arithEq_of_rows h (row := 588) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_10
  simp only [← c37302, ← c37303, ← c37304] at e_588_10
  have e_588_11 := arithEq_of_rows h (row := 588) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_11
  simp only [← c37305, ← c37306, k2, ← c37307] at e_588_11
  have hr := e_588_11
  simp only [e_588_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3051)) := by
  have c37308 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37309 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37310 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37311 := (publicBatchWrapper89_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37312 := (publicBatchWrapper89_copies1166 a h).1
  have c37313 := (publicBatchWrapper89_copies1166 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_12 := arithEq_of_rows h (row := 588) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_12
  simp only [← c37308, ← c37309, ← c37310] at e_588_12
  have e_588_13 := arithEq_of_rows h (row := 588) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_13
  simp only [← c37311, ← c37312, k2, ← c37313] at e_588_13
  have hr := e_588_13
  simp only [e_588_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3052)) := by
  have c37314 := (publicBatchWrapper89_copies1166 a h).2.2.1
  have c37315 := (publicBatchWrapper89_copies1166 a h).2.2.2.1
  have c37316 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.1
  have c37317 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.1
  have c37318 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.1
  have c37319 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_14 := arithEq_of_rows h (row := 588) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_14
  simp only [← c37314, ← c37315, ← c37316] at e_588_14
  have e_588_15 := arithEq_of_rows h (row := 588) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_15
  simp only [← c37317, ← c37318, k2, ← c37319] at e_588_15
  have hr := e_588_15
  simp only [e_588_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3053)) := by
  have c37320 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.1
  have c37321 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.1
  have c37322 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37323 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37324 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37325 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_16 := arithEq_of_rows h (row := 588) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_16
  simp only [← c37320, ← c37321, ← c37322] at e_588_16
  have e_588_17 := arithEq_of_rows h (row := 588) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_17
  simp only [← c37323, ← c37324, k2, ← c37325] at e_588_17
  have hr := e_588_17
  simp only [e_588_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 588 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3054)) := by
  have c37326 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37327 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37328 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37329 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37330 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37331 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_588_18 := arithEq_of_rows h (row := 588) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_18
  simp only [← c37326, ← c37327, ← c37328] at e_588_18
  have e_588_19 := arithEq_of_rows h (row := 588) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_19
  simp only [← c37329, ← c37330, k2, ← c37331] at e_588_19
  have hr := e_588_19
  simp only [e_588_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3055)) := by
  have c37332 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37333 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37334 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37335 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37336 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37337 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_0 := arithEq_of_rows h (row := 589) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_0
  simp only [← c37332, ← c37333, ← c37334] at e_589_0
  have e_589_1 := arithEq_of_rows h (row := 589) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_1
  simp only [← c37335, ← c37336, k2, ← c37337] at e_589_1
  have hr := e_589_1
  simp only [e_589_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3056)) := by
  have c37338 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37339 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37340 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37341 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37342 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37343 := (publicBatchWrapper89_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_2 := arithEq_of_rows h (row := 589) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_2
  simp only [← c37338, ← c37339, ← c37340] at e_589_2
  have e_589_3 := arithEq_of_rows h (row := 589) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_3
  simp only [← c37341, ← c37342, k2, ← c37343] at e_589_3
  have hr := e_589_3
  simp only [e_589_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3057)) := by
  have c37344 := (publicBatchWrapper89_copies1167 a h).1
  have c37345 := (publicBatchWrapper89_copies1167 a h).2.1
  have c37346 := (publicBatchWrapper89_copies1167 a h).2.2.1
  have c37347 := (publicBatchWrapper89_copies1167 a h).2.2.2.1
  have c37348 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.1
  have c37349 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_4 := arithEq_of_rows h (row := 589) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_4
  simp only [← c37344, ← c37345, ← c37346] at e_589_4
  have e_589_5 := arithEq_of_rows h (row := 589) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_5
  simp only [← c37347, ← c37348, k2, ← c37349] at e_589_5
  have hr := e_589_5
  simp only [e_589_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3058)) := by
  have c37350 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.1
  have c37351 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.1
  have c37352 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.1
  have c37353 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.1
  have c37354 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37355 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_6 := arithEq_of_rows h (row := 589) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_6
  simp only [← c37350, ← c37351, ← c37352] at e_589_6
  have e_589_7 := arithEq_of_rows h (row := 589) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_7
  simp only [← c37353, ← c37354, k2, ← c37355] at e_589_7
  have hr := e_589_7
  simp only [e_589_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3059)) := by
  have c37356 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37357 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37358 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37359 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37360 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37361 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_8 := arithEq_of_rows h (row := 589) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_8
  simp only [← c37356, ← c37357, ← c37358] at e_589_8
  have e_589_9 := arithEq_of_rows h (row := 589) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_9
  simp only [← c37359, ← c37360, k2, ← c37361] at e_589_9
  have hr := e_589_9
  simp only [e_589_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3060)) := by
  have c37362 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37363 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37364 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37365 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37366 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37367 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_10 := arithEq_of_rows h (row := 589) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_10
  simp only [← c37362, ← c37363, ← c37364] at e_589_10
  have e_589_11 := arithEq_of_rows h (row := 589) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_11
  simp only [← c37365, ← c37366, k2, ← c37367] at e_589_11
  have hr := e_589_11
  simp only [e_589_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3061)) := by
  have c37368 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37369 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37370 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37371 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37372 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37373 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_12 := arithEq_of_rows h (row := 589) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_12
  simp only [← c37368, ← c37369, ← c37370] at e_589_12
  have e_589_13 := arithEq_of_rows h (row := 589) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_13
  simp only [← c37371, ← c37372, k2, ← c37373] at e_589_13
  have hr := e_589_13
  simp only [e_589_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3062)) := by
  have c37374 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37375 := (publicBatchWrapper89_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37376 := (publicBatchWrapper89_copies1168 a h).1
  have c37377 := (publicBatchWrapper89_copies1168 a h).2.1
  have c37378 := (publicBatchWrapper89_copies1168 a h).2.2.1
  have c37379 := (publicBatchWrapper89_copies1168 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_14 := arithEq_of_rows h (row := 589) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_14
  simp only [← c37374, ← c37375, ← c37376] at e_589_14
  have e_589_15 := arithEq_of_rows h (row := 589) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_15
  simp only [← c37377, ← c37378, k2, ← c37379] at e_589_15
  have hr := e_589_15
  simp only [e_589_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3063)) := by
  have c37380 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.1
  have c37381 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.1
  have c37382 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.1
  have c37383 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.1
  have c37384 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.1
  have c37385 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_16 := arithEq_of_rows h (row := 589) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_16
  simp only [← c37380, ← c37381, ← c37382] at e_589_16
  have e_589_17 := arithEq_of_rows h (row := 589) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_17
  simp only [← c37383, ← c37384, k2, ← c37385] at e_589_17
  have hr := e_589_17
  simp only [e_589_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 589 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3064)) := by
  have c37386 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37387 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37388 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37389 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37390 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37391 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_589_18 := arithEq_of_rows h (row := 589) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_18
  simp only [← c37386, ← c37387, ← c37388] at e_589_18
  have e_589_19 := arithEq_of_rows h (row := 589) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_19
  simp only [← c37389, ← c37390, k2, ← c37391] at e_589_19
  have hr := e_589_19
  simp only [e_589_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3065)) := by
  have c37392 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37393 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37394 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37395 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37396 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37397 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_0 := arithEq_of_rows h (row := 590) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_0
  simp only [← c37392, ← c37393, ← c37394] at e_590_0
  have e_590_1 := arithEq_of_rows h (row := 590) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_1
  simp only [← c37395, ← c37396, k2, ← c37397] at e_590_1
  have hr := e_590_1
  simp only [e_590_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3066)) := by
  have c37398 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37399 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37400 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37401 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37402 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37403 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_2 := arithEq_of_rows h (row := 590) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_2
  simp only [← c37398, ← c37399, ← c37400] at e_590_2
  have e_590_3 := arithEq_of_rows h (row := 590) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_3
  simp only [← c37401, ← c37402, k2, ← c37403] at e_590_3
  have hr := e_590_3
  simp only [e_590_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3067)) := by
  have c37404 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37405 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37406 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37407 := (publicBatchWrapper89_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37408 := (publicBatchWrapper89_copies1169 a h).1
  have c37409 := (publicBatchWrapper89_copies1169 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_4 := arithEq_of_rows h (row := 590) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_4
  simp only [← c37404, ← c37405, ← c37406] at e_590_4
  have e_590_5 := arithEq_of_rows h (row := 590) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_5
  simp only [← c37407, ← c37408, k2, ← c37409] at e_590_5
  have hr := e_590_5
  simp only [e_590_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3068)) := by
  have c37410 := (publicBatchWrapper89_copies1169 a h).2.2.1
  have c37411 := (publicBatchWrapper89_copies1169 a h).2.2.2.1
  have c37412 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.1
  have c37413 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.1
  have c37414 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.1
  have c37415 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_6 := arithEq_of_rows h (row := 590) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_6
  simp only [← c37410, ← c37411, ← c37412] at e_590_6
  have e_590_7 := arithEq_of_rows h (row := 590) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_7
  simp only [← c37413, ← c37414, k2, ← c37415] at e_590_7
  have hr := e_590_7
  simp only [e_590_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3069)) := by
  have c37416 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.1
  have c37417 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.1
  have c37418 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37419 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37420 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37421 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_8 := arithEq_of_rows h (row := 590) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_8
  simp only [← c37416, ← c37417, ← c37418] at e_590_8
  have e_590_9 := arithEq_of_rows h (row := 590) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_9
  simp only [← c37419, ← c37420, k2, ← c37421] at e_590_9
  have hr := e_590_9
  simp only [e_590_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3070)) := by
  have c37422 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37423 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37424 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37425 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37426 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37427 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_10 := arithEq_of_rows h (row := 590) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_10
  simp only [← c37422, ← c37423, ← c37424] at e_590_10
  have e_590_11 := arithEq_of_rows h (row := 590) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_11
  simp only [← c37425, ← c37426, k2, ← c37427] at e_590_11
  have hr := e_590_11
  simp only [e_590_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3071)) := by
  have c37428 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37429 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37430 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37431 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37432 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37433 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_12 := arithEq_of_rows h (row := 590) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_12
  simp only [← c37428, ← c37429, ← c37430] at e_590_12
  have e_590_13 := arithEq_of_rows h (row := 590) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_13
  simp only [← c37431, ← c37432, k2, ← c37433] at e_590_13
  have hr := e_590_13
  simp only [e_590_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3072)) := by
  have c37434 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37435 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37436 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37437 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37438 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37439 := (publicBatchWrapper89_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_14 := arithEq_of_rows h (row := 590) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_14
  simp only [← c37434, ← c37435, ← c37436] at e_590_14
  have e_590_15 := arithEq_of_rows h (row := 590) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_15
  simp only [← c37437, ← c37438, k2, ← c37439] at e_590_15
  have hr := e_590_15
  simp only [e_590_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3073)) := by
  have c37440 := (publicBatchWrapper89_copies1170 a h).1
  have c37441 := (publicBatchWrapper89_copies1170 a h).2.1
  have c37442 := (publicBatchWrapper89_copies1170 a h).2.2.1
  have c37443 := (publicBatchWrapper89_copies1170 a h).2.2.2.1
  have c37444 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.1
  have c37445 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_16 := arithEq_of_rows h (row := 590) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_16
  simp only [← c37440, ← c37441, ← c37442] at e_590_16
  have e_590_17 := arithEq_of_rows h (row := 590) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_17
  simp only [← c37443, ← c37444, k2, ← c37445] at e_590_17
  have hr := e_590_17
  simp only [e_590_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 590 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3074)) := by
  have c37446 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.1
  have c37447 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.1
  have c37448 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.1
  have c37449 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.1
  have c37450 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37451 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_590_18 := arithEq_of_rows h (row := 590) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_18
  simp only [← c37446, ← c37447, ← c37448] at e_590_18
  have e_590_19 := arithEq_of_rows h (row := 590) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_19
  simp only [← c37449, ← c37450, k2, ← c37451] at e_590_19
  have hr := e_590_19
  simp only [e_590_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3075)) := by
  have c37452 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37453 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37454 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37455 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37456 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37457 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_0 := arithEq_of_rows h (row := 591) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_0
  simp only [← c37452, ← c37453, ← c37454] at e_591_0
  have e_591_1 := arithEq_of_rows h (row := 591) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_1
  simp only [← c37455, ← c37456, k2, ← c37457] at e_591_1
  have hr := e_591_1
  simp only [e_591_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3076)) := by
  have c37458 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37459 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37460 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37461 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37462 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37463 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_2 := arithEq_of_rows h (row := 591) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_2
  simp only [← c37458, ← c37459, ← c37460] at e_591_2
  have e_591_3 := arithEq_of_rows h (row := 591) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_3
  simp only [← c37461, ← c37462, k2, ← c37463] at e_591_3
  have hr := e_591_3
  simp only [e_591_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3077)) := by
  have c37464 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37465 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37466 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37467 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37468 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37469 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_4 := arithEq_of_rows h (row := 591) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_4
  simp only [← c37464, ← c37465, ← c37466] at e_591_4
  have e_591_5 := arithEq_of_rows h (row := 591) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_5
  simp only [← c37467, ← c37468, k2, ← c37469] at e_591_5
  have hr := e_591_5
  simp only [e_591_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3078)) := by
  have c37470 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37471 := (publicBatchWrapper89_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37472 := (publicBatchWrapper89_copies1171 a h).1
  have c37473 := (publicBatchWrapper89_copies1171 a h).2.1
  have c37474 := (publicBatchWrapper89_copies1171 a h).2.2.1
  have c37475 := (publicBatchWrapper89_copies1171 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_6 := arithEq_of_rows h (row := 591) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_6
  simp only [← c37470, ← c37471, ← c37472] at e_591_6
  have e_591_7 := arithEq_of_rows h (row := 591) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_7
  simp only [← c37473, ← c37474, k2, ← c37475] at e_591_7
  have hr := e_591_7
  simp only [e_591_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3079)) := by
  have c37476 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.1
  have c37477 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.1
  have c37478 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.1
  have c37479 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.1
  have c37480 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.1
  have c37481 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_8 := arithEq_of_rows h (row := 591) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_8
  simp only [← c37476, ← c37477, ← c37478] at e_591_8
  have e_591_9 := arithEq_of_rows h (row := 591) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_9
  simp only [← c37479, ← c37480, k2, ← c37481] at e_591_9
  have hr := e_591_9
  simp only [e_591_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3080)) := by
  have c37482 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37483 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37484 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37485 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37486 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37487 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_10 := arithEq_of_rows h (row := 591) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_10
  simp only [← c37482, ← c37483, ← c37484] at e_591_10
  have e_591_11 := arithEq_of_rows h (row := 591) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_11
  simp only [← c37485, ← c37486, k2, ← c37487] at e_591_11
  have hr := e_591_11
  simp only [e_591_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3081)) := by
  have c37488 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37489 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37490 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37491 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37492 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37493 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_12 := arithEq_of_rows h (row := 591) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_12
  simp only [← c37488, ← c37489, ← c37490] at e_591_12
  have e_591_13 := arithEq_of_rows h (row := 591) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_13
  simp only [← c37491, ← c37492, k2, ← c37493] at e_591_13
  have hr := e_591_13
  simp only [e_591_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3082)) := by
  have c37494 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37495 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37496 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37497 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37498 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37499 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_14 := arithEq_of_rows h (row := 591) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_14
  simp only [← c37494, ← c37495, ← c37496] at e_591_14
  have e_591_15 := arithEq_of_rows h (row := 591) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_15
  simp only [← c37497, ← c37498, k2, ← c37499] at e_591_15
  have hr := e_591_15
  simp only [e_591_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3083)) := by
  have c37500 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37501 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37502 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37503 := (publicBatchWrapper89_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37504 := (publicBatchWrapper89_copies1172 a h).1
  have c37505 := (publicBatchWrapper89_copies1172 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_16 := arithEq_of_rows h (row := 591) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_16
  simp only [← c37500, ← c37501, ← c37502] at e_591_16
  have e_591_17 := arithEq_of_rows h (row := 591) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_17
  simp only [← c37503, ← c37504, k2, ← c37505] at e_591_17
  have hr := e_591_17
  simp only [e_591_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 591 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3084)) := by
  have c37506 := (publicBatchWrapper89_copies1172 a h).2.2.1
  have c37507 := (publicBatchWrapper89_copies1172 a h).2.2.2.1
  have c37508 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.1
  have c37509 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.1
  have c37510 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.1
  have c37511 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_591_18 := arithEq_of_rows h (row := 591) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_18
  simp only [← c37506, ← c37507, ← c37508] at e_591_18
  have e_591_19 := arithEq_of_rows h (row := 591) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_19
  simp only [← c37509, ← c37510, k2, ← c37511] at e_591_19
  have hr := e_591_19
  simp only [e_591_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3085)) := by
  have c37512 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.1
  have c37513 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.1
  have c37514 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37515 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37516 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37517 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_0 := arithEq_of_rows h (row := 592) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_0
  simp only [← c37512, ← c37513, ← c37514] at e_592_0
  have e_592_1 := arithEq_of_rows h (row := 592) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_1
  simp only [← c37515, ← c37516, k2, ← c37517] at e_592_1
  have hr := e_592_1
  simp only [e_592_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3086)) := by
  have c37518 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37519 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37520 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37521 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37522 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37523 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_2 := arithEq_of_rows h (row := 592) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_2
  simp only [← c37518, ← c37519, ← c37520] at e_592_2
  have e_592_3 := arithEq_of_rows h (row := 592) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_3
  simp only [← c37521, ← c37522, k2, ← c37523] at e_592_3
  have hr := e_592_3
  simp only [e_592_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3087)) := by
  have c37524 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37525 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37526 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37527 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37528 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37529 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_4 := arithEq_of_rows h (row := 592) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_4
  simp only [← c37524, ← c37525, ← c37526] at e_592_4
  have e_592_5 := arithEq_of_rows h (row := 592) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_5
  simp only [← c37527, ← c37528, k2, ← c37529] at e_592_5
  have hr := e_592_5
  simp only [e_592_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3088)) := by
  have c37530 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37531 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37532 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37533 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37534 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37535 := (publicBatchWrapper89_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_6 := arithEq_of_rows h (row := 592) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_6
  simp only [← c37530, ← c37531, ← c37532] at e_592_6
  have e_592_7 := arithEq_of_rows h (row := 592) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_7
  simp only [← c37533, ← c37534, k2, ← c37535] at e_592_7
  have hr := e_592_7
  simp only [e_592_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3089)) := by
  have c37536 := (publicBatchWrapper89_copies1173 a h).1
  have c37537 := (publicBatchWrapper89_copies1173 a h).2.1
  have c37538 := (publicBatchWrapper89_copies1173 a h).2.2.1
  have c37539 := (publicBatchWrapper89_copies1173 a h).2.2.2.1
  have c37540 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.1
  have c37541 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_8 := arithEq_of_rows h (row := 592) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_8
  simp only [← c37536, ← c37537, ← c37538] at e_592_8
  have e_592_9 := arithEq_of_rows h (row := 592) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_9
  simp only [← c37539, ← c37540, k2, ← c37541] at e_592_9
  have hr := e_592_9
  simp only [e_592_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3090)) := by
  have c37542 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.1
  have c37543 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.1
  have c37544 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.1
  have c37545 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.1
  have c37546 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37547 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_10 := arithEq_of_rows h (row := 592) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_10
  simp only [← c37542, ← c37543, ← c37544] at e_592_10
  have e_592_11 := arithEq_of_rows h (row := 592) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_11
  simp only [← c37545, ← c37546, k2, ← c37547] at e_592_11
  have hr := e_592_11
  simp only [e_592_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3091)) := by
  have c37548 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37549 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37550 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37551 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37552 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37553 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_12 := arithEq_of_rows h (row := 592) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_12
  simp only [← c37548, ← c37549, ← c37550] at e_592_12
  have e_592_13 := arithEq_of_rows h (row := 592) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_13
  simp only [← c37551, ← c37552, k2, ← c37553] at e_592_13
  have hr := e_592_13
  simp only [e_592_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3092)) := by
  have c37554 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37555 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37556 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37557 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37558 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37559 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_14 := arithEq_of_rows h (row := 592) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_14
  simp only [← c37554, ← c37555, ← c37556] at e_592_14
  have e_592_15 := arithEq_of_rows h (row := 592) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_15
  simp only [← c37557, ← c37558, k2, ← c37559] at e_592_15
  have hr := e_592_15
  simp only [e_592_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3093)) := by
  have c37560 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37561 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37562 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37563 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37564 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37565 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_16 := arithEq_of_rows h (row := 592) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_16
  simp only [← c37560, ← c37561, ← c37562] at e_592_16
  have e_592_17 := arithEq_of_rows h (row := 592) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_17
  simp only [← c37563, ← c37564, k2, ← c37565] at e_592_17
  have hr := e_592_17
  simp only [e_592_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 592 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3094)) := by
  have c37566 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37567 := (publicBatchWrapper89_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37568 := (publicBatchWrapper89_copies1174 a h).1
  have c37569 := (publicBatchWrapper89_copies1174 a h).2.1
  have c37570 := (publicBatchWrapper89_copies1174 a h).2.2.1
  have c37571 := (publicBatchWrapper89_copies1174 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_592_18 := arithEq_of_rows h (row := 592) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_18
  simp only [← c37566, ← c37567, ← c37568] at e_592_18
  have e_592_19 := arithEq_of_rows h (row := 592) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_19
  simp only [← c37569, ← c37570, k2, ← c37571] at e_592_19
  have hr := e_592_19
  simp only [e_592_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3095)) := by
  have c37572 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.1
  have c37573 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.1
  have c37574 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.1
  have c37575 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.1
  have c37576 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.1
  have c37577 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_0 := arithEq_of_rows h (row := 593) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_0
  simp only [← c37572, ← c37573, ← c37574] at e_593_0
  have e_593_1 := arithEq_of_rows h (row := 593) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_1
  simp only [← c37575, ← c37576, k2, ← c37577] at e_593_1
  have hr := e_593_1
  simp only [e_593_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3096)) := by
  have c37578 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37579 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37580 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37581 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37582 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37583 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_2 := arithEq_of_rows h (row := 593) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_2
  simp only [← c37578, ← c37579, ← c37580] at e_593_2
  have e_593_3 := arithEq_of_rows h (row := 593) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_3
  simp only [← c37581, ← c37582, k2, ← c37583] at e_593_3
  have hr := e_593_3
  simp only [e_593_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3097)) := by
  have c37584 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37585 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37586 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37587 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37588 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37589 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_4 := arithEq_of_rows h (row := 593) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_4
  simp only [← c37584, ← c37585, ← c37586] at e_593_4
  have e_593_5 := arithEq_of_rows h (row := 593) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_5
  simp only [← c37587, ← c37588, k2, ← c37589] at e_593_5
  have hr := e_593_5
  simp only [e_593_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3098)) := by
  have c37590 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37591 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37592 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37593 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37594 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37595 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_6 := arithEq_of_rows h (row := 593) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_6
  simp only [← c37590, ← c37591, ← c37592] at e_593_6
  have e_593_7 := arithEq_of_rows h (row := 593) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_7
  simp only [← c37593, ← c37594, k2, ← c37595] at e_593_7
  have hr := e_593_7
  simp only [e_593_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3099)) := by
  have c37596 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37597 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37598 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37599 := (publicBatchWrapper89_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37600 := (publicBatchWrapper89_copies1175 a h).1
  have c37601 := (publicBatchWrapper89_copies1175 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_8 := arithEq_of_rows h (row := 593) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_8
  simp only [← c37596, ← c37597, ← c37598] at e_593_8
  have e_593_9 := arithEq_of_rows h (row := 593) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_9
  simp only [← c37599, ← c37600, k2, ← c37601] at e_593_9
  have hr := e_593_9
  simp only [e_593_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3100)) := by
  have c37602 := (publicBatchWrapper89_copies1175 a h).2.2.1
  have c37603 := (publicBatchWrapper89_copies1175 a h).2.2.2.1
  have c37604 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.1
  have c37605 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.1
  have c37606 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.1
  have c37607 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_10 := arithEq_of_rows h (row := 593) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_10
  simp only [← c37602, ← c37603, ← c37604] at e_593_10
  have e_593_11 := arithEq_of_rows h (row := 593) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_11
  simp only [← c37605, ← c37606, k2, ← c37607] at e_593_11
  have hr := e_593_11
  simp only [e_593_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3101)) := by
  have c37608 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.1
  have c37609 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.1
  have c37610 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37611 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37612 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37613 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_12 := arithEq_of_rows h (row := 593) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_12
  simp only [← c37608, ← c37609, ← c37610] at e_593_12
  have e_593_13 := arithEq_of_rows h (row := 593) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_13
  simp only [← c37611, ← c37612, k2, ← c37613] at e_593_13
  have hr := e_593_13
  simp only [e_593_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3102)) := by
  have c37614 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37615 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37616 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37617 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37618 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37619 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_14 := arithEq_of_rows h (row := 593) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_14
  simp only [← c37614, ← c37615, ← c37616] at e_593_14
  have e_593_15 := arithEq_of_rows h (row := 593) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_15
  simp only [← c37617, ← c37618, k2, ← c37619] at e_593_15
  have hr := e_593_15
  simp only [e_593_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3103)) := by
  have c37620 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37621 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37622 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37623 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37624 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37625 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_16 := arithEq_of_rows h (row := 593) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_16
  simp only [← c37620, ← c37621, ← c37622] at e_593_16
  have e_593_17 := arithEq_of_rows h (row := 593) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_17
  simp only [← c37623, ← c37624, k2, ← c37625] at e_593_17
  have hr := e_593_17
  simp only [e_593_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 593 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3104)) := by
  have c37626 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37627 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37628 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37629 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37630 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37631 := (publicBatchWrapper89_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_593_18 := arithEq_of_rows h (row := 593) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_18
  simp only [← c37626, ← c37627, ← c37628] at e_593_18
  have e_593_19 := arithEq_of_rows h (row := 593) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_19
  simp only [← c37629, ← c37630, k2, ← c37631] at e_593_19
  have hr := e_593_19
  simp only [e_593_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3105)) := by
  have c37632 := (publicBatchWrapper89_copies1176 a h).1
  have c37633 := (publicBatchWrapper89_copies1176 a h).2.1
  have c37634 := (publicBatchWrapper89_copies1176 a h).2.2.1
  have c37635 := (publicBatchWrapper89_copies1176 a h).2.2.2.1
  have c37636 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.1
  have c37637 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_0 := arithEq_of_rows h (row := 594) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_0
  simp only [← c37632, ← c37633, ← c37634] at e_594_0
  have e_594_1 := arithEq_of_rows h (row := 594) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_1
  simp only [← c37635, ← c37636, k2, ← c37637] at e_594_1
  have hr := e_594_1
  simp only [e_594_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3106)) := by
  have c37638 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.1
  have c37639 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.1
  have c37640 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.1
  have c37641 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.1
  have c37642 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37643 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_2 := arithEq_of_rows h (row := 594) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_2
  simp only [← c37638, ← c37639, ← c37640] at e_594_2
  have e_594_3 := arithEq_of_rows h (row := 594) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_3
  simp only [← c37641, ← c37642, k2, ← c37643] at e_594_3
  have hr := e_594_3
  simp only [e_594_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3107)) := by
  have c37644 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37645 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37646 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37647 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37648 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37649 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_4 := arithEq_of_rows h (row := 594) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_4
  simp only [← c37644, ← c37645, ← c37646] at e_594_4
  have e_594_5 := arithEq_of_rows h (row := 594) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_5
  simp only [← c37647, ← c37648, k2, ← c37649] at e_594_5
  have hr := e_594_5
  simp only [e_594_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3108)) := by
  have c37650 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37651 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37652 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37653 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37654 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37655 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_6 := arithEq_of_rows h (row := 594) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_6
  simp only [← c37650, ← c37651, ← c37652] at e_594_6
  have e_594_7 := arithEq_of_rows h (row := 594) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_7
  simp only [← c37653, ← c37654, k2, ← c37655] at e_594_7
  have hr := e_594_7
  simp only [e_594_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3109)) := by
  have c37656 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37657 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37658 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37659 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37660 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37661 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_8 := arithEq_of_rows h (row := 594) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_8
  simp only [← c37656, ← c37657, ← c37658] at e_594_8
  have e_594_9 := arithEq_of_rows h (row := 594) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_9
  simp only [← c37659, ← c37660, k2, ← c37661] at e_594_9
  have hr := e_594_9
  simp only [e_594_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3110)) := by
  have c37662 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37663 := (publicBatchWrapper89_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37664 := (publicBatchWrapper89_copies1177 a h).1
  have c37665 := (publicBatchWrapper89_copies1177 a h).2.1
  have c37666 := (publicBatchWrapper89_copies1177 a h).2.2.1
  have c37667 := (publicBatchWrapper89_copies1177 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_10 := arithEq_of_rows h (row := 594) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_10
  simp only [← c37662, ← c37663, ← c37664] at e_594_10
  have e_594_11 := arithEq_of_rows h (row := 594) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_11
  simp only [← c37665, ← c37666, k2, ← c37667] at e_594_11
  have hr := e_594_11
  simp only [e_594_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3111)) := by
  have c37668 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.1
  have c37669 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.1
  have c37670 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.1
  have c37671 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.1
  have c37672 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.1
  have c37673 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_12 := arithEq_of_rows h (row := 594) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_12
  simp only [← c37668, ← c37669, ← c37670] at e_594_12
  have e_594_13 := arithEq_of_rows h (row := 594) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_13
  simp only [← c37671, ← c37672, k2, ← c37673] at e_594_13
  have hr := e_594_13
  simp only [e_594_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3112)) := by
  have c37674 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37675 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37676 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37677 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37678 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37679 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_14 := arithEq_of_rows h (row := 594) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_14
  simp only [← c37674, ← c37675, ← c37676] at e_594_14
  have e_594_15 := arithEq_of_rows h (row := 594) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_15
  simp only [← c37677, ← c37678, k2, ← c37679] at e_594_15
  have hr := e_594_15
  simp only [e_594_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3113)) := by
  have c37680 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37681 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37682 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37683 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37684 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37685 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_16 := arithEq_of_rows h (row := 594) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_16
  simp only [← c37680, ← c37681, ← c37682] at e_594_16
  have e_594_17 := arithEq_of_rows h (row := 594) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_17
  simp only [← c37683, ← c37684, k2, ← c37685] at e_594_17
  have hr := e_594_17
  simp only [e_594_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 594 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3114)) := by
  have c37686 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37687 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37688 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37689 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37690 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37691 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_594_18 := arithEq_of_rows h (row := 594) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_18
  simp only [← c37686, ← c37687, ← c37688] at e_594_18
  have e_594_19 := arithEq_of_rows h (row := 594) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_19
  simp only [← c37689, ← c37690, k2, ← c37691] at e_594_19
  have hr := e_594_19
  simp only [e_594_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3115)) := by
  have c37692 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37693 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37694 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37695 := (publicBatchWrapper89_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37696 := (publicBatchWrapper89_copies1178 a h).1
  have c37697 := (publicBatchWrapper89_copies1178 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_0 := arithEq_of_rows h (row := 595) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_0
  simp only [← c37692, ← c37693, ← c37694] at e_595_0
  have e_595_1 := arithEq_of_rows h (row := 595) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_1
  simp only [← c37695, ← c37696, k2, ← c37697] at e_595_1
  have hr := e_595_1
  simp only [e_595_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3116)) := by
  have c37698 := (publicBatchWrapper89_copies1178 a h).2.2.1
  have c37699 := (publicBatchWrapper89_copies1178 a h).2.2.2.1
  have c37700 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.1
  have c37701 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.1
  have c37702 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.1
  have c37703 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_2 := arithEq_of_rows h (row := 595) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_2
  simp only [← c37698, ← c37699, ← c37700] at e_595_2
  have e_595_3 := arithEq_of_rows h (row := 595) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_3
  simp only [← c37701, ← c37702, k2, ← c37703] at e_595_3
  have hr := e_595_3
  simp only [e_595_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3117)) := by
  have c37704 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.1
  have c37705 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.1
  have c37706 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37707 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37708 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37709 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_4 := arithEq_of_rows h (row := 595) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_4
  simp only [← c37704, ← c37705, ← c37706] at e_595_4
  have e_595_5 := arithEq_of_rows h (row := 595) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_5
  simp only [← c37707, ← c37708, k2, ← c37709] at e_595_5
  have hr := e_595_5
  simp only [e_595_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3118)) := by
  have c37710 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37711 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37712 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37713 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37714 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37715 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_6 := arithEq_of_rows h (row := 595) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_6
  simp only [← c37710, ← c37711, ← c37712] at e_595_6
  have e_595_7 := arithEq_of_rows h (row := 595) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_7
  simp only [← c37713, ← c37714, k2, ← c37715] at e_595_7
  have hr := e_595_7
  simp only [e_595_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3119)) := by
  have c37716 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37717 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37718 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37719 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37720 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37721 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_8 := arithEq_of_rows h (row := 595) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_8
  simp only [← c37716, ← c37717, ← c37718] at e_595_8
  have e_595_9 := arithEq_of_rows h (row := 595) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_9
  simp only [← c37719, ← c37720, k2, ← c37721] at e_595_9
  have hr := e_595_9
  simp only [e_595_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3120)) := by
  have c37722 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37723 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37724 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37725 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37726 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37727 := (publicBatchWrapper89_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_10 := arithEq_of_rows h (row := 595) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_10
  simp only [← c37722, ← c37723, ← c37724] at e_595_10
  have e_595_11 := arithEq_of_rows h (row := 595) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_11
  simp only [← c37725, ← c37726, k2, ← c37727] at e_595_11
  have hr := e_595_11
  simp only [e_595_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3121)) := by
  have c37728 := (publicBatchWrapper89_copies1179 a h).1
  have c37729 := (publicBatchWrapper89_copies1179 a h).2.1
  have c37730 := (publicBatchWrapper89_copies1179 a h).2.2.1
  have c37731 := (publicBatchWrapper89_copies1179 a h).2.2.2.1
  have c37732 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.1
  have c37733 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_12 := arithEq_of_rows h (row := 595) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_12
  simp only [← c37728, ← c37729, ← c37730] at e_595_12
  have e_595_13 := arithEq_of_rows h (row := 595) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_13
  simp only [← c37731, ← c37732, k2, ← c37733] at e_595_13
  have hr := e_595_13
  simp only [e_595_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3122)) := by
  have c37734 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.1
  have c37735 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.1
  have c37736 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.1
  have c37737 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.1
  have c37738 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37739 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_14 := arithEq_of_rows h (row := 595) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_14
  simp only [← c37734, ← c37735, ← c37736] at e_595_14
  have e_595_15 := arithEq_of_rows h (row := 595) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_15
  simp only [← c37737, ← c37738, k2, ← c37739] at e_595_15
  have hr := e_595_15
  simp only [e_595_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3123)) := by
  have c37740 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37741 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37742 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37743 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37744 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37745 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_16 := arithEq_of_rows h (row := 595) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_16
  simp only [← c37740, ← c37741, ← c37742] at e_595_16
  have e_595_17 := arithEq_of_rows h (row := 595) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_17
  simp only [← c37743, ← c37744, k2, ← c37745] at e_595_17
  have hr := e_595_17
  simp only [e_595_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 595 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3124)) := by
  have c37746 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37747 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37748 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37749 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37750 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37751 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_595_18 := arithEq_of_rows h (row := 595) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_18
  simp only [← c37746, ← c37747, ← c37748] at e_595_18
  have e_595_19 := arithEq_of_rows h (row := 595) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_19
  simp only [← c37749, ← c37750, k2, ← c37751] at e_595_19
  have hr := e_595_19
  simp only [e_595_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3125)) := by
  have c37752 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37753 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37754 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37755 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37756 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37757 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_0 := arithEq_of_rows h (row := 596) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_0
  simp only [← c37752, ← c37753, ← c37754] at e_596_0
  have e_596_1 := arithEq_of_rows h (row := 596) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_1
  simp only [← c37755, ← c37756, k2, ← c37757] at e_596_1
  have hr := e_596_1
  simp only [e_596_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3126)) := by
  have c37758 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37759 := (publicBatchWrapper89_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37760 := (publicBatchWrapper89_copies1180 a h).1
  have c37761 := (publicBatchWrapper89_copies1180 a h).2.1
  have c37762 := (publicBatchWrapper89_copies1180 a h).2.2.1
  have c37763 := (publicBatchWrapper89_copies1180 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_2 := arithEq_of_rows h (row := 596) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_2
  simp only [← c37758, ← c37759, ← c37760] at e_596_2
  have e_596_3 := arithEq_of_rows h (row := 596) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_3
  simp only [← c37761, ← c37762, k2, ← c37763] at e_596_3
  have hr := e_596_3
  simp only [e_596_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3127)) := by
  have c37764 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.1
  have c37765 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.1
  have c37766 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.1
  have c37767 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.1
  have c37768 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.1
  have c37769 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_4 := arithEq_of_rows h (row := 596) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_4
  simp only [← c37764, ← c37765, ← c37766] at e_596_4
  have e_596_5 := arithEq_of_rows h (row := 596) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_5
  simp only [← c37767, ← c37768, k2, ← c37769] at e_596_5
  have hr := e_596_5
  simp only [e_596_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3128)) := by
  have c37770 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37771 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37772 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37773 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37774 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37775 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_6 := arithEq_of_rows h (row := 596) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_6
  simp only [← c37770, ← c37771, ← c37772] at e_596_6
  have e_596_7 := arithEq_of_rows h (row := 596) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_7
  simp only [← c37773, ← c37774, k2, ← c37775] at e_596_7
  have hr := e_596_7
  simp only [e_596_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3129)) := by
  have c37776 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37777 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37778 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37779 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37780 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37781 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_8 := arithEq_of_rows h (row := 596) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_8
  simp only [← c37776, ← c37777, ← c37778] at e_596_8
  have e_596_9 := arithEq_of_rows h (row := 596) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_9
  simp only [← c37779, ← c37780, k2, ← c37781] at e_596_9
  have hr := e_596_9
  simp only [e_596_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3130)) := by
  have c37782 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37783 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37784 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37785 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37786 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37787 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_10 := arithEq_of_rows h (row := 596) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_10
  simp only [← c37782, ← c37783, ← c37784] at e_596_10
  have e_596_11 := arithEq_of_rows h (row := 596) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_11
  simp only [← c37785, ← c37786, k2, ← c37787] at e_596_11
  have hr := e_596_11
  simp only [e_596_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3131)) := by
  have c37788 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37789 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37790 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37791 := (publicBatchWrapper89_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37792 := (publicBatchWrapper89_copies1181 a h).1
  have c37793 := (publicBatchWrapper89_copies1181 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_12 := arithEq_of_rows h (row := 596) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_12
  simp only [← c37788, ← c37789, ← c37790] at e_596_12
  have e_596_13 := arithEq_of_rows h (row := 596) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_13
  simp only [← c37791, ← c37792, k2, ← c37793] at e_596_13
  have hr := e_596_13
  simp only [e_596_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3132)) := by
  have c37794 := (publicBatchWrapper89_copies1181 a h).2.2.1
  have c37795 := (publicBatchWrapper89_copies1181 a h).2.2.2.1
  have c37796 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.1
  have c37797 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.1
  have c37798 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.1
  have c37799 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_14 := arithEq_of_rows h (row := 596) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_14
  simp only [← c37794, ← c37795, ← c37796] at e_596_14
  have e_596_15 := arithEq_of_rows h (row := 596) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_15
  simp only [← c37797, ← c37798, k2, ← c37799] at e_596_15
  have hr := e_596_15
  simp only [e_596_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3133)) := by
  have c37800 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.1
  have c37801 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.1
  have c37802 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37803 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37804 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37805 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_16 := arithEq_of_rows h (row := 596) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_16
  simp only [← c37800, ← c37801, ← c37802] at e_596_16
  have e_596_17 := arithEq_of_rows h (row := 596) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_17
  simp only [← c37803, ← c37804, k2, ← c37805] at e_596_17
  have hr := e_596_17
  simp only [e_596_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 596 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3134)) := by
  have c37806 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37807 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37808 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37809 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37810 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37811 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_596_18 := arithEq_of_rows h (row := 596) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_18
  simp only [← c37806, ← c37807, ← c37808] at e_596_18
  have e_596_19 := arithEq_of_rows h (row := 596) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_19
  simp only [← c37809, ← c37810, k2, ← c37811] at e_596_19
  have hr := e_596_19
  simp only [e_596_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3135)) := by
  have c37812 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37813 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37814 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37815 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37816 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37817 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_0 := arithEq_of_rows h (row := 597) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_0
  simp only [← c37812, ← c37813, ← c37814] at e_597_0
  have e_597_1 := arithEq_of_rows h (row := 597) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_1
  simp only [← c37815, ← c37816, k2, ← c37817] at e_597_1
  have hr := e_597_1
  simp only [e_597_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3136)) := by
  have c37818 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37819 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37820 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37821 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37822 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37823 := (publicBatchWrapper89_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_2 := arithEq_of_rows h (row := 597) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_2
  simp only [← c37818, ← c37819, ← c37820] at e_597_2
  have e_597_3 := arithEq_of_rows h (row := 597) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_3
  simp only [← c37821, ← c37822, k2, ← c37823] at e_597_3
  have hr := e_597_3
  simp only [e_597_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3137)) := by
  have c37824 := (publicBatchWrapper89_copies1182 a h).1
  have c37825 := (publicBatchWrapper89_copies1182 a h).2.1
  have c37826 := (publicBatchWrapper89_copies1182 a h).2.2.1
  have c37827 := (publicBatchWrapper89_copies1182 a h).2.2.2.1
  have c37828 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.1
  have c37829 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_4 := arithEq_of_rows h (row := 597) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_4
  simp only [← c37824, ← c37825, ← c37826] at e_597_4
  have e_597_5 := arithEq_of_rows h (row := 597) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_5
  simp only [← c37827, ← c37828, k2, ← c37829] at e_597_5
  have hr := e_597_5
  simp only [e_597_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3138)) := by
  have c37830 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.1
  have c37831 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.1
  have c37832 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.1
  have c37833 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.1
  have c37834 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37835 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_6 := arithEq_of_rows h (row := 597) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_6
  simp only [← c37830, ← c37831, ← c37832] at e_597_6
  have e_597_7 := arithEq_of_rows h (row := 597) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_7
  simp only [← c37833, ← c37834, k2, ← c37835] at e_597_7
  have hr := e_597_7
  simp only [e_597_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3139)) := by
  have c37836 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37837 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37838 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37839 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37840 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37841 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_8 := arithEq_of_rows h (row := 597) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_8
  simp only [← c37836, ← c37837, ← c37838] at e_597_8
  have e_597_9 := arithEq_of_rows h (row := 597) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_9
  simp only [← c37839, ← c37840, k2, ← c37841] at e_597_9
  have hr := e_597_9
  simp only [e_597_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3140)) := by
  have c37842 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37843 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37844 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37845 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37846 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37847 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_10 := arithEq_of_rows h (row := 597) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_10
  simp only [← c37842, ← c37843, ← c37844] at e_597_10
  have e_597_11 := arithEq_of_rows h (row := 597) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_11
  simp only [← c37845, ← c37846, k2, ← c37847] at e_597_11
  have hr := e_597_11
  simp only [e_597_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3141)) := by
  have c37848 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37849 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37850 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37851 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37852 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37853 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_12 := arithEq_of_rows h (row := 597) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_12
  simp only [← c37848, ← c37849, ← c37850] at e_597_12
  have e_597_13 := arithEq_of_rows h (row := 597) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_13
  simp only [← c37851, ← c37852, k2, ← c37853] at e_597_13
  have hr := e_597_13
  simp only [e_597_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3142)) := by
  have c37854 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37855 := (publicBatchWrapper89_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37856 := (publicBatchWrapper89_copies1183 a h).1
  have c37857 := (publicBatchWrapper89_copies1183 a h).2.1
  have c37858 := (publicBatchWrapper89_copies1183 a h).2.2.1
  have c37859 := (publicBatchWrapper89_copies1183 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_14 := arithEq_of_rows h (row := 597) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_14
  simp only [← c37854, ← c37855, ← c37856] at e_597_14
  have e_597_15 := arithEq_of_rows h (row := 597) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_15
  simp only [← c37857, ← c37858, k2, ← c37859] at e_597_15
  have hr := e_597_15
  simp only [e_597_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3143)) := by
  have c37860 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.1
  have c37861 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.1
  have c37862 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.1
  have c37863 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.1
  have c37864 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.1
  have c37865 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_16 := arithEq_of_rows h (row := 597) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_16
  simp only [← c37860, ← c37861, ← c37862] at e_597_16
  have e_597_17 := arithEq_of_rows h (row := 597) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_17
  simp only [← c37863, ← c37864, k2, ← c37865] at e_597_17
  have hr := e_597_17
  simp only [e_597_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 597 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3144)) := by
  have c37866 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37867 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37868 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37869 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37870 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37871 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_597_18 := arithEq_of_rows h (row := 597) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_18
  simp only [← c37866, ← c37867, ← c37868] at e_597_18
  have e_597_19 := arithEq_of_rows h (row := 597) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_19
  simp only [← c37869, ← c37870, k2, ← c37871] at e_597_19
  have hr := e_597_19
  simp only [e_597_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3145)) := by
  have c37872 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37873 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37874 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37875 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37876 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37877 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_0 := arithEq_of_rows h (row := 598) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_0
  simp only [← c37872, ← c37873, ← c37874] at e_598_0
  have e_598_1 := arithEq_of_rows h (row := 598) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_1
  simp only [← c37875, ← c37876, k2, ← c37877] at e_598_1
  have hr := e_598_1
  simp only [e_598_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3146)) := by
  have c37878 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37879 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37880 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37881 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37882 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37883 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_2 := arithEq_of_rows h (row := 598) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_2
  simp only [← c37878, ← c37879, ← c37880] at e_598_2
  have e_598_3 := arithEq_of_rows h (row := 598) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_3
  simp only [← c37881, ← c37882, k2, ← c37883] at e_598_3
  have hr := e_598_3
  simp only [e_598_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3147)) := by
  have c37884 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37885 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37886 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37887 := (publicBatchWrapper89_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37888 := (publicBatchWrapper89_copies1184 a h).1
  have c37889 := (publicBatchWrapper89_copies1184 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_4 := arithEq_of_rows h (row := 598) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_4
  simp only [← c37884, ← c37885, ← c37886] at e_598_4
  have e_598_5 := arithEq_of_rows h (row := 598) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_5
  simp only [← c37887, ← c37888, k2, ← c37889] at e_598_5
  have hr := e_598_5
  simp only [e_598_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3148)) := by
  have c37890 := (publicBatchWrapper89_copies1184 a h).2.2.1
  have c37891 := (publicBatchWrapper89_copies1184 a h).2.2.2.1
  have c37892 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.1
  have c37893 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.1
  have c37894 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.1
  have c37895 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_6 := arithEq_of_rows h (row := 598) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_6
  simp only [← c37890, ← c37891, ← c37892] at e_598_6
  have e_598_7 := arithEq_of_rows h (row := 598) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_7
  simp only [← c37893, ← c37894, k2, ← c37895] at e_598_7
  have hr := e_598_7
  simp only [e_598_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3149)) := by
  have c37896 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.1
  have c37897 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.1
  have c37898 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37899 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37900 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37901 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_8 := arithEq_of_rows h (row := 598) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_8
  simp only [← c37896, ← c37897, ← c37898] at e_598_8
  have e_598_9 := arithEq_of_rows h (row := 598) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_9
  simp only [← c37899, ← c37900, k2, ← c37901] at e_598_9
  have hr := e_598_9
  simp only [e_598_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3150)) := by
  have c37902 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37903 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37904 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37905 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37906 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37907 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_10 := arithEq_of_rows h (row := 598) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_10
  simp only [← c37902, ← c37903, ← c37904] at e_598_10
  have e_598_11 := arithEq_of_rows h (row := 598) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_11
  simp only [← c37905, ← c37906, k2, ← c37907] at e_598_11
  have hr := e_598_11
  simp only [e_598_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3151)) := by
  have c37908 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37909 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37910 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37911 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37912 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37913 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_12 := arithEq_of_rows h (row := 598) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_12
  simp only [← c37908, ← c37909, ← c37910] at e_598_12
  have e_598_13 := arithEq_of_rows h (row := 598) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_13
  simp only [← c37911, ← c37912, k2, ← c37913] at e_598_13
  have hr := e_598_13
  simp only [e_598_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3152)) := by
  have c37914 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37915 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37916 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37917 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37918 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37919 := (publicBatchWrapper89_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_14 := arithEq_of_rows h (row := 598) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_14
  simp only [← c37914, ← c37915, ← c37916] at e_598_14
  have e_598_15 := arithEq_of_rows h (row := 598) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_15
  simp only [← c37917, ← c37918, k2, ← c37919] at e_598_15
  have hr := e_598_15
  simp only [e_598_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3153)) := by
  have c37920 := (publicBatchWrapper89_copies1185 a h).1
  have c37921 := (publicBatchWrapper89_copies1185 a h).2.1
  have c37922 := (publicBatchWrapper89_copies1185 a h).2.2.1
  have c37923 := (publicBatchWrapper89_copies1185 a h).2.2.2.1
  have c37924 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.1
  have c37925 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_16 := arithEq_of_rows h (row := 598) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_16
  simp only [← c37920, ← c37921, ← c37922] at e_598_16
  have e_598_17 := arithEq_of_rows h (row := 598) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_17
  simp only [← c37923, ← c37924, k2, ← c37925] at e_598_17
  have hr := e_598_17
  simp only [e_598_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 598 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3154)) := by
  have c37926 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.1
  have c37927 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.1
  have c37928 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.1
  have c37929 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.1
  have c37930 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37931 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_598_18 := arithEq_of_rows h (row := 598) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_18
  simp only [← c37926, ← c37927, ← c37928] at e_598_18
  have e_598_19 := arithEq_of_rows h (row := 598) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_19
  simp only [← c37929, ← c37930, k2, ← c37931] at e_598_19
  have hr := e_598_19
  simp only [e_598_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3155)) := by
  have c37932 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37933 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37934 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37935 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37936 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37937 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_0 := arithEq_of_rows h (row := 599) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_0
  simp only [← c37932, ← c37933, ← c37934] at e_599_0
  have e_599_1 := arithEq_of_rows h (row := 599) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_1
  simp only [← c37935, ← c37936, k2, ← c37937] at e_599_1
  have hr := e_599_1
  simp only [e_599_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3156)) := by
  have c37938 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37939 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37940 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37941 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37942 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37943 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_2 := arithEq_of_rows h (row := 599) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_2
  simp only [← c37938, ← c37939, ← c37940] at e_599_2
  have e_599_3 := arithEq_of_rows h (row := 599) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_3
  simp only [← c37941, ← c37942, k2, ← c37943] at e_599_3
  have hr := e_599_3
  simp only [e_599_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3157)) := by
  have c37944 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37945 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37946 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37947 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37948 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37949 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_4 := arithEq_of_rows h (row := 599) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_4
  simp only [← c37944, ← c37945, ← c37946] at e_599_4
  have e_599_5 := arithEq_of_rows h (row := 599) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_5
  simp only [← c37947, ← c37948, k2, ← c37949] at e_599_5
  have hr := e_599_5
  simp only [e_599_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3158)) := by
  have c37950 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37951 := (publicBatchWrapper89_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37952 := (publicBatchWrapper89_copies1186 a h).1
  have c37953 := (publicBatchWrapper89_copies1186 a h).2.1
  have c37954 := (publicBatchWrapper89_copies1186 a h).2.2.1
  have c37955 := (publicBatchWrapper89_copies1186 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_6 := arithEq_of_rows h (row := 599) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_6
  simp only [← c37950, ← c37951, ← c37952] at e_599_6
  have e_599_7 := arithEq_of_rows h (row := 599) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_7
  simp only [← c37953, ← c37954, k2, ← c37955] at e_599_7
  have hr := e_599_7
  simp only [e_599_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3159)) := by
  have c37956 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.1
  have c37957 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.1
  have c37958 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.1
  have c37959 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.1
  have c37960 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.1
  have c37961 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_8 := arithEq_of_rows h (row := 599) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_8
  simp only [← c37956, ← c37957, ← c37958] at e_599_8
  have e_599_9 := arithEq_of_rows h (row := 599) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_9
  simp only [← c37959, ← c37960, k2, ← c37961] at e_599_9
  have hr := e_599_9
  simp only [e_599_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3160)) := by
  have c37962 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37963 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37964 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37965 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37966 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37967 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_10 := arithEq_of_rows h (row := 599) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_10
  simp only [← c37962, ← c37963, ← c37964] at e_599_10
  have e_599_11 := arithEq_of_rows h (row := 599) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_11
  simp only [← c37965, ← c37966, k2, ← c37967] at e_599_11
  have hr := e_599_11
  simp only [e_599_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3161)) := by
  have c37968 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37969 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37970 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37971 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37972 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37973 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_12 := arithEq_of_rows h (row := 599) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_12
  simp only [← c37968, ← c37969, ← c37970] at e_599_12
  have e_599_13 := arithEq_of_rows h (row := 599) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_13
  simp only [← c37971, ← c37972, k2, ← c37973] at e_599_13
  have hr := e_599_13
  simp only [e_599_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3162)) := by
  have c37974 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37975 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37976 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37977 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37978 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37979 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_14 := arithEq_of_rows h (row := 599) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_14
  simp only [← c37974, ← c37975, ← c37976] at e_599_14
  have e_599_15 := arithEq_of_rows h (row := 599) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_15
  simp only [← c37977, ← c37978, k2, ← c37979] at e_599_15
  have hr := e_599_15
  simp only [e_599_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3163)) := by
  have c37980 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37981 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37982 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37983 := (publicBatchWrapper89_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37984 := (publicBatchWrapper89_copies1187 a h).1
  have c37985 := (publicBatchWrapper89_copies1187 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_16 := arithEq_of_rows h (row := 599) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_16
  simp only [← c37980, ← c37981, ← c37982] at e_599_16
  have e_599_17 := arithEq_of_rows h (row := 599) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_17
  simp only [← c37983, ← c37984, k2, ← c37985] at e_599_17
  have hr := e_599_17
  simp only [e_599_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 599 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3164)) := by
  have c37986 := (publicBatchWrapper89_copies1187 a h).2.2.1
  have c37987 := (publicBatchWrapper89_copies1187 a h).2.2.2.1
  have c37988 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.1
  have c37989 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.1
  have c37990 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.1
  have c37991 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_599_18 := arithEq_of_rows h (row := 599) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_18
  simp only [← c37986, ← c37987, ← c37988] at e_599_18
  have e_599_19 := arithEq_of_rows h (row := 599) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_19
  simp only [← c37989, ← c37990, k2, ← c37991] at e_599_19
  have hr := e_599_19
  simp only [e_599_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3165)) := by
  have c37992 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.1
  have c37993 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.1
  have c37994 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37995 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37996 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37997 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_0 := arithEq_of_rows h (row := 600) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_0
  simp only [← c37992, ← c37993, ← c37994] at e_600_0
  have e_600_1 := arithEq_of_rows h (row := 600) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_1
  simp only [← c37995, ← c37996, k2, ← c37997] at e_600_1
  have hr := e_600_1
  simp only [e_600_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3166)) := by
  have c37998 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37999 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38000 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38001 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38002 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38003 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_2 := arithEq_of_rows h (row := 600) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_2
  simp only [← c37998, ← c37999, ← c38000] at e_600_2
  have e_600_3 := arithEq_of_rows h (row := 600) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_3
  simp only [← c38001, ← c38002, k2, ← c38003] at e_600_3
  have hr := e_600_3
  simp only [e_600_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3167)) := by
  have c38004 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38005 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38006 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38007 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38008 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38009 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_4 := arithEq_of_rows h (row := 600) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_4
  simp only [← c38004, ← c38005, ← c38006] at e_600_4
  have e_600_5 := arithEq_of_rows h (row := 600) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_5
  simp only [← c38007, ← c38008, k2, ← c38009] at e_600_5
  have hr := e_600_5
  simp only [e_600_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3168)) := by
  have c38010 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38011 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38012 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38013 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38014 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38015 := (publicBatchWrapper89_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_6 := arithEq_of_rows h (row := 600) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_6
  simp only [← c38010, ← c38011, ← c38012] at e_600_6
  have e_600_7 := arithEq_of_rows h (row := 600) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_7
  simp only [← c38013, ← c38014, k2, ← c38015] at e_600_7
  have hr := e_600_7
  simp only [e_600_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3169)) := by
  have c38016 := (publicBatchWrapper89_copies1188 a h).1
  have c38017 := (publicBatchWrapper89_copies1188 a h).2.1
  have c38018 := (publicBatchWrapper89_copies1188 a h).2.2.1
  have c38019 := (publicBatchWrapper89_copies1188 a h).2.2.2.1
  have c38020 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.1
  have c38021 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_8 := arithEq_of_rows h (row := 600) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_8
  simp only [← c38016, ← c38017, ← c38018] at e_600_8
  have e_600_9 := arithEq_of_rows h (row := 600) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_9
  simp only [← c38019, ← c38020, k2, ← c38021] at e_600_9
  have hr := e_600_9
  simp only [e_600_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3210)) := by
  have c38022 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.1
  have c38023 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.1
  have c38024 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.1
  have c38025 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.1
  have c38026 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38027 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_10 := arithEq_of_rows h (row := 600) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_10
  simp only [← c38022, ← c38023, ← c38024] at e_600_10
  have e_600_11 := arithEq_of_rows h (row := 600) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_11
  simp only [← c38025, ← c38026, k2, ← c38027] at e_600_11
  have hr := e_600_11
  simp only [e_600_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3211)) := by
  have c38028 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38029 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38030 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38031 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38032 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38033 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_12 := arithEq_of_rows h (row := 600) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_12
  simp only [← c38028, ← c38029, ← c38030] at e_600_12
  have e_600_13 := arithEq_of_rows h (row := 600) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_13
  simp only [← c38031, ← c38032, k2, ← c38033] at e_600_13
  have hr := e_600_13
  simp only [e_600_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3212)) := by
  have c38034 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38035 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38036 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38037 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38038 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38039 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_14 := arithEq_of_rows h (row := 600) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_14
  simp only [← c38034, ← c38035, ← c38036] at e_600_14
  have e_600_15 := arithEq_of_rows h (row := 600) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_15
  simp only [← c38037, ← c38038, k2, ← c38039] at e_600_15
  have hr := e_600_15
  simp only [e_600_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3213)) := by
  have c38040 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38041 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38042 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38043 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38044 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38045 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_16 := arithEq_of_rows h (row := 600) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_16
  simp only [← c38040, ← c38041, ← c38042] at e_600_16
  have e_600_17 := arithEq_of_rows h (row := 600) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_17
  simp only [← c38043, ← c38044, k2, ← c38045] at e_600_17
  have hr := e_600_17
  simp only [e_600_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 600 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3214)) := by
  have c38046 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38047 := (publicBatchWrapper89_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38048 := (publicBatchWrapper89_copies1189 a h).1
  have c38049 := (publicBatchWrapper89_copies1189 a h).2.1
  have c38050 := (publicBatchWrapper89_copies1189 a h).2.2.1
  have c38051 := (publicBatchWrapper89_copies1189 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_600_18 := arithEq_of_rows h (row := 600) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_18
  simp only [← c38046, ← c38047, ← c38048] at e_600_18
  have e_600_19 := arithEq_of_rows h (row := 600) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_19
  simp only [← c38049, ← c38050, k2, ← c38051] at e_600_19
  have hr := e_600_19
  simp only [e_600_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3215)) := by
  have c38052 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.1
  have c38053 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.1
  have c38054 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.1
  have c38055 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.1
  have c38056 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.1
  have c38057 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_0 := arithEq_of_rows h (row := 601) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_0
  simp only [← c38052, ← c38053, ← c38054] at e_601_0
  have e_601_1 := arithEq_of_rows h (row := 601) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_1
  simp only [← c38055, ← c38056, k2, ← c38057] at e_601_1
  have hr := e_601_1
  simp only [e_601_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3216)) := by
  have c38058 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38059 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38060 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38061 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38062 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38063 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_2 := arithEq_of_rows h (row := 601) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_2
  simp only [← c38058, ← c38059, ← c38060] at e_601_2
  have e_601_3 := arithEq_of_rows h (row := 601) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_3
  simp only [← c38061, ← c38062, k2, ← c38063] at e_601_3
  have hr := e_601_3
  simp only [e_601_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3217)) := by
  have c38064 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38065 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38066 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38067 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38068 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38069 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_4 := arithEq_of_rows h (row := 601) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_4
  simp only [← c38064, ← c38065, ← c38066] at e_601_4
  have e_601_5 := arithEq_of_rows h (row := 601) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_5
  simp only [← c38067, ← c38068, k2, ← c38069] at e_601_5
  have hr := e_601_5
  simp only [e_601_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3218)) := by
  have c38070 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38071 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38072 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38073 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38074 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38075 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_6 := arithEq_of_rows h (row := 601) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_6
  simp only [← c38070, ← c38071, ← c38072] at e_601_6
  have e_601_7 := arithEq_of_rows h (row := 601) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_7
  simp only [← c38073, ← c38074, k2, ← c38075] at e_601_7
  have hr := e_601_7
  simp only [e_601_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3219)) := by
  have c38076 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38077 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38078 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38079 := (publicBatchWrapper89_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38080 := (publicBatchWrapper89_copies1190 a h).1
  have c38081 := (publicBatchWrapper89_copies1190 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_8 := arithEq_of_rows h (row := 601) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_8
  simp only [← c38076, ← c38077, ← c38078] at e_601_8
  have e_601_9 := arithEq_of_rows h (row := 601) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_9
  simp only [← c38079, ← c38080, k2, ← c38081] at e_601_9
  have hr := e_601_9
  simp only [e_601_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3220)) := by
  have c38082 := (publicBatchWrapper89_copies1190 a h).2.2.1
  have c38083 := (publicBatchWrapper89_copies1190 a h).2.2.2.1
  have c38084 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.1
  have c38085 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.1
  have c38086 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.1
  have c38087 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_10 := arithEq_of_rows h (row := 601) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_10
  simp only [← c38082, ← c38083, ← c38084] at e_601_10
  have e_601_11 := arithEq_of_rows h (row := 601) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_11
  simp only [← c38085, ← c38086, k2, ← c38087] at e_601_11
  have hr := e_601_11
  simp only [e_601_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3221)) := by
  have c38088 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.1
  have c38089 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.1
  have c38090 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38091 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38092 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38093 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_12 := arithEq_of_rows h (row := 601) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_12
  simp only [← c38088, ← c38089, ← c38090] at e_601_12
  have e_601_13 := arithEq_of_rows h (row := 601) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_13
  simp only [← c38091, ← c38092, k2, ← c38093] at e_601_13
  have hr := e_601_13
  simp only [e_601_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3222)) := by
  have c38094 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38095 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38096 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38097 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38098 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38099 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_14 := arithEq_of_rows h (row := 601) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_14
  simp only [← c38094, ← c38095, ← c38096] at e_601_14
  have e_601_15 := arithEq_of_rows h (row := 601) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_15
  simp only [← c38097, ← c38098, k2, ← c38099] at e_601_15
  have hr := e_601_15
  simp only [e_601_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3223)) := by
  have c38100 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38101 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38102 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38103 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38104 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38105 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_16 := arithEq_of_rows h (row := 601) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_16
  simp only [← c38100, ← c38101, ← c38102] at e_601_16
  have e_601_17 := arithEq_of_rows h (row := 601) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_17
  simp only [← c38103, ← c38104, k2, ← c38105] at e_601_17
  have hr := e_601_17
  simp only [e_601_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 601 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3224)) := by
  have c38106 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38107 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38108 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38109 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38110 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38111 := (publicBatchWrapper89_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_601_18 := arithEq_of_rows h (row := 601) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_18
  simp only [← c38106, ← c38107, ← c38108] at e_601_18
  have e_601_19 := arithEq_of_rows h (row := 601) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_19
  simp only [← c38109, ← c38110, k2, ← c38111] at e_601_19
  have hr := e_601_19
  simp only [e_601_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3225)) := by
  have c38112 := (publicBatchWrapper89_copies1191 a h).1
  have c38113 := (publicBatchWrapper89_copies1191 a h).2.1
  have c38114 := (publicBatchWrapper89_copies1191 a h).2.2.1
  have c38115 := (publicBatchWrapper89_copies1191 a h).2.2.2.1
  have c38116 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.1
  have c38117 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_0 := arithEq_of_rows h (row := 602) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_0
  simp only [← c38112, ← c38113, ← c38114] at e_602_0
  have e_602_1 := arithEq_of_rows h (row := 602) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_1
  simp only [← c38115, ← c38116, k2, ← c38117] at e_602_1
  have hr := e_602_1
  simp only [e_602_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3226)) := by
  have c38118 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.1
  have c38119 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.1
  have c38120 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.1
  have c38121 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.1
  have c38122 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38123 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_2 := arithEq_of_rows h (row := 602) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_2
  simp only [← c38118, ← c38119, ← c38120] at e_602_2
  have e_602_3 := arithEq_of_rows h (row := 602) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_3
  simp only [← c38121, ← c38122, k2, ← c38123] at e_602_3
  have hr := e_602_3
  simp only [e_602_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3227)) := by
  have c38124 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38125 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38126 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38127 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38128 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38129 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_4 := arithEq_of_rows h (row := 602) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_4
  simp only [← c38124, ← c38125, ← c38126] at e_602_4
  have e_602_5 := arithEq_of_rows h (row := 602) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_5
  simp only [← c38127, ← c38128, k2, ← c38129] at e_602_5
  have hr := e_602_5
  simp only [e_602_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3228)) := by
  have c38130 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38131 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38132 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38133 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38134 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38135 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_6 := arithEq_of_rows h (row := 602) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_6
  simp only [← c38130, ← c38131, ← c38132] at e_602_6
  have e_602_7 := arithEq_of_rows h (row := 602) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_7
  simp only [← c38133, ← c38134, k2, ← c38135] at e_602_7
  have hr := e_602_7
  simp only [e_602_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3229)) := by
  have c38136 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38137 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38138 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38139 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38140 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38141 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_8 := arithEq_of_rows h (row := 602) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_8
  simp only [← c38136, ← c38137, ← c38138] at e_602_8
  have e_602_9 := arithEq_of_rows h (row := 602) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_9
  simp only [← c38139, ← c38140, k2, ← c38141] at e_602_9
  have hr := e_602_9
  simp only [e_602_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3230)) := by
  have c38142 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38143 := (publicBatchWrapper89_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38144 := (publicBatchWrapper89_copies1192 a h).1
  have c38145 := (publicBatchWrapper89_copies1192 a h).2.1
  have c38146 := (publicBatchWrapper89_copies1192 a h).2.2.1
  have c38147 := (publicBatchWrapper89_copies1192 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_10 := arithEq_of_rows h (row := 602) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_10
  simp only [← c38142, ← c38143, ← c38144] at e_602_10
  have e_602_11 := arithEq_of_rows h (row := 602) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_11
  simp only [← c38145, ← c38146, k2, ← c38147] at e_602_11
  have hr := e_602_11
  simp only [e_602_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3231)) := by
  have c38148 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.1
  have c38149 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.1
  have c38150 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.1
  have c38151 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.1
  have c38152 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.1
  have c38153 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_12 := arithEq_of_rows h (row := 602) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_12
  simp only [← c38148, ← c38149, ← c38150] at e_602_12
  have e_602_13 := arithEq_of_rows h (row := 602) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_13
  simp only [← c38151, ← c38152, k2, ← c38153] at e_602_13
  have hr := e_602_13
  simp only [e_602_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3232)) := by
  have c38154 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38155 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38156 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38157 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38158 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38159 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_14 := arithEq_of_rows h (row := 602) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_14
  simp only [← c38154, ← c38155, ← c38156] at e_602_14
  have e_602_15 := arithEq_of_rows h (row := 602) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_15
  simp only [← c38157, ← c38158, k2, ← c38159] at e_602_15
  have hr := e_602_15
  simp only [e_602_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3233)) := by
  have c38160 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38161 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38162 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38163 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38164 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38165 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_16 := arithEq_of_rows h (row := 602) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_16
  simp only [← c38160, ← c38161, ← c38162] at e_602_16
  have e_602_17 := arithEq_of_rows h (row := 602) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_17
  simp only [← c38163, ← c38164, k2, ← c38165] at e_602_17
  have hr := e_602_17
  simp only [e_602_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 602 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3234)) := by
  have c38166 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38167 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38168 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38169 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38170 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38171 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_602_18 := arithEq_of_rows h (row := 602) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_18
  simp only [← c38166, ← c38167, ← c38168] at e_602_18
  have e_602_19 := arithEq_of_rows h (row := 602) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_19
  simp only [← c38169, ← c38170, k2, ← c38171] at e_602_19
  have hr := e_602_19
  simp only [e_602_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3235)) := by
  have c38172 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38173 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38174 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38175 := (publicBatchWrapper89_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38176 := (publicBatchWrapper89_copies1193 a h).1
  have c38177 := (publicBatchWrapper89_copies1193 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_0 := arithEq_of_rows h (row := 603) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_0
  simp only [← c38172, ← c38173, ← c38174] at e_603_0
  have e_603_1 := arithEq_of_rows h (row := 603) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_1
  simp only [← c38175, ← c38176, k2, ← c38177] at e_603_1
  have hr := e_603_1
  simp only [e_603_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3236)) := by
  have c38178 := (publicBatchWrapper89_copies1193 a h).2.2.1
  have c38179 := (publicBatchWrapper89_copies1193 a h).2.2.2.1
  have c38180 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.1
  have c38181 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.1
  have c38182 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.1
  have c38183 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_2 := arithEq_of_rows h (row := 603) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_2
  simp only [← c38178, ← c38179, ← c38180] at e_603_2
  have e_603_3 := arithEq_of_rows h (row := 603) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_3
  simp only [← c38181, ← c38182, k2, ← c38183] at e_603_3
  have hr := e_603_3
  simp only [e_603_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3237)) := by
  have c38184 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.1
  have c38185 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.1
  have c38186 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38187 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38188 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38189 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_4 := arithEq_of_rows h (row := 603) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_4
  simp only [← c38184, ← c38185, ← c38186] at e_603_4
  have e_603_5 := arithEq_of_rows h (row := 603) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_5
  simp only [← c38187, ← c38188, k2, ← c38189] at e_603_5
  have hr := e_603_5
  simp only [e_603_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3238)) := by
  have c38190 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38191 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38192 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38193 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38194 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38195 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_6 := arithEq_of_rows h (row := 603) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_6
  simp only [← c38190, ← c38191, ← c38192] at e_603_6
  have e_603_7 := arithEq_of_rows h (row := 603) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_7
  simp only [← c38193, ← c38194, k2, ← c38195] at e_603_7
  have hr := e_603_7
  simp only [e_603_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3239)) := by
  have c38196 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38197 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38198 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38199 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38200 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38201 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_8 := arithEq_of_rows h (row := 603) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_8
  simp only [← c38196, ← c38197, ← c38198] at e_603_8
  have e_603_9 := arithEq_of_rows h (row := 603) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_9
  simp only [← c38199, ← c38200, k2, ← c38201] at e_603_9
  have hr := e_603_9
  simp only [e_603_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3240)) := by
  have c38202 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38203 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38204 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38205 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38206 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38207 := (publicBatchWrapper89_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_10 := arithEq_of_rows h (row := 603) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_10
  simp only [← c38202, ← c38203, ← c38204] at e_603_10
  have e_603_11 := arithEq_of_rows h (row := 603) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_11
  simp only [← c38205, ← c38206, k2, ← c38207] at e_603_11
  have hr := e_603_11
  simp only [e_603_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3241)) := by
  have c38208 := (publicBatchWrapper89_copies1194 a h).1
  have c38209 := (publicBatchWrapper89_copies1194 a h).2.1
  have c38210 := (publicBatchWrapper89_copies1194 a h).2.2.1
  have c38211 := (publicBatchWrapper89_copies1194 a h).2.2.2.1
  have c38212 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.1
  have c38213 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_12 := arithEq_of_rows h (row := 603) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_12
  simp only [← c38208, ← c38209, ← c38210] at e_603_12
  have e_603_13 := arithEq_of_rows h (row := 603) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_13
  simp only [← c38211, ← c38212, k2, ← c38213] at e_603_13
  have hr := e_603_13
  simp only [e_603_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3242)) := by
  have c38214 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.1
  have c38215 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.1
  have c38216 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.1
  have c38217 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.1
  have c38218 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38219 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_14 := arithEq_of_rows h (row := 603) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_14
  simp only [← c38214, ← c38215, ← c38216] at e_603_14
  have e_603_15 := arithEq_of_rows h (row := 603) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_15
  simp only [← c38217, ← c38218, k2, ← c38219] at e_603_15
  have hr := e_603_15
  simp only [e_603_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3243)) := by
  have c38220 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38221 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38222 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38223 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38224 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38225 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_16 := arithEq_of_rows h (row := 603) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_16
  simp only [← c38220, ← c38221, ← c38222] at e_603_16
  have e_603_17 := arithEq_of_rows h (row := 603) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_17
  simp only [← c38223, ← c38224, k2, ← c38225] at e_603_17
  have hr := e_603_17
  simp only [e_603_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 603 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3244)) := by
  have c38226 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38227 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38228 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38229 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38230 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38231 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_603_18 := arithEq_of_rows h (row := 603) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_18
  simp only [← c38226, ← c38227, ← c38228] at e_603_18
  have e_603_19 := arithEq_of_rows h (row := 603) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_19
  simp only [← c38229, ← c38230, k2, ← c38231] at e_603_19
  have hr := e_603_19
  simp only [e_603_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3245)) := by
  have c38232 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38233 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38234 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38235 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38236 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38237 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_0 := arithEq_of_rows h (row := 604) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_0
  simp only [← c38232, ← c38233, ← c38234] at e_604_0
  have e_604_1 := arithEq_of_rows h (row := 604) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_1
  simp only [← c38235, ← c38236, k2, ← c38237] at e_604_1
  have hr := e_604_1
  simp only [e_604_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3246)) := by
  have c38238 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38239 := (publicBatchWrapper89_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38240 := (publicBatchWrapper89_copies1195 a h).1
  have c38241 := (publicBatchWrapper89_copies1195 a h).2.1
  have c38242 := (publicBatchWrapper89_copies1195 a h).2.2.1
  have c38243 := (publicBatchWrapper89_copies1195 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_2 := arithEq_of_rows h (row := 604) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_2
  simp only [← c38238, ← c38239, ← c38240] at e_604_2
  have e_604_3 := arithEq_of_rows h (row := 604) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_3
  simp only [← c38241, ← c38242, k2, ← c38243] at e_604_3
  have hr := e_604_3
  simp only [e_604_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3247)) := by
  have c38244 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.1
  have c38245 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.1
  have c38246 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.1
  have c38247 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.1
  have c38248 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.1
  have c38249 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_4 := arithEq_of_rows h (row := 604) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_4
  simp only [← c38244, ← c38245, ← c38246] at e_604_4
  have e_604_5 := arithEq_of_rows h (row := 604) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_5
  simp only [← c38247, ← c38248, k2, ← c38249] at e_604_5
  have hr := e_604_5
  simp only [e_604_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3248)) := by
  have c38250 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38251 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38252 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38253 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38254 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38255 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_6 := arithEq_of_rows h (row := 604) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_6
  simp only [← c38250, ← c38251, ← c38252] at e_604_6
  have e_604_7 := arithEq_of_rows h (row := 604) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_7
  simp only [← c38253, ← c38254, k2, ← c38255] at e_604_7
  have hr := e_604_7
  simp only [e_604_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3249)) := by
  have c38256 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38257 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38258 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38259 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38260 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38261 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_8 := arithEq_of_rows h (row := 604) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_8
  simp only [← c38256, ← c38257, ← c38258] at e_604_8
  have e_604_9 := arithEq_of_rows h (row := 604) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_9
  simp only [← c38259, ← c38260, k2, ← c38261] at e_604_9
  have hr := e_604_9
  simp only [e_604_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3250)) := by
  have c38262 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38263 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38264 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38265 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38266 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38267 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_10 := arithEq_of_rows h (row := 604) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_10
  simp only [← c38262, ← c38263, ← c38264] at e_604_10
  have e_604_11 := arithEq_of_rows h (row := 604) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_11
  simp only [← c38265, ← c38266, k2, ← c38267] at e_604_11
  have hr := e_604_11
  simp only [e_604_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3251)) := by
  have c38268 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38269 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38270 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38271 := (publicBatchWrapper89_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38272 := (publicBatchWrapper89_copies1196 a h).1
  have c38273 := (publicBatchWrapper89_copies1196 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_12 := arithEq_of_rows h (row := 604) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_12
  simp only [← c38268, ← c38269, ← c38270] at e_604_12
  have e_604_13 := arithEq_of_rows h (row := 604) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_13
  simp only [← c38271, ← c38272, k2, ← c38273] at e_604_13
  have hr := e_604_13
  simp only [e_604_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3252)) := by
  have c38274 := (publicBatchWrapper89_copies1196 a h).2.2.1
  have c38275 := (publicBatchWrapper89_copies1196 a h).2.2.2.1
  have c38276 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.1
  have c38277 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.1
  have c38278 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.1
  have c38279 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_14 := arithEq_of_rows h (row := 604) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_14
  simp only [← c38274, ← c38275, ← c38276] at e_604_14
  have e_604_15 := arithEq_of_rows h (row := 604) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_15
  simp only [← c38277, ← c38278, k2, ← c38279] at e_604_15
  have hr := e_604_15
  simp only [e_604_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3253)) := by
  have c38280 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.1
  have c38281 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.1
  have c38282 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38283 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38284 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38285 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_16 := arithEq_of_rows h (row := 604) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_16
  simp only [← c38280, ← c38281, ← c38282] at e_604_16
  have e_604_17 := arithEq_of_rows h (row := 604) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_17
  simp only [← c38283, ← c38284, k2, ← c38285] at e_604_17
  have hr := e_604_17
  simp only [e_604_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 604 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3254)) := by
  have c38286 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38287 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38288 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38289 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38290 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38291 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_604_18 := arithEq_of_rows h (row := 604) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_18
  simp only [← c38286, ← c38287, ← c38288] at e_604_18
  have e_604_19 := arithEq_of_rows h (row := 604) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_19
  simp only [← c38289, ← c38290, k2, ← c38291] at e_604_19
  have hr := e_604_19
  simp only [e_604_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3255)) := by
  have c38292 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38293 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38294 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38295 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38296 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38297 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_0 := arithEq_of_rows h (row := 605) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_0
  simp only [← c38292, ← c38293, ← c38294] at e_605_0
  have e_605_1 := arithEq_of_rows h (row := 605) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_1
  simp only [← c38295, ← c38296, k2, ← c38297] at e_605_1
  have hr := e_605_1
  simp only [e_605_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3256)) := by
  have c38298 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38299 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38300 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38301 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38302 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38303 := (publicBatchWrapper89_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_2 := arithEq_of_rows h (row := 605) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_2
  simp only [← c38298, ← c38299, ← c38300] at e_605_2
  have e_605_3 := arithEq_of_rows h (row := 605) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_3
  simp only [← c38301, ← c38302, k2, ← c38303] at e_605_3
  have hr := e_605_3
  simp only [e_605_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3257)) := by
  have c38304 := (publicBatchWrapper89_copies1197 a h).1
  have c38305 := (publicBatchWrapper89_copies1197 a h).2.1
  have c38306 := (publicBatchWrapper89_copies1197 a h).2.2.1
  have c38307 := (publicBatchWrapper89_copies1197 a h).2.2.2.1
  have c38308 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.1
  have c38309 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_4 := arithEq_of_rows h (row := 605) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_4
  simp only [← c38304, ← c38305, ← c38306] at e_605_4
  have e_605_5 := arithEq_of_rows h (row := 605) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_5
  simp only [← c38307, ← c38308, k2, ← c38309] at e_605_5
  have hr := e_605_5
  simp only [e_605_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3258)) := by
  have c38310 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.1
  have c38311 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.1
  have c38312 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.1
  have c38313 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.1
  have c38314 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38315 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_6 := arithEq_of_rows h (row := 605) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_6
  simp only [← c38310, ← c38311, ← c38312] at e_605_6
  have e_605_7 := arithEq_of_rows h (row := 605) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_7
  simp only [← c38313, ← c38314, k2, ← c38315] at e_605_7
  have hr := e_605_7
  simp only [e_605_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3259)) := by
  have c38316 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38317 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38318 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38319 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38320 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38321 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_8 := arithEq_of_rows h (row := 605) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_8
  simp only [← c38316, ← c38317, ← c38318] at e_605_8
  have e_605_9 := arithEq_of_rows h (row := 605) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_9
  simp only [← c38319, ← c38320, k2, ← c38321] at e_605_9
  have hr := e_605_9
  simp only [e_605_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3260)) := by
  have c38322 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38323 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38324 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38325 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38326 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38327 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_10 := arithEq_of_rows h (row := 605) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_10
  simp only [← c38322, ← c38323, ← c38324] at e_605_10
  have e_605_11 := arithEq_of_rows h (row := 605) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_11
  simp only [← c38325, ← c38326, k2, ← c38327] at e_605_11
  have hr := e_605_11
  simp only [e_605_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3261)) := by
  have c38328 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38329 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38330 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38331 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38332 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38333 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_12 := arithEq_of_rows h (row := 605) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_12
  simp only [← c38328, ← c38329, ← c38330] at e_605_12
  have e_605_13 := arithEq_of_rows h (row := 605) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_13
  simp only [← c38331, ← c38332, k2, ← c38333] at e_605_13
  have hr := e_605_13
  simp only [e_605_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3262)) := by
  have c38334 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38335 := (publicBatchWrapper89_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38336 := (publicBatchWrapper89_copies1198 a h).1
  have c38337 := (publicBatchWrapper89_copies1198 a h).2.1
  have c38338 := (publicBatchWrapper89_copies1198 a h).2.2.1
  have c38339 := (publicBatchWrapper89_copies1198 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_14 := arithEq_of_rows h (row := 605) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_14
  simp only [← c38334, ← c38335, ← c38336] at e_605_14
  have e_605_15 := arithEq_of_rows h (row := 605) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_15
  simp only [← c38337, ← c38338, k2, ← c38339] at e_605_15
  have hr := e_605_15
  simp only [e_605_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3263)) := by
  have c38340 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.1
  have c38341 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.1
  have c38342 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.1
  have c38343 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.1
  have c38344 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.1
  have c38345 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_16 := arithEq_of_rows h (row := 605) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_16
  simp only [← c38340, ← c38341, ← c38342] at e_605_16
  have e_605_17 := arithEq_of_rows h (row := 605) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_17
  simp only [← c38343, ← c38344, k2, ← c38345] at e_605_17
  have hr := e_605_17
  simp only [e_605_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 605 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3264)) := by
  have c38346 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38347 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38348 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38349 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38350 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38351 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_605_18 := arithEq_of_rows h (row := 605) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_18
  simp only [← c38346, ← c38347, ← c38348] at e_605_18
  have e_605_19 := arithEq_of_rows h (row := 605) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_19
  simp only [← c38349, ← c38350, k2, ← c38351] at e_605_19
  have hr := e_605_19
  simp only [e_605_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3265)) := by
  have c38352 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38353 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38354 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38355 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38356 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38357 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_0 := arithEq_of_rows h (row := 606) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_0
  simp only [← c38352, ← c38353, ← c38354] at e_606_0
  have e_606_1 := arithEq_of_rows h (row := 606) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_1
  simp only [← c38355, ← c38356, k2, ← c38357] at e_606_1
  have hr := e_606_1
  simp only [e_606_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3266)) := by
  have c38358 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38359 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38360 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38361 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38362 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38363 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_2 := arithEq_of_rows h (row := 606) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_2
  simp only [← c38358, ← c38359, ← c38360] at e_606_2
  have e_606_3 := arithEq_of_rows h (row := 606) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_3
  simp only [← c38361, ← c38362, k2, ← c38363] at e_606_3
  have hr := e_606_3
  simp only [e_606_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3267)) := by
  have c38364 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38365 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38366 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38367 := (publicBatchWrapper89_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38368 := (publicBatchWrapper89_copies1199 a h).1
  have c38369 := (publicBatchWrapper89_copies1199 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_4 := arithEq_of_rows h (row := 606) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_4
  simp only [← c38364, ← c38365, ← c38366] at e_606_4
  have e_606_5 := arithEq_of_rows h (row := 606) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_5
  simp only [← c38367, ← c38368, k2, ← c38369] at e_606_5
  have hr := e_606_5
  simp only [e_606_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3268)) := by
  have c38370 := (publicBatchWrapper89_copies1199 a h).2.2.1
  have c38371 := (publicBatchWrapper89_copies1199 a h).2.2.2.1
  have c38372 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.1
  have c38373 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.1
  have c38374 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.1
  have c38375 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_6 := arithEq_of_rows h (row := 606) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_6
  simp only [← c38370, ← c38371, ← c38372] at e_606_6
  have e_606_7 := arithEq_of_rows h (row := 606) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_7
  simp only [← c38373, ← c38374, k2, ← c38375] at e_606_7
  have hr := e_606_7
  simp only [e_606_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3269)) := by
  have c38376 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.1
  have c38377 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.1
  have c38378 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38379 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38380 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38381 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_8 := arithEq_of_rows h (row := 606) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_8
  simp only [← c38376, ← c38377, ← c38378] at e_606_8
  have e_606_9 := arithEq_of_rows h (row := 606) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_9
  simp only [← c38379, ← c38380, k2, ← c38381] at e_606_9
  have hr := e_606_9
  simp only [e_606_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3270)) := by
  have c38382 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38383 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38384 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38385 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38386 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38387 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_10 := arithEq_of_rows h (row := 606) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_10
  simp only [← c38382, ← c38383, ← c38384] at e_606_10
  have e_606_11 := arithEq_of_rows h (row := 606) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_11
  simp only [← c38385, ← c38386, k2, ← c38387] at e_606_11
  have hr := e_606_11
  simp only [e_606_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3271)) := by
  have c38388 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38389 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38390 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38391 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38392 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38393 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_12 := arithEq_of_rows h (row := 606) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_12
  simp only [← c38388, ← c38389, ← c38390] at e_606_12
  have e_606_13 := arithEq_of_rows h (row := 606) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_13
  simp only [← c38391, ← c38392, k2, ← c38393] at e_606_13
  have hr := e_606_13
  simp only [e_606_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3272)) := by
  have c38394 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38395 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38396 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38397 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38398 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38399 := (publicBatchWrapper89_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_14 := arithEq_of_rows h (row := 606) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_14
  simp only [← c38394, ← c38395, ← c38396] at e_606_14
  have e_606_15 := arithEq_of_rows h (row := 606) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_15
  simp only [← c38397, ← c38398, k2, ← c38399] at e_606_15
  have hr := e_606_15
  simp only [e_606_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3273)) := by
  have c38400 := (publicBatchWrapper89_copies1200 a h).1
  have c38401 := (publicBatchWrapper89_copies1200 a h).2.1
  have c38402 := (publicBatchWrapper89_copies1200 a h).2.2.1
  have c38403 := (publicBatchWrapper89_copies1200 a h).2.2.2.1
  have c38404 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.1
  have c38405 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_16 := arithEq_of_rows h (row := 606) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_16
  simp only [← c38400, ← c38401, ← c38402] at e_606_16
  have e_606_17 := arithEq_of_rows h (row := 606) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_17
  simp only [← c38403, ← c38404, k2, ← c38405] at e_606_17
  have hr := e_606_17
  simp only [e_606_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 606 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3274)) := by
  have c38406 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.1
  have c38407 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.1
  have c38408 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.1
  have c38409 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.1
  have c38410 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38411 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_606_18 := arithEq_of_rows h (row := 606) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_18
  simp only [← c38406, ← c38407, ← c38408] at e_606_18
  have e_606_19 := arithEq_of_rows h (row := 606) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_19
  simp only [← c38409, ← c38410, k2, ← c38411] at e_606_19
  have hr := e_606_19
  simp only [e_606_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3275)) := by
  have c38412 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38413 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38414 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38415 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38416 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38417 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_0 := arithEq_of_rows h (row := 607) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_0
  simp only [← c38412, ← c38413, ← c38414] at e_607_0
  have e_607_1 := arithEq_of_rows h (row := 607) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_1
  simp only [← c38415, ← c38416, k2, ← c38417] at e_607_1
  have hr := e_607_1
  simp only [e_607_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3276)) := by
  have c38418 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38419 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38420 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38421 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38422 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38423 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_2 := arithEq_of_rows h (row := 607) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_2
  simp only [← c38418, ← c38419, ← c38420] at e_607_2
  have e_607_3 := arithEq_of_rows h (row := 607) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_3
  simp only [← c38421, ← c38422, k2, ← c38423] at e_607_3
  have hr := e_607_3
  simp only [e_607_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3277)) := by
  have c38424 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38425 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38426 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38427 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38428 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38429 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_4 := arithEq_of_rows h (row := 607) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_4
  simp only [← c38424, ← c38425, ← c38426] at e_607_4
  have e_607_5 := arithEq_of_rows h (row := 607) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_5
  simp only [← c38427, ← c38428, k2, ← c38429] at e_607_5
  have hr := e_607_5
  simp only [e_607_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3278)) := by
  have c38430 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38431 := (publicBatchWrapper89_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38432 := (publicBatchWrapper89_copies1201 a h).1
  have c38433 := (publicBatchWrapper89_copies1201 a h).2.1
  have c38434 := (publicBatchWrapper89_copies1201 a h).2.2.1
  have c38435 := (publicBatchWrapper89_copies1201 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_6 := arithEq_of_rows h (row := 607) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_6
  simp only [← c38430, ← c38431, ← c38432] at e_607_6
  have e_607_7 := arithEq_of_rows h (row := 607) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_7
  simp only [← c38433, ← c38434, k2, ← c38435] at e_607_7
  have hr := e_607_7
  simp only [e_607_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3279)) := by
  have c38436 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.1
  have c38437 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.1
  have c38438 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.1
  have c38439 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.1
  have c38440 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.1
  have c38441 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_8 := arithEq_of_rows h (row := 607) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_8
  simp only [← c38436, ← c38437, ← c38438] at e_607_8
  have e_607_9 := arithEq_of_rows h (row := 607) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_9
  simp only [← c38439, ← c38440, k2, ← c38441] at e_607_9
  have hr := e_607_9
  simp only [e_607_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3280)) := by
  have c38442 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38443 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38444 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38445 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38446 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38447 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_10 := arithEq_of_rows h (row := 607) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_10
  simp only [← c38442, ← c38443, ← c38444] at e_607_10
  have e_607_11 := arithEq_of_rows h (row := 607) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_11
  simp only [← c38445, ← c38446, k2, ← c38447] at e_607_11
  have hr := e_607_11
  simp only [e_607_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3281)) := by
  have c38448 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38449 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38450 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38451 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38452 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38453 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_12 := arithEq_of_rows h (row := 607) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_12
  simp only [← c38448, ← c38449, ← c38450] at e_607_12
  have e_607_13 := arithEq_of_rows h (row := 607) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_13
  simp only [← c38451, ← c38452, k2, ← c38453] at e_607_13
  have hr := e_607_13
  simp only [e_607_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3282)) := by
  have c38454 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38455 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38456 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38457 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38458 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38459 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_14 := arithEq_of_rows h (row := 607) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_14
  simp only [← c38454, ← c38455, ← c38456] at e_607_14
  have e_607_15 := arithEq_of_rows h (row := 607) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_15
  simp only [← c38457, ← c38458, k2, ← c38459] at e_607_15
  have hr := e_607_15
  simp only [e_607_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3283)) := by
  have c38460 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38461 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38462 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38463 := (publicBatchWrapper89_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38464 := (publicBatchWrapper89_copies1202 a h).1
  have c38465 := (publicBatchWrapper89_copies1202 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_16 := arithEq_of_rows h (row := 607) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_16
  simp only [← c38460, ← c38461, ← c38462] at e_607_16
  have e_607_17 := arithEq_of_rows h (row := 607) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_17
  simp only [← c38463, ← c38464, k2, ← c38465] at e_607_17
  have hr := e_607_17
  simp only [e_607_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 607 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3284)) := by
  have c38466 := (publicBatchWrapper89_copies1202 a h).2.2.1
  have c38467 := (publicBatchWrapper89_copies1202 a h).2.2.2.1
  have c38468 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.1
  have c38469 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.1
  have c38470 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.1
  have c38471 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_607_18 := arithEq_of_rows h (row := 607) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_18
  simp only [← c38466, ← c38467, ← c38468] at e_607_18
  have e_607_19 := arithEq_of_rows h (row := 607) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_19
  simp only [← c38469, ← c38470, k2, ← c38471] at e_607_19
  have hr := e_607_19
  simp only [e_607_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3285)) := by
  have c38472 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.1
  have c38473 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.1
  have c38474 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38475 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38476 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38477 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_0 := arithEq_of_rows h (row := 608) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_0
  simp only [← c38472, ← c38473, ← c38474] at e_608_0
  have e_608_1 := arithEq_of_rows h (row := 608) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_1
  simp only [← c38475, ← c38476, k2, ← c38477] at e_608_1
  have hr := e_608_1
  simp only [e_608_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3286)) := by
  have c38478 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38479 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38480 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38481 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38482 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38483 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_2 := arithEq_of_rows h (row := 608) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_2
  simp only [← c38478, ← c38479, ← c38480] at e_608_2
  have e_608_3 := arithEq_of_rows h (row := 608) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_3
  simp only [← c38481, ← c38482, k2, ← c38483] at e_608_3
  have hr := e_608_3
  simp only [e_608_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3287)) := by
  have c38484 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38485 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38486 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38487 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38488 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38489 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_4 := arithEq_of_rows h (row := 608) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_4
  simp only [← c38484, ← c38485, ← c38486] at e_608_4
  have e_608_5 := arithEq_of_rows h (row := 608) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_5
  simp only [← c38487, ← c38488, k2, ← c38489] at e_608_5
  have hr := e_608_5
  simp only [e_608_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3288)) := by
  have c38490 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38491 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38492 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38493 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38494 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38495 := (publicBatchWrapper89_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_6 := arithEq_of_rows h (row := 608) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_6
  simp only [← c38490, ← c38491, ← c38492] at e_608_6
  have e_608_7 := arithEq_of_rows h (row := 608) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_7
  simp only [← c38493, ← c38494, k2, ← c38495] at e_608_7
  have hr := e_608_7
  simp only [e_608_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3289)) := by
  have c38496 := (publicBatchWrapper89_copies1203 a h).1
  have c38497 := (publicBatchWrapper89_copies1203 a h).2.1
  have c38498 := (publicBatchWrapper89_copies1203 a h).2.2.1
  have c38499 := (publicBatchWrapper89_copies1203 a h).2.2.2.1
  have c38500 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.1
  have c38501 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_8 := arithEq_of_rows h (row := 608) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_8
  simp only [← c38496, ← c38497, ← c38498] at e_608_8
  have e_608_9 := arithEq_of_rows h (row := 608) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_9
  simp only [← c38499, ← c38500, k2, ← c38501] at e_608_9
  have hr := e_608_9
  simp only [e_608_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3290)) := by
  have c38502 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.1
  have c38503 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.1
  have c38504 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.1
  have c38505 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.1
  have c38506 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38507 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_10 := arithEq_of_rows h (row := 608) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_10
  simp only [← c38502, ← c38503, ← c38504] at e_608_10
  have e_608_11 := arithEq_of_rows h (row := 608) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_11
  simp only [← c38505, ← c38506, k2, ← c38507] at e_608_11
  have hr := e_608_11
  simp only [e_608_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3291)) := by
  have c38508 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38509 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38510 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38511 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38512 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38513 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_12 := arithEq_of_rows h (row := 608) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_12
  simp only [← c38508, ← c38509, ← c38510] at e_608_12
  have e_608_13 := arithEq_of_rows h (row := 608) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_13
  simp only [← c38511, ← c38512, k2, ← c38513] at e_608_13
  have hr := e_608_13
  simp only [e_608_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3292)) := by
  have c38514 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38515 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38516 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38517 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38518 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38519 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_14 := arithEq_of_rows h (row := 608) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_14
  simp only [← c38514, ← c38515, ← c38516] at e_608_14
  have e_608_15 := arithEq_of_rows h (row := 608) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_15
  simp only [← c38517, ← c38518, k2, ← c38519] at e_608_15
  have hr := e_608_15
  simp only [e_608_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3293)) := by
  have c38520 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38521 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38522 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38523 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38524 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38525 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_16 := arithEq_of_rows h (row := 608) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_16
  simp only [← c38520, ← c38521, ← c38522] at e_608_16
  have e_608_17 := arithEq_of_rows h (row := 608) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_17
  simp only [← c38523, ← c38524, k2, ← c38525] at e_608_17
  have hr := e_608_17
  simp only [e_608_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 608 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3294)) := by
  have c38526 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38527 := (publicBatchWrapper89_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38528 := (publicBatchWrapper89_copies1204 a h).1
  have c38529 := (publicBatchWrapper89_copies1204 a h).2.1
  have c38530 := (publicBatchWrapper89_copies1204 a h).2.2.1
  have c38531 := (publicBatchWrapper89_copies1204 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_608_18 := arithEq_of_rows h (row := 608) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_18
  simp only [← c38526, ← c38527, ← c38528] at e_608_18
  have e_608_19 := arithEq_of_rows h (row := 608) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_19
  simp only [← c38529, ← c38530, k2, ← c38531] at e_608_19
  have hr := e_608_19
  simp only [e_608_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3295)) := by
  have c38532 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.1
  have c38533 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.1
  have c38534 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.1
  have c38535 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.1
  have c38536 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.1
  have c38537 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_0 := arithEq_of_rows h (row := 609) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_0
  simp only [← c38532, ← c38533, ← c38534] at e_609_0
  have e_609_1 := arithEq_of_rows h (row := 609) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_1
  simp only [← c38535, ← c38536, k2, ← c38537] at e_609_1
  have hr := e_609_1
  simp only [e_609_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3296)) := by
  have c38538 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38539 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38540 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38541 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38542 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38543 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_2 := arithEq_of_rows h (row := 609) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_2
  simp only [← c38538, ← c38539, ← c38540] at e_609_2
  have e_609_3 := arithEq_of_rows h (row := 609) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_3
  simp only [← c38541, ← c38542, k2, ← c38543] at e_609_3
  have hr := e_609_3
  simp only [e_609_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3297)) := by
  have c38544 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38545 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38546 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38547 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38548 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38549 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_4 := arithEq_of_rows h (row := 609) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_4
  simp only [← c38544, ← c38545, ← c38546] at e_609_4
  have e_609_5 := arithEq_of_rows h (row := 609) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_5
  simp only [← c38547, ← c38548, k2, ← c38549] at e_609_5
  have hr := e_609_5
  simp only [e_609_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3298)) := by
  have c38550 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38551 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38552 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38553 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38554 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38555 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_6 := arithEq_of_rows h (row := 609) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_6
  simp only [← c38550, ← c38551, ← c38552] at e_609_6
  have e_609_7 := arithEq_of_rows h (row := 609) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_7
  simp only [← c38553, ← c38554, k2, ← c38555] at e_609_7
  have hr := e_609_7
  simp only [e_609_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3299)) := by
  have c38556 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38557 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38558 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38559 := (publicBatchWrapper89_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38560 := (publicBatchWrapper89_copies1205 a h).1
  have c38561 := (publicBatchWrapper89_copies1205 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_8 := arithEq_of_rows h (row := 609) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_8
  simp only [← c38556, ← c38557, ← c38558] at e_609_8
  have e_609_9 := arithEq_of_rows h (row := 609) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_9
  simp only [← c38559, ← c38560, k2, ← c38561] at e_609_9
  have hr := e_609_9
  simp only [e_609_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3300)) := by
  have c38562 := (publicBatchWrapper89_copies1205 a h).2.2.1
  have c38563 := (publicBatchWrapper89_copies1205 a h).2.2.2.1
  have c38564 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.1
  have c38565 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.1
  have c38566 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.1
  have c38567 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_10 := arithEq_of_rows h (row := 609) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_10
  simp only [← c38562, ← c38563, ← c38564] at e_609_10
  have e_609_11 := arithEq_of_rows h (row := 609) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_11
  simp only [← c38565, ← c38566, k2, ← c38567] at e_609_11
  have hr := e_609_11
  simp only [e_609_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3301)) := by
  have c38568 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.1
  have c38569 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.1
  have c38570 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38571 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38572 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38573 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_12 := arithEq_of_rows h (row := 609) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_12
  simp only [← c38568, ← c38569, ← c38570] at e_609_12
  have e_609_13 := arithEq_of_rows h (row := 609) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_13
  simp only [← c38571, ← c38572, k2, ← c38573] at e_609_13
  have hr := e_609_13
  simp only [e_609_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3302)) := by
  have c38574 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38575 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38576 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38577 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38578 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38579 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_14 := arithEq_of_rows h (row := 609) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_14
  simp only [← c38574, ← c38575, ← c38576] at e_609_14
  have e_609_15 := arithEq_of_rows h (row := 609) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_15
  simp only [← c38577, ← c38578, k2, ← c38579] at e_609_15
  have hr := e_609_15
  simp only [e_609_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3303)) := by
  have c38580 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38581 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38582 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38583 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38584 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38585 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_16 := arithEq_of_rows h (row := 609) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_16
  simp only [← c38580, ← c38581, ← c38582] at e_609_16
  have e_609_17 := arithEq_of_rows h (row := 609) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_17
  simp only [← c38583, ← c38584, k2, ← c38585] at e_609_17
  have hr := e_609_17
  simp only [e_609_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 609 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3304)) := by
  have c38586 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38587 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38588 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38589 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38590 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38591 := (publicBatchWrapper89_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_609_18 := arithEq_of_rows h (row := 609) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_18
  simp only [← c38586, ← c38587, ← c38588] at e_609_18
  have e_609_19 := arithEq_of_rows h (row := 609) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_19
  simp only [← c38589, ← c38590, k2, ← c38591] at e_609_19
  have hr := e_609_19
  simp only [e_609_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3305)) := by
  have c38592 := (publicBatchWrapper89_copies1206 a h).1
  have c38593 := (publicBatchWrapper89_copies1206 a h).2.1
  have c38594 := (publicBatchWrapper89_copies1206 a h).2.2.1
  have c38595 := (publicBatchWrapper89_copies1206 a h).2.2.2.1
  have c38596 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.1
  have c38597 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_0 := arithEq_of_rows h (row := 610) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_0
  simp only [← c38592, ← c38593, ← c38594] at e_610_0
  have e_610_1 := arithEq_of_rows h (row := 610) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_1
  simp only [← c38595, ← c38596, k2, ← c38597] at e_610_1
  have hr := e_610_1
  simp only [e_610_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3306)) := by
  have c38598 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.1
  have c38599 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.1
  have c38600 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.1
  have c38601 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.1
  have c38602 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38603 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_2 := arithEq_of_rows h (row := 610) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_2
  simp only [← c38598, ← c38599, ← c38600] at e_610_2
  have e_610_3 := arithEq_of_rows h (row := 610) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_3
  simp only [← c38601, ← c38602, k2, ← c38603] at e_610_3
  have hr := e_610_3
  simp only [e_610_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3307)) := by
  have c38604 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38605 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38606 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38607 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38608 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38609 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_4 := arithEq_of_rows h (row := 610) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_4
  simp only [← c38604, ← c38605, ← c38606] at e_610_4
  have e_610_5 := arithEq_of_rows h (row := 610) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_5
  simp only [← c38607, ← c38608, k2, ← c38609] at e_610_5
  have hr := e_610_5
  simp only [e_610_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3308)) := by
  have c38610 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38611 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38612 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38613 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38614 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38615 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_6 := arithEq_of_rows h (row := 610) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_6
  simp only [← c38610, ← c38611, ← c38612] at e_610_6
  have e_610_7 := arithEq_of_rows h (row := 610) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_7
  simp only [← c38613, ← c38614, k2, ← c38615] at e_610_7
  have hr := e_610_7
  simp only [e_610_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3309)) := by
  have c38616 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38617 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38618 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38619 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38620 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38621 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_8 := arithEq_of_rows h (row := 610) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_8
  simp only [← c38616, ← c38617, ← c38618] at e_610_8
  have e_610_9 := arithEq_of_rows h (row := 610) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_9
  simp only [← c38619, ← c38620, k2, ← c38621] at e_610_9
  have hr := e_610_9
  simp only [e_610_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3310)) := by
  have c38622 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38623 := (publicBatchWrapper89_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38624 := (publicBatchWrapper89_copies1207 a h).1
  have c38625 := (publicBatchWrapper89_copies1207 a h).2.1
  have c38626 := (publicBatchWrapper89_copies1207 a h).2.2.1
  have c38627 := (publicBatchWrapper89_copies1207 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_10 := arithEq_of_rows h (row := 610) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_10
  simp only [← c38622, ← c38623, ← c38624] at e_610_10
  have e_610_11 := arithEq_of_rows h (row := 610) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_11
  simp only [← c38625, ← c38626, k2, ← c38627] at e_610_11
  have hr := e_610_11
  simp only [e_610_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3311)) := by
  have c38628 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.1
  have c38629 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.1
  have c38630 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.1
  have c38631 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.1
  have c38632 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.1
  have c38633 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_12 := arithEq_of_rows h (row := 610) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_12
  simp only [← c38628, ← c38629, ← c38630] at e_610_12
  have e_610_13 := arithEq_of_rows h (row := 610) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_13
  simp only [← c38631, ← c38632, k2, ← c38633] at e_610_13
  have hr := e_610_13
  simp only [e_610_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3312)) := by
  have c38634 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38635 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38636 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38637 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38638 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38639 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_14 := arithEq_of_rows h (row := 610) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_14
  simp only [← c38634, ← c38635, ← c38636] at e_610_14
  have e_610_15 := arithEq_of_rows h (row := 610) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_15
  simp only [← c38637, ← c38638, k2, ← c38639] at e_610_15
  have hr := e_610_15
  simp only [e_610_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3313)) := by
  have c38640 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38641 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38642 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38643 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38644 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38645 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_16 := arithEq_of_rows h (row := 610) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_16
  simp only [← c38640, ← c38641, ← c38642] at e_610_16
  have e_610_17 := arithEq_of_rows h (row := 610) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_17
  simp only [← c38643, ← c38644, k2, ← c38645] at e_610_17
  have hr := e_610_17
  simp only [e_610_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 610 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3314)) := by
  have c38646 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38647 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38648 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38649 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38650 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38651 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_610_18 := arithEq_of_rows h (row := 610) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_18
  simp only [← c38646, ← c38647, ← c38648] at e_610_18
  have e_610_19 := arithEq_of_rows h (row := 610) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_19
  simp only [← c38649, ← c38650, k2, ← c38651] at e_610_19
  have hr := e_610_19
  simp only [e_610_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3315)) := by
  have c38652 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38653 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38654 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38655 := (publicBatchWrapper89_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38656 := (publicBatchWrapper89_copies1208 a h).1
  have c38657 := (publicBatchWrapper89_copies1208 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_0 := arithEq_of_rows h (row := 611) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_0
  simp only [← c38652, ← c38653, ← c38654] at e_611_0
  have e_611_1 := arithEq_of_rows h (row := 611) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_1
  simp only [← c38655, ← c38656, k2, ← c38657] at e_611_1
  have hr := e_611_1
  simp only [e_611_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3316)) := by
  have c38658 := (publicBatchWrapper89_copies1208 a h).2.2.1
  have c38659 := (publicBatchWrapper89_copies1208 a h).2.2.2.1
  have c38660 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.1
  have c38661 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.1
  have c38662 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.1
  have c38663 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_2 := arithEq_of_rows h (row := 611) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_2
  simp only [← c38658, ← c38659, ← c38660] at e_611_2
  have e_611_3 := arithEq_of_rows h (row := 611) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_3
  simp only [← c38661, ← c38662, k2, ← c38663] at e_611_3
  have hr := e_611_3
  simp only [e_611_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3317)) := by
  have c38664 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.1
  have c38665 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.1
  have c38666 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38667 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38668 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38669 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_4 := arithEq_of_rows h (row := 611) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_4
  simp only [← c38664, ← c38665, ← c38666] at e_611_4
  have e_611_5 := arithEq_of_rows h (row := 611) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_5
  simp only [← c38667, ← c38668, k2, ← c38669] at e_611_5
  have hr := e_611_5
  simp only [e_611_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3318)) := by
  have c38670 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38671 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38672 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38673 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38674 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38675 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_6 := arithEq_of_rows h (row := 611) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_6
  simp only [← c38670, ← c38671, ← c38672] at e_611_6
  have e_611_7 := arithEq_of_rows h (row := 611) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_7
  simp only [← c38673, ← c38674, k2, ← c38675] at e_611_7
  have hr := e_611_7
  simp only [e_611_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3319)) := by
  have c38676 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38677 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38678 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38679 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38680 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38681 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_8 := arithEq_of_rows h (row := 611) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_8
  simp only [← c38676, ← c38677, ← c38678] at e_611_8
  have e_611_9 := arithEq_of_rows h (row := 611) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_9
  simp only [← c38679, ← c38680, k2, ← c38681] at e_611_9
  have hr := e_611_9
  simp only [e_611_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3320)) := by
  have c38682 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38683 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38684 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38685 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38686 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38687 := (publicBatchWrapper89_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_10 := arithEq_of_rows h (row := 611) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_10
  simp only [← c38682, ← c38683, ← c38684] at e_611_10
  have e_611_11 := arithEq_of_rows h (row := 611) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_11
  simp only [← c38685, ← c38686, k2, ← c38687] at e_611_11
  have hr := e_611_11
  simp only [e_611_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3321)) := by
  have c38688 := (publicBatchWrapper89_copies1209 a h).1
  have c38689 := (publicBatchWrapper89_copies1209 a h).2.1
  have c38690 := (publicBatchWrapper89_copies1209 a h).2.2.1
  have c38691 := (publicBatchWrapper89_copies1209 a h).2.2.2.1
  have c38692 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.1
  have c38693 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_12 := arithEq_of_rows h (row := 611) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_12
  simp only [← c38688, ← c38689, ← c38690] at e_611_12
  have e_611_13 := arithEq_of_rows h (row := 611) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_13
  simp only [← c38691, ← c38692, k2, ← c38693] at e_611_13
  have hr := e_611_13
  simp only [e_611_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3322)) := by
  have c38694 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.1
  have c38695 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.1
  have c38696 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.1
  have c38697 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.1
  have c38698 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38699 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_14 := arithEq_of_rows h (row := 611) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_14
  simp only [← c38694, ← c38695, ← c38696] at e_611_14
  have e_611_15 := arithEq_of_rows h (row := 611) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_15
  simp only [← c38697, ← c38698, k2, ← c38699] at e_611_15
  have hr := e_611_15
  simp only [e_611_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3323)) := by
  have c38700 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38701 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38702 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38703 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38704 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38705 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_16 := arithEq_of_rows h (row := 611) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_16
  simp only [← c38700, ← c38701, ← c38702] at e_611_16
  have e_611_17 := arithEq_of_rows h (row := 611) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_17
  simp only [← c38703, ← c38704, k2, ← c38705] at e_611_17
  have hr := e_611_17
  simp only [e_611_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 611 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3324)) := by
  have c38706 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38707 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38708 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38709 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38710 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38711 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_611_18 := arithEq_of_rows h (row := 611) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_18
  simp only [← c38706, ← c38707, ← c38708] at e_611_18
  have e_611_19 := arithEq_of_rows h (row := 611) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_19
  simp only [← c38709, ← c38710, k2, ← c38711] at e_611_19
  have hr := e_611_19
  simp only [e_611_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3325)) := by
  have c38712 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38713 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38714 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38715 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38716 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38717 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_0 := arithEq_of_rows h (row := 612) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_0
  simp only [← c38712, ← c38713, ← c38714] at e_612_0
  have e_612_1 := arithEq_of_rows h (row := 612) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_1
  simp only [← c38715, ← c38716, k2, ← c38717] at e_612_1
  have hr := e_612_1
  simp only [e_612_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3326)) := by
  have c38718 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38719 := (publicBatchWrapper89_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38720 := (publicBatchWrapper89_copies1210 a h).1
  have c38721 := (publicBatchWrapper89_copies1210 a h).2.1
  have c38722 := (publicBatchWrapper89_copies1210 a h).2.2.1
  have c38723 := (publicBatchWrapper89_copies1210 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_2 := arithEq_of_rows h (row := 612) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_2
  simp only [← c38718, ← c38719, ← c38720] at e_612_2
  have e_612_3 := arithEq_of_rows h (row := 612) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_3
  simp only [← c38721, ← c38722, k2, ← c38723] at e_612_3
  have hr := e_612_3
  simp only [e_612_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3327)) := by
  have c38724 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.1
  have c38725 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.1
  have c38726 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.1
  have c38727 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.1
  have c38728 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.1
  have c38729 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_4 := arithEq_of_rows h (row := 612) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_4
  simp only [← c38724, ← c38725, ← c38726] at e_612_4
  have e_612_5 := arithEq_of_rows h (row := 612) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_5
  simp only [← c38727, ← c38728, k2, ← c38729] at e_612_5
  have hr := e_612_5
  simp only [e_612_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3328)) := by
  have c38730 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38731 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38732 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38733 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38734 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38735 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_6 := arithEq_of_rows h (row := 612) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_6
  simp only [← c38730, ← c38731, ← c38732] at e_612_6
  have e_612_7 := arithEq_of_rows h (row := 612) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_7
  simp only [← c38733, ← c38734, k2, ← c38735] at e_612_7
  have hr := e_612_7
  simp only [e_612_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3329)) := by
  have c38736 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38737 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38738 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38739 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38740 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38741 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_8 := arithEq_of_rows h (row := 612) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_8
  simp only [← c38736, ← c38737, ← c38738] at e_612_8
  have e_612_9 := arithEq_of_rows h (row := 612) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_9
  simp only [← c38739, ← c38740, k2, ← c38741] at e_612_9
  have hr := e_612_9
  simp only [e_612_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3330)) := by
  have c38742 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38743 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38744 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38745 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38746 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38747 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_10 := arithEq_of_rows h (row := 612) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_10
  simp only [← c38742, ← c38743, ← c38744] at e_612_10
  have e_612_11 := arithEq_of_rows h (row := 612) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_11
  simp only [← c38745, ← c38746, k2, ← c38747] at e_612_11
  have hr := e_612_11
  simp only [e_612_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3331)) := by
  have c38748 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38749 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38750 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38751 := (publicBatchWrapper89_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38752 := (publicBatchWrapper89_copies1211 a h).1
  have c38753 := (publicBatchWrapper89_copies1211 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_12 := arithEq_of_rows h (row := 612) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_12
  simp only [← c38748, ← c38749, ← c38750] at e_612_12
  have e_612_13 := arithEq_of_rows h (row := 612) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_13
  simp only [← c38751, ← c38752, k2, ← c38753] at e_612_13
  have hr := e_612_13
  simp only [e_612_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3332)) := by
  have c38754 := (publicBatchWrapper89_copies1211 a h).2.2.1
  have c38755 := (publicBatchWrapper89_copies1211 a h).2.2.2.1
  have c38756 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.1
  have c38757 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.1
  have c38758 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.1
  have c38759 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_14 := arithEq_of_rows h (row := 612) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_14
  simp only [← c38754, ← c38755, ← c38756] at e_612_14
  have e_612_15 := arithEq_of_rows h (row := 612) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_15
  simp only [← c38757, ← c38758, k2, ← c38759] at e_612_15
  have hr := e_612_15
  simp only [e_612_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3333)) := by
  have c38760 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.1
  have c38761 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.1
  have c38762 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38763 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38764 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38765 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_16 := arithEq_of_rows h (row := 612) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_16
  simp only [← c38760, ← c38761, ← c38762] at e_612_16
  have e_612_17 := arithEq_of_rows h (row := 612) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_17
  simp only [← c38763, ← c38764, k2, ← c38765] at e_612_17
  have hr := e_612_17
  simp only [e_612_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 612 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3334)) := by
  have c38766 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38767 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38768 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38769 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38770 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38771 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_612_18 := arithEq_of_rows h (row := 612) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_18
  simp only [← c38766, ← c38767, ← c38768] at e_612_18
  have e_612_19 := arithEq_of_rows h (row := 612) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_19
  simp only [← c38769, ← c38770, k2, ← c38771] at e_612_19
  have hr := e_612_19
  simp only [e_612_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3335)) := by
  have c38772 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38773 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38774 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38775 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38776 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38777 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_0 := arithEq_of_rows h (row := 613) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_0
  simp only [← c38772, ← c38773, ← c38774] at e_613_0
  have e_613_1 := arithEq_of_rows h (row := 613) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_1
  simp only [← c38775, ← c38776, k2, ← c38777] at e_613_1
  have hr := e_613_1
  simp only [e_613_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3336)) := by
  have c38778 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38779 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38780 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38781 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38782 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38783 := (publicBatchWrapper89_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_2 := arithEq_of_rows h (row := 613) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_2
  simp only [← c38778, ← c38779, ← c38780] at e_613_2
  have e_613_3 := arithEq_of_rows h (row := 613) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_3
  simp only [← c38781, ← c38782, k2, ← c38783] at e_613_3
  have hr := e_613_3
  simp only [e_613_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3337)) := by
  have c38784 := (publicBatchWrapper89_copies1212 a h).1
  have c38785 := (publicBatchWrapper89_copies1212 a h).2.1
  have c38786 := (publicBatchWrapper89_copies1212 a h).2.2.1
  have c38787 := (publicBatchWrapper89_copies1212 a h).2.2.2.1
  have c38788 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.1
  have c38789 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_4 := arithEq_of_rows h (row := 613) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_4
  simp only [← c38784, ← c38785, ← c38786] at e_613_4
  have e_613_5 := arithEq_of_rows h (row := 613) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_5
  simp only [← c38787, ← c38788, k2, ← c38789] at e_613_5
  have hr := e_613_5
  simp only [e_613_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3338)) := by
  have c38790 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.1
  have c38791 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.1
  have c38792 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.1
  have c38793 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.1
  have c38794 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38795 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_6 := arithEq_of_rows h (row := 613) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_6
  simp only [← c38790, ← c38791, ← c38792] at e_613_6
  have e_613_7 := arithEq_of_rows h (row := 613) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_7
  simp only [← c38793, ← c38794, k2, ← c38795] at e_613_7
  have hr := e_613_7
  simp only [e_613_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3339)) := by
  have c38796 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38797 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38798 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38799 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38800 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38801 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_8 := arithEq_of_rows h (row := 613) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_8
  simp only [← c38796, ← c38797, ← c38798] at e_613_8
  have e_613_9 := arithEq_of_rows h (row := 613) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_9
  simp only [← c38799, ← c38800, k2, ← c38801] at e_613_9
  have hr := e_613_9
  simp only [e_613_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3340)) := by
  have c38802 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38803 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38804 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38805 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38806 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38807 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_10 := arithEq_of_rows h (row := 613) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_10
  simp only [← c38802, ← c38803, ← c38804] at e_613_10
  have e_613_11 := arithEq_of_rows h (row := 613) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_11
  simp only [← c38805, ← c38806, k2, ← c38807] at e_613_11
  have hr := e_613_11
  simp only [e_613_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3341)) := by
  have c38808 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38809 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38810 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38811 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38812 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38813 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_12 := arithEq_of_rows h (row := 613) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_12
  simp only [← c38808, ← c38809, ← c38810] at e_613_12
  have e_613_13 := arithEq_of_rows h (row := 613) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_13
  simp only [← c38811, ← c38812, k2, ← c38813] at e_613_13
  have hr := e_613_13
  simp only [e_613_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3342)) := by
  have c38814 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38815 := (publicBatchWrapper89_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38816 := (publicBatchWrapper89_copies1213 a h).1
  have c38817 := (publicBatchWrapper89_copies1213 a h).2.1
  have c38818 := (publicBatchWrapper89_copies1213 a h).2.2.1
  have c38819 := (publicBatchWrapper89_copies1213 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_14 := arithEq_of_rows h (row := 613) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_14
  simp only [← c38814, ← c38815, ← c38816] at e_613_14
  have e_613_15 := arithEq_of_rows h (row := 613) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_15
  simp only [← c38817, ← c38818, k2, ← c38819] at e_613_15
  have hr := e_613_15
  simp only [e_613_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3343)) := by
  have c38820 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.1
  have c38821 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.1
  have c38822 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.1
  have c38823 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.1
  have c38824 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.1
  have c38825 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_16 := arithEq_of_rows h (row := 613) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_16
  simp only [← c38820, ← c38821, ← c38822] at e_613_16
  have e_613_17 := arithEq_of_rows h (row := 613) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_17
  simp only [← c38823, ← c38824, k2, ← c38825] at e_613_17
  have hr := e_613_17
  simp only [e_613_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 613 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3344)) := by
  have c38826 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38827 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38828 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38829 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38830 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38831 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_613_18 := arithEq_of_rows h (row := 613) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_18
  simp only [← c38826, ← c38827, ← c38828] at e_613_18
  have e_613_19 := arithEq_of_rows h (row := 613) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_19
  simp only [← c38829, ← c38830, k2, ← c38831] at e_613_19
  have hr := e_613_19
  simp only [e_613_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3345)) := by
  have c38832 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38833 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38834 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38835 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38836 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38837 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_0 := arithEq_of_rows h (row := 614) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_0
  simp only [← c38832, ← c38833, ← c38834] at e_614_0
  have e_614_1 := arithEq_of_rows h (row := 614) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_1
  simp only [← c38835, ← c38836, k2, ← c38837] at e_614_1
  have hr := e_614_1
  simp only [e_614_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3346)) := by
  have c38838 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38839 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38840 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38841 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38842 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38843 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_2 := arithEq_of_rows h (row := 614) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_2
  simp only [← c38838, ← c38839, ← c38840] at e_614_2
  have e_614_3 := arithEq_of_rows h (row := 614) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_3
  simp only [← c38841, ← c38842, k2, ← c38843] at e_614_3
  have hr := e_614_3
  simp only [e_614_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3347)) := by
  have c38844 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38845 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38846 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38847 := (publicBatchWrapper89_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38848 := (publicBatchWrapper89_copies1214 a h).1
  have c38849 := (publicBatchWrapper89_copies1214 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_4 := arithEq_of_rows h (row := 614) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_4
  simp only [← c38844, ← c38845, ← c38846] at e_614_4
  have e_614_5 := arithEq_of_rows h (row := 614) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_5
  simp only [← c38847, ← c38848, k2, ← c38849] at e_614_5
  have hr := e_614_5
  simp only [e_614_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3348)) := by
  have c38850 := (publicBatchWrapper89_copies1214 a h).2.2.1
  have c38851 := (publicBatchWrapper89_copies1214 a h).2.2.2.1
  have c38852 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.1
  have c38853 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.1
  have c38854 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.1
  have c38855 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_6 := arithEq_of_rows h (row := 614) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_6
  simp only [← c38850, ← c38851, ← c38852] at e_614_6
  have e_614_7 := arithEq_of_rows h (row := 614) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_7
  simp only [← c38853, ← c38854, k2, ← c38855] at e_614_7
  have hr := e_614_7
  simp only [e_614_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3349)) := by
  have c38856 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.1
  have c38857 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.1
  have c38858 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38859 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38860 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38861 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_8 := arithEq_of_rows h (row := 614) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_8
  simp only [← c38856, ← c38857, ← c38858] at e_614_8
  have e_614_9 := arithEq_of_rows h (row := 614) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_9
  simp only [← c38859, ← c38860, k2, ← c38861] at e_614_9
  have hr := e_614_9
  simp only [e_614_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3350)) := by
  have c38862 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38863 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38864 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38865 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38866 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38867 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_10 := arithEq_of_rows h (row := 614) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_10
  simp only [← c38862, ← c38863, ← c38864] at e_614_10
  have e_614_11 := arithEq_of_rows h (row := 614) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_11
  simp only [← c38865, ← c38866, k2, ← c38867] at e_614_11
  have hr := e_614_11
  simp only [e_614_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3351)) := by
  have c38868 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38869 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38870 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38871 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38872 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38873 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_12 := arithEq_of_rows h (row := 614) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_12
  simp only [← c38868, ← c38869, ← c38870] at e_614_12
  have e_614_13 := arithEq_of_rows h (row := 614) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_13
  simp only [← c38871, ← c38872, k2, ← c38873] at e_614_13
  have hr := e_614_13
  simp only [e_614_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3352)) := by
  have c38874 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38875 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38876 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38877 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38878 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38879 := (publicBatchWrapper89_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_14 := arithEq_of_rows h (row := 614) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_14
  simp only [← c38874, ← c38875, ← c38876] at e_614_14
  have e_614_15 := arithEq_of_rows h (row := 614) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_15
  simp only [← c38877, ← c38878, k2, ← c38879] at e_614_15
  have hr := e_614_15
  simp only [e_614_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3353)) := by
  have c38880 := (publicBatchWrapper89_copies1215 a h).1
  have c38881 := (publicBatchWrapper89_copies1215 a h).2.1
  have c38882 := (publicBatchWrapper89_copies1215 a h).2.2.1
  have c38883 := (publicBatchWrapper89_copies1215 a h).2.2.2.1
  have c38884 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.1
  have c38885 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_16 := arithEq_of_rows h (row := 614) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_16
  simp only [← c38880, ← c38881, ← c38882] at e_614_16
  have e_614_17 := arithEq_of_rows h (row := 614) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_17
  simp only [← c38883, ← c38884, k2, ← c38885] at e_614_17
  have hr := e_614_17
  simp only [e_614_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 614 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3354)) := by
  have c38886 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.1
  have c38887 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.1
  have c38888 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.1
  have c38889 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.1
  have c38890 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38891 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_614_18 := arithEq_of_rows h (row := 614) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_18
  simp only [← c38886, ← c38887, ← c38888] at e_614_18
  have e_614_19 := arithEq_of_rows h (row := 614) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_19
  simp only [← c38889, ← c38890, k2, ← c38891] at e_614_19
  have hr := e_614_19
  simp only [e_614_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3355)) := by
  have c38892 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38893 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38894 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38895 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38896 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38897 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_0 := arithEq_of_rows h (row := 615) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_0
  simp only [← c38892, ← c38893, ← c38894] at e_615_0
  have e_615_1 := arithEq_of_rows h (row := 615) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_1
  simp only [← c38895, ← c38896, k2, ← c38897] at e_615_1
  have hr := e_615_1
  simp only [e_615_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3356)) := by
  have c38898 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38899 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38900 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38901 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38902 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38903 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_2 := arithEq_of_rows h (row := 615) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_2
  simp only [← c38898, ← c38899, ← c38900] at e_615_2
  have e_615_3 := arithEq_of_rows h (row := 615) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_3
  simp only [← c38901, ← c38902, k2, ← c38903] at e_615_3
  have hr := e_615_3
  simp only [e_615_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3357)) := by
  have c38904 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38905 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38906 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38907 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38908 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38909 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_4 := arithEq_of_rows h (row := 615) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_4
  simp only [← c38904, ← c38905, ← c38906] at e_615_4
  have e_615_5 := arithEq_of_rows h (row := 615) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_5
  simp only [← c38907, ← c38908, k2, ← c38909] at e_615_5
  have hr := e_615_5
  simp only [e_615_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3358)) := by
  have c38910 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38911 := (publicBatchWrapper89_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38912 := (publicBatchWrapper89_copies1216 a h).1
  have c38913 := (publicBatchWrapper89_copies1216 a h).2.1
  have c38914 := (publicBatchWrapper89_copies1216 a h).2.2.1
  have c38915 := (publicBatchWrapper89_copies1216 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_6 := arithEq_of_rows h (row := 615) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_6
  simp only [← c38910, ← c38911, ← c38912] at e_615_6
  have e_615_7 := arithEq_of_rows h (row := 615) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_7
  simp only [← c38913, ← c38914, k2, ← c38915] at e_615_7
  have hr := e_615_7
  simp only [e_615_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3359)) := by
  have c38916 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.1
  have c38917 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.1
  have c38918 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.1
  have c38919 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.1
  have c38920 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.1
  have c38921 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_8 := arithEq_of_rows h (row := 615) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_8
  simp only [← c38916, ← c38917, ← c38918] at e_615_8
  have e_615_9 := arithEq_of_rows h (row := 615) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_9
  simp only [← c38919, ← c38920, k2, ← c38921] at e_615_9
  have hr := e_615_9
  simp only [e_615_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3360)) := by
  have c38922 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38923 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38924 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38925 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38926 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38927 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_10 := arithEq_of_rows h (row := 615) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_10
  simp only [← c38922, ← c38923, ← c38924] at e_615_10
  have e_615_11 := arithEq_of_rows h (row := 615) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_11
  simp only [← c38925, ← c38926, k2, ← c38927] at e_615_11
  have hr := e_615_11
  simp only [e_615_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3361)) := by
  have c38928 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38929 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38930 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38931 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38932 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38933 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_12 := arithEq_of_rows h (row := 615) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_12
  simp only [← c38928, ← c38929, ← c38930] at e_615_12
  have e_615_13 := arithEq_of_rows h (row := 615) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_13
  simp only [← c38931, ← c38932, k2, ← c38933] at e_615_13
  have hr := e_615_13
  simp only [e_615_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3362)) := by
  have c38934 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38935 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38936 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38937 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38938 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38939 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_14 := arithEq_of_rows h (row := 615) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_14
  simp only [← c38934, ← c38935, ← c38936] at e_615_14
  have e_615_15 := arithEq_of_rows h (row := 615) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_15
  simp only [← c38937, ← c38938, k2, ← c38939] at e_615_15
  have hr := e_615_15
  simp only [e_615_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3363)) := by
  have c38940 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38941 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38942 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38943 := (publicBatchWrapper89_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38944 := (publicBatchWrapper89_copies1217 a h).1
  have c38945 := (publicBatchWrapper89_copies1217 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_16 := arithEq_of_rows h (row := 615) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_16
  simp only [← c38940, ← c38941, ← c38942] at e_615_16
  have e_615_17 := arithEq_of_rows h (row := 615) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_17
  simp only [← c38943, ← c38944, k2, ← c38945] at e_615_17
  have hr := e_615_17
  simp only [e_615_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 615 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3364)) := by
  have c38946 := (publicBatchWrapper89_copies1217 a h).2.2.1
  have c38947 := (publicBatchWrapper89_copies1217 a h).2.2.2.1
  have c38948 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.1
  have c38949 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.1
  have c38950 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.1
  have c38951 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_615_18 := arithEq_of_rows h (row := 615) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_18
  simp only [← c38946, ← c38947, ← c38948] at e_615_18
  have e_615_19 := arithEq_of_rows h (row := 615) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_19
  simp only [← c38949, ← c38950, k2, ← c38951] at e_615_19
  have hr := e_615_19
  simp only [e_615_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3365)) := by
  have c38952 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.1
  have c38953 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.1
  have c38954 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38955 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38956 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38957 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_0 := arithEq_of_rows h (row := 616) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_0
  simp only [← c38952, ← c38953, ← c38954] at e_616_0
  have e_616_1 := arithEq_of_rows h (row := 616) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_1
  simp only [← c38955, ← c38956, k2, ← c38957] at e_616_1
  have hr := e_616_1
  simp only [e_616_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3366)) := by
  have c38958 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38959 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38960 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38961 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38962 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38963 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_2 := arithEq_of_rows h (row := 616) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_2
  simp only [← c38958, ← c38959, ← c38960] at e_616_2
  have e_616_3 := arithEq_of_rows h (row := 616) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_3
  simp only [← c38961, ← c38962, k2, ← c38963] at e_616_3
  have hr := e_616_3
  simp only [e_616_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3367)) := by
  have c38964 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38965 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38966 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38967 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38968 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38969 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_4 := arithEq_of_rows h (row := 616) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_4
  simp only [← c38964, ← c38965, ← c38966] at e_616_4
  have e_616_5 := arithEq_of_rows h (row := 616) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_5
  simp only [← c38967, ← c38968, k2, ← c38969] at e_616_5
  have hr := e_616_5
  simp only [e_616_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3368)) := by
  have c38970 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38971 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38972 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38973 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38974 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38975 := (publicBatchWrapper89_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_6 := arithEq_of_rows h (row := 616) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_6
  simp only [← c38970, ← c38971, ← c38972] at e_616_6
  have e_616_7 := arithEq_of_rows h (row := 616) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_7
  simp only [← c38973, ← c38974, k2, ← c38975] at e_616_7
  have hr := e_616_7
  simp only [e_616_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3369)) := by
  have c38976 := (publicBatchWrapper89_copies1218 a h).1
  have c38977 := (publicBatchWrapper89_copies1218 a h).2.1
  have c38978 := (publicBatchWrapper89_copies1218 a h).2.2.1
  have c38979 := (publicBatchWrapper89_copies1218 a h).2.2.2.1
  have c38980 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.1
  have c38981 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_8 := arithEq_of_rows h (row := 616) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_8
  simp only [← c38976, ← c38977, ← c38978] at e_616_8
  have e_616_9 := arithEq_of_rows h (row := 616) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_9
  simp only [← c38979, ← c38980, k2, ← c38981] at e_616_9
  have hr := e_616_9
  simp only [e_616_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3410)) := by
  have c38982 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.1
  have c38983 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.1
  have c38984 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.1
  have c38985 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.1
  have c38986 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38987 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_10 := arithEq_of_rows h (row := 616) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_10
  simp only [← c38982, ← c38983, ← c38984] at e_616_10
  have e_616_11 := arithEq_of_rows h (row := 616) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_11
  simp only [← c38985, ← c38986, k2, ← c38987] at e_616_11
  have hr := e_616_11
  simp only [e_616_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3411)) := by
  have c38988 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38989 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38990 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38991 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38992 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38993 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_12 := arithEq_of_rows h (row := 616) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_12
  simp only [← c38988, ← c38989, ← c38990] at e_616_12
  have e_616_13 := arithEq_of_rows h (row := 616) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_13
  simp only [← c38991, ← c38992, k2, ← c38993] at e_616_13
  have hr := e_616_13
  simp only [e_616_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3412)) := by
  have c38994 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38995 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38996 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38997 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38998 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38999 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_14 := arithEq_of_rows h (row := 616) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_14
  simp only [← c38994, ← c38995, ← c38996] at e_616_14
  have e_616_15 := arithEq_of_rows h (row := 616) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_15
  simp only [← c38997, ← c38998, k2, ← c38999] at e_616_15
  have hr := e_616_15
  simp only [e_616_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3413)) := by
  have c39000 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39001 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39002 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39003 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39004 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39005 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_16 := arithEq_of_rows h (row := 616) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_16
  simp only [← c39000, ← c39001, ← c39002] at e_616_16
  have e_616_17 := arithEq_of_rows h (row := 616) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_17
  simp only [← c39003, ← c39004, k2, ← c39005] at e_616_17
  have hr := e_616_17
  simp only [e_616_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 616 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3414)) := by
  have c39006 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39007 := (publicBatchWrapper89_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39008 := (publicBatchWrapper89_copies1219 a h).1
  have c39009 := (publicBatchWrapper89_copies1219 a h).2.1
  have c39010 := (publicBatchWrapper89_copies1219 a h).2.2.1
  have c39011 := (publicBatchWrapper89_copies1219 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_616_18 := arithEq_of_rows h (row := 616) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_18
  simp only [← c39006, ← c39007, ← c39008] at e_616_18
  have e_616_19 := arithEq_of_rows h (row := 616) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_19
  simp only [← c39009, ← c39010, k2, ← c39011] at e_616_19
  have hr := e_616_19
  simp only [e_616_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3415)) := by
  have c39012 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.1
  have c39013 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.1
  have c39014 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.1
  have c39015 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.1
  have c39016 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.1
  have c39017 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_0 := arithEq_of_rows h (row := 617) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_0
  simp only [← c39012, ← c39013, ← c39014] at e_617_0
  have e_617_1 := arithEq_of_rows h (row := 617) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_1
  simp only [← c39015, ← c39016, k2, ← c39017] at e_617_1
  have hr := e_617_1
  simp only [e_617_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3416)) := by
  have c39018 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39019 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39020 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39021 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39022 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39023 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_2 := arithEq_of_rows h (row := 617) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_2
  simp only [← c39018, ← c39019, ← c39020] at e_617_2
  have e_617_3 := arithEq_of_rows h (row := 617) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_3
  simp only [← c39021, ← c39022, k2, ← c39023] at e_617_3
  have hr := e_617_3
  simp only [e_617_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3417)) := by
  have c39024 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39025 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39026 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39027 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39028 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39029 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_4 := arithEq_of_rows h (row := 617) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_4
  simp only [← c39024, ← c39025, ← c39026] at e_617_4
  have e_617_5 := arithEq_of_rows h (row := 617) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_5
  simp only [← c39027, ← c39028, k2, ← c39029] at e_617_5
  have hr := e_617_5
  simp only [e_617_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3418)) := by
  have c39030 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39031 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39032 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39033 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39034 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39035 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_6 := arithEq_of_rows h (row := 617) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_6
  simp only [← c39030, ← c39031, ← c39032] at e_617_6
  have e_617_7 := arithEq_of_rows h (row := 617) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_7
  simp only [← c39033, ← c39034, k2, ← c39035] at e_617_7
  have hr := e_617_7
  simp only [e_617_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3419)) := by
  have c39036 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39037 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39038 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39039 := (publicBatchWrapper89_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39040 := (publicBatchWrapper89_copies1220 a h).1
  have c39041 := (publicBatchWrapper89_copies1220 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_8 := arithEq_of_rows h (row := 617) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_8
  simp only [← c39036, ← c39037, ← c39038] at e_617_8
  have e_617_9 := arithEq_of_rows h (row := 617) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_9
  simp only [← c39039, ← c39040, k2, ← c39041] at e_617_9
  have hr := e_617_9
  simp only [e_617_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3420)) := by
  have c39042 := (publicBatchWrapper89_copies1220 a h).2.2.1
  have c39043 := (publicBatchWrapper89_copies1220 a h).2.2.2.1
  have c39044 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.1
  have c39045 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.1
  have c39046 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.1
  have c39047 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_10 := arithEq_of_rows h (row := 617) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_10
  simp only [← c39042, ← c39043, ← c39044] at e_617_10
  have e_617_11 := arithEq_of_rows h (row := 617) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_11
  simp only [← c39045, ← c39046, k2, ← c39047] at e_617_11
  have hr := e_617_11
  simp only [e_617_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3421)) := by
  have c39048 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.1
  have c39049 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.1
  have c39050 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39051 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39052 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39053 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_12 := arithEq_of_rows h (row := 617) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_12
  simp only [← c39048, ← c39049, ← c39050] at e_617_12
  have e_617_13 := arithEq_of_rows h (row := 617) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_13
  simp only [← c39051, ← c39052, k2, ← c39053] at e_617_13
  have hr := e_617_13
  simp only [e_617_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3422)) := by
  have c39054 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39055 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39056 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39057 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39058 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39059 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_14 := arithEq_of_rows h (row := 617) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_14
  simp only [← c39054, ← c39055, ← c39056] at e_617_14
  have e_617_15 := arithEq_of_rows h (row := 617) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_15
  simp only [← c39057, ← c39058, k2, ← c39059] at e_617_15
  have hr := e_617_15
  simp only [e_617_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3423)) := by
  have c39060 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39061 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39062 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39063 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39064 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39065 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_16 := arithEq_of_rows h (row := 617) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_16
  simp only [← c39060, ← c39061, ← c39062] at e_617_16
  have e_617_17 := arithEq_of_rows h (row := 617) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_17
  simp only [← c39063, ← c39064, k2, ← c39065] at e_617_17
  have hr := e_617_17
  simp only [e_617_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 617 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3424)) := by
  have c39066 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39067 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39068 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39069 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39070 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39071 := (publicBatchWrapper89_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_617_18 := arithEq_of_rows h (row := 617) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_18
  simp only [← c39066, ← c39067, ← c39068] at e_617_18
  have e_617_19 := arithEq_of_rows h (row := 617) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_19
  simp only [← c39069, ← c39070, k2, ← c39071] at e_617_19
  have hr := e_617_19
  simp only [e_617_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3425)) := by
  have c39072 := (publicBatchWrapper89_copies1221 a h).1
  have c39073 := (publicBatchWrapper89_copies1221 a h).2.1
  have c39074 := (publicBatchWrapper89_copies1221 a h).2.2.1
  have c39075 := (publicBatchWrapper89_copies1221 a h).2.2.2.1
  have c39076 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.1
  have c39077 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_0 := arithEq_of_rows h (row := 618) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_0
  simp only [← c39072, ← c39073, ← c39074] at e_618_0
  have e_618_1 := arithEq_of_rows h (row := 618) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_1
  simp only [← c39075, ← c39076, k2, ← c39077] at e_618_1
  have hr := e_618_1
  simp only [e_618_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3426)) := by
  have c39078 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.1
  have c39079 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.1
  have c39080 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.1
  have c39081 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.1
  have c39082 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39083 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_2 := arithEq_of_rows h (row := 618) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_2
  simp only [← c39078, ← c39079, ← c39080] at e_618_2
  have e_618_3 := arithEq_of_rows h (row := 618) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_3
  simp only [← c39081, ← c39082, k2, ← c39083] at e_618_3
  have hr := e_618_3
  simp only [e_618_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3427)) := by
  have c39084 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39085 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39086 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39087 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39088 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39089 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_4 := arithEq_of_rows h (row := 618) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_4
  simp only [← c39084, ← c39085, ← c39086] at e_618_4
  have e_618_5 := arithEq_of_rows h (row := 618) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_5
  simp only [← c39087, ← c39088, k2, ← c39089] at e_618_5
  have hr := e_618_5
  simp only [e_618_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3428)) := by
  have c39090 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39091 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39092 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39093 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39094 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39095 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_6 := arithEq_of_rows h (row := 618) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_6
  simp only [← c39090, ← c39091, ← c39092] at e_618_6
  have e_618_7 := arithEq_of_rows h (row := 618) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_7
  simp only [← c39093, ← c39094, k2, ← c39095] at e_618_7
  have hr := e_618_7
  simp only [e_618_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3429)) := by
  have c39096 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39097 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39098 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39099 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39100 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39101 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_8 := arithEq_of_rows h (row := 618) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_8
  simp only [← c39096, ← c39097, ← c39098] at e_618_8
  have e_618_9 := arithEq_of_rows h (row := 618) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_9
  simp only [← c39099, ← c39100, k2, ← c39101] at e_618_9
  have hr := e_618_9
  simp only [e_618_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3430)) := by
  have c39102 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39103 := (publicBatchWrapper89_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39104 := (publicBatchWrapper89_copies1222 a h).1
  have c39105 := (publicBatchWrapper89_copies1222 a h).2.1
  have c39106 := (publicBatchWrapper89_copies1222 a h).2.2.1
  have c39107 := (publicBatchWrapper89_copies1222 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_10 := arithEq_of_rows h (row := 618) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_10
  simp only [← c39102, ← c39103, ← c39104] at e_618_10
  have e_618_11 := arithEq_of_rows h (row := 618) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_11
  simp only [← c39105, ← c39106, k2, ← c39107] at e_618_11
  have hr := e_618_11
  simp only [e_618_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3431)) := by
  have c39108 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.1
  have c39109 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.1
  have c39110 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.1
  have c39111 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.1
  have c39112 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.1
  have c39113 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_12 := arithEq_of_rows h (row := 618) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_12
  simp only [← c39108, ← c39109, ← c39110] at e_618_12
  have e_618_13 := arithEq_of_rows h (row := 618) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_13
  simp only [← c39111, ← c39112, k2, ← c39113] at e_618_13
  have hr := e_618_13
  simp only [e_618_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3432)) := by
  have c39114 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39115 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39116 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39117 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39118 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39119 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_14 := arithEq_of_rows h (row := 618) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_14
  simp only [← c39114, ← c39115, ← c39116] at e_618_14
  have e_618_15 := arithEq_of_rows h (row := 618) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_15
  simp only [← c39117, ← c39118, k2, ← c39119] at e_618_15
  have hr := e_618_15
  simp only [e_618_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3433)) := by
  have c39120 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39121 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39122 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39123 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39124 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39125 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_16 := arithEq_of_rows h (row := 618) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_16
  simp only [← c39120, ← c39121, ← c39122] at e_618_16
  have e_618_17 := arithEq_of_rows h (row := 618) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_17
  simp only [← c39123, ← c39124, k2, ← c39125] at e_618_17
  have hr := e_618_17
  simp only [e_618_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 618 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3434)) := by
  have c39126 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39127 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39128 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39129 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39130 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39131 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_618_18 := arithEq_of_rows h (row := 618) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_18
  simp only [← c39126, ← c39127, ← c39128] at e_618_18
  have e_618_19 := arithEq_of_rows h (row := 618) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_19
  simp only [← c39129, ← c39130, k2, ← c39131] at e_618_19
  have hr := e_618_19
  simp only [e_618_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3435)) := by
  have c39132 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39133 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39134 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39135 := (publicBatchWrapper89_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39136 := (publicBatchWrapper89_copies1223 a h).1
  have c39137 := (publicBatchWrapper89_copies1223 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_0 := arithEq_of_rows h (row := 619) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_0
  simp only [← c39132, ← c39133, ← c39134] at e_619_0
  have e_619_1 := arithEq_of_rows h (row := 619) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_1
  simp only [← c39135, ← c39136, k2, ← c39137] at e_619_1
  have hr := e_619_1
  simp only [e_619_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3436)) := by
  have c39138 := (publicBatchWrapper89_copies1223 a h).2.2.1
  have c39139 := (publicBatchWrapper89_copies1223 a h).2.2.2.1
  have c39140 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.1
  have c39141 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.1
  have c39142 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.1
  have c39143 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_2 := arithEq_of_rows h (row := 619) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_2
  simp only [← c39138, ← c39139, ← c39140] at e_619_2
  have e_619_3 := arithEq_of_rows h (row := 619) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_3
  simp only [← c39141, ← c39142, k2, ← c39143] at e_619_3
  have hr := e_619_3
  simp only [e_619_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3437)) := by
  have c39144 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.1
  have c39145 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.1
  have c39146 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39147 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39148 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39149 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_4 := arithEq_of_rows h (row := 619) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_4
  simp only [← c39144, ← c39145, ← c39146] at e_619_4
  have e_619_5 := arithEq_of_rows h (row := 619) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_5
  simp only [← c39147, ← c39148, k2, ← c39149] at e_619_5
  have hr := e_619_5
  simp only [e_619_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3438)) := by
  have c39150 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39151 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39152 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39153 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39154 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39155 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_6 := arithEq_of_rows h (row := 619) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_6
  simp only [← c39150, ← c39151, ← c39152] at e_619_6
  have e_619_7 := arithEq_of_rows h (row := 619) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_7
  simp only [← c39153, ← c39154, k2, ← c39155] at e_619_7
  have hr := e_619_7
  simp only [e_619_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3439)) := by
  have c39156 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39157 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39158 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39159 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39160 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39161 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_8 := arithEq_of_rows h (row := 619) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_8
  simp only [← c39156, ← c39157, ← c39158] at e_619_8
  have e_619_9 := arithEq_of_rows h (row := 619) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_9
  simp only [← c39159, ← c39160, k2, ← c39161] at e_619_9
  have hr := e_619_9
  simp only [e_619_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3440)) := by
  have c39162 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39163 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39164 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39165 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39166 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39167 := (publicBatchWrapper89_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_10 := arithEq_of_rows h (row := 619) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_10
  simp only [← c39162, ← c39163, ← c39164] at e_619_10
  have e_619_11 := arithEq_of_rows h (row := 619) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_11
  simp only [← c39165, ← c39166, k2, ← c39167] at e_619_11
  have hr := e_619_11
  simp only [e_619_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3441)) := by
  have c39168 := (publicBatchWrapper89_copies1224 a h).1
  have c39169 := (publicBatchWrapper89_copies1224 a h).2.1
  have c39170 := (publicBatchWrapper89_copies1224 a h).2.2.1
  have c39171 := (publicBatchWrapper89_copies1224 a h).2.2.2.1
  have c39172 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.1
  have c39173 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_12 := arithEq_of_rows h (row := 619) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_12
  simp only [← c39168, ← c39169, ← c39170] at e_619_12
  have e_619_13 := arithEq_of_rows h (row := 619) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_13
  simp only [← c39171, ← c39172, k2, ← c39173] at e_619_13
  have hr := e_619_13
  simp only [e_619_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3442)) := by
  have c39174 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.1
  have c39175 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.1
  have c39176 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.1
  have c39177 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.1
  have c39178 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39179 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_14 := arithEq_of_rows h (row := 619) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_14
  simp only [← c39174, ← c39175, ← c39176] at e_619_14
  have e_619_15 := arithEq_of_rows h (row := 619) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_15
  simp only [← c39177, ← c39178, k2, ← c39179] at e_619_15
  have hr := e_619_15
  simp only [e_619_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3443)) := by
  have c39180 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39181 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39182 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39183 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39184 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39185 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_16 := arithEq_of_rows h (row := 619) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_16
  simp only [← c39180, ← c39181, ← c39182] at e_619_16
  have e_619_17 := arithEq_of_rows h (row := 619) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_17
  simp only [← c39183, ← c39184, k2, ← c39185] at e_619_17
  have hr := e_619_17
  simp only [e_619_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 619 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3444)) := by
  have c39186 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39187 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39188 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39189 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39190 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39191 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_619_18 := arithEq_of_rows h (row := 619) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_18
  simp only [← c39186, ← c39187, ← c39188] at e_619_18
  have e_619_19 := arithEq_of_rows h (row := 619) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_19
  simp only [← c39189, ← c39190, k2, ← c39191] at e_619_19
  have hr := e_619_19
  simp only [e_619_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3445)) := by
  have c39192 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39193 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39194 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39195 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39196 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39197 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_0 := arithEq_of_rows h (row := 620) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_0
  simp only [← c39192, ← c39193, ← c39194] at e_620_0
  have e_620_1 := arithEq_of_rows h (row := 620) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_1
  simp only [← c39195, ← c39196, k2, ← c39197] at e_620_1
  have hr := e_620_1
  simp only [e_620_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3446)) := by
  have c39198 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39199 := (publicBatchWrapper89_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39200 := (publicBatchWrapper89_copies1225 a h).1
  have c39201 := (publicBatchWrapper89_copies1225 a h).2.1
  have c39202 := (publicBatchWrapper89_copies1225 a h).2.2.1
  have c39203 := (publicBatchWrapper89_copies1225 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_2 := arithEq_of_rows h (row := 620) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_2
  simp only [← c39198, ← c39199, ← c39200] at e_620_2
  have e_620_3 := arithEq_of_rows h (row := 620) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_3
  simp only [← c39201, ← c39202, k2, ← c39203] at e_620_3
  have hr := e_620_3
  simp only [e_620_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3447)) := by
  have c39204 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.1
  have c39205 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.1
  have c39206 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.1
  have c39207 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.1
  have c39208 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.1
  have c39209 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_4 := arithEq_of_rows h (row := 620) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_4
  simp only [← c39204, ← c39205, ← c39206] at e_620_4
  have e_620_5 := arithEq_of_rows h (row := 620) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_5
  simp only [← c39207, ← c39208, k2, ← c39209] at e_620_5
  have hr := e_620_5
  simp only [e_620_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3448)) := by
  have c39210 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39211 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39212 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39213 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39214 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39215 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_6 := arithEq_of_rows h (row := 620) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_6
  simp only [← c39210, ← c39211, ← c39212] at e_620_6
  have e_620_7 := arithEq_of_rows h (row := 620) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_7
  simp only [← c39213, ← c39214, k2, ← c39215] at e_620_7
  have hr := e_620_7
  simp only [e_620_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3449)) := by
  have c39216 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39217 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39218 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39219 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39220 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39221 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_8 := arithEq_of_rows h (row := 620) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_8
  simp only [← c39216, ← c39217, ← c39218] at e_620_8
  have e_620_9 := arithEq_of_rows h (row := 620) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_9
  simp only [← c39219, ← c39220, k2, ← c39221] at e_620_9
  have hr := e_620_9
  simp only [e_620_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3450)) := by
  have c39222 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39223 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39224 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39225 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39226 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39227 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_10 := arithEq_of_rows h (row := 620) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_10
  simp only [← c39222, ← c39223, ← c39224] at e_620_10
  have e_620_11 := arithEq_of_rows h (row := 620) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_11
  simp only [← c39225, ← c39226, k2, ← c39227] at e_620_11
  have hr := e_620_11
  simp only [e_620_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3451)) := by
  have c39228 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39229 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39230 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39231 := (publicBatchWrapper89_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39232 := (publicBatchWrapper89_copies1226 a h).1
  have c39233 := (publicBatchWrapper89_copies1226 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_12 := arithEq_of_rows h (row := 620) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_12
  simp only [← c39228, ← c39229, ← c39230] at e_620_12
  have e_620_13 := arithEq_of_rows h (row := 620) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_13
  simp only [← c39231, ← c39232, k2, ← c39233] at e_620_13
  have hr := e_620_13
  simp only [e_620_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3452)) := by
  have c39234 := (publicBatchWrapper89_copies1226 a h).2.2.1
  have c39235 := (publicBatchWrapper89_copies1226 a h).2.2.2.1
  have c39236 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.1
  have c39237 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.1
  have c39238 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.1
  have c39239 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_14 := arithEq_of_rows h (row := 620) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_14
  simp only [← c39234, ← c39235, ← c39236] at e_620_14
  have e_620_15 := arithEq_of_rows h (row := 620) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_15
  simp only [← c39237, ← c39238, k2, ← c39239] at e_620_15
  have hr := e_620_15
  simp only [e_620_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3453)) := by
  have c39240 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.1
  have c39241 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.1
  have c39242 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39243 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39244 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39245 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_16 := arithEq_of_rows h (row := 620) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_16
  simp only [← c39240, ← c39241, ← c39242] at e_620_16
  have e_620_17 := arithEq_of_rows h (row := 620) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_17
  simp only [← c39243, ← c39244, k2, ← c39245] at e_620_17
  have hr := e_620_17
  simp only [e_620_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 620 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3454)) := by
  have c39246 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39247 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39248 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39249 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39250 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39251 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_620_18 := arithEq_of_rows h (row := 620) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_18
  simp only [← c39246, ← c39247, ← c39248] at e_620_18
  have e_620_19 := arithEq_of_rows h (row := 620) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_19
  simp only [← c39249, ← c39250, k2, ← c39251] at e_620_19
  have hr := e_620_19
  simp only [e_620_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3455)) := by
  have c39252 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39253 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39254 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39255 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39256 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39257 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_0 := arithEq_of_rows h (row := 621) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_0
  simp only [← c39252, ← c39253, ← c39254] at e_621_0
  have e_621_1 := arithEq_of_rows h (row := 621) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_1
  simp only [← c39255, ← c39256, k2, ← c39257] at e_621_1
  have hr := e_621_1
  simp only [e_621_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3456)) := by
  have c39258 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39259 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39260 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39261 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39262 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39263 := (publicBatchWrapper89_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_2 := arithEq_of_rows h (row := 621) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_2
  simp only [← c39258, ← c39259, ← c39260] at e_621_2
  have e_621_3 := arithEq_of_rows h (row := 621) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_3
  simp only [← c39261, ← c39262, k2, ← c39263] at e_621_3
  have hr := e_621_3
  simp only [e_621_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3457)) := by
  have c39264 := (publicBatchWrapper89_copies1227 a h).1
  have c39265 := (publicBatchWrapper89_copies1227 a h).2.1
  have c39266 := (publicBatchWrapper89_copies1227 a h).2.2.1
  have c39267 := (publicBatchWrapper89_copies1227 a h).2.2.2.1
  have c39268 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.1
  have c39269 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_4 := arithEq_of_rows h (row := 621) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_4
  simp only [← c39264, ← c39265, ← c39266] at e_621_4
  have e_621_5 := arithEq_of_rows h (row := 621) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_5
  simp only [← c39267, ← c39268, k2, ← c39269] at e_621_5
  have hr := e_621_5
  simp only [e_621_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3458)) := by
  have c39270 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.1
  have c39271 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.1
  have c39272 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.1
  have c39273 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.1
  have c39274 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39275 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_6 := arithEq_of_rows h (row := 621) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_6
  simp only [← c39270, ← c39271, ← c39272] at e_621_6
  have e_621_7 := arithEq_of_rows h (row := 621) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_7
  simp only [← c39273, ← c39274, k2, ← c39275] at e_621_7
  have hr := e_621_7
  simp only [e_621_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3459)) := by
  have c39276 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39277 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39278 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39279 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39280 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39281 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_8 := arithEq_of_rows h (row := 621) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_8
  simp only [← c39276, ← c39277, ← c39278] at e_621_8
  have e_621_9 := arithEq_of_rows h (row := 621) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_9
  simp only [← c39279, ← c39280, k2, ← c39281] at e_621_9
  have hr := e_621_9
  simp only [e_621_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3460)) := by
  have c39282 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39283 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39284 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39285 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39286 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39287 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_10 := arithEq_of_rows h (row := 621) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_10
  simp only [← c39282, ← c39283, ← c39284] at e_621_10
  have e_621_11 := arithEq_of_rows h (row := 621) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_11
  simp only [← c39285, ← c39286, k2, ← c39287] at e_621_11
  have hr := e_621_11
  simp only [e_621_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3461)) := by
  have c39288 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39289 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39290 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39291 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39292 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39293 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_12 := arithEq_of_rows h (row := 621) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_12
  simp only [← c39288, ← c39289, ← c39290] at e_621_12
  have e_621_13 := arithEq_of_rows h (row := 621) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_13
  simp only [← c39291, ← c39292, k2, ← c39293] at e_621_13
  have hr := e_621_13
  simp only [e_621_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3462)) := by
  have c39294 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39295 := (publicBatchWrapper89_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39296 := (publicBatchWrapper89_copies1228 a h).1
  have c39297 := (publicBatchWrapper89_copies1228 a h).2.1
  have c39298 := (publicBatchWrapper89_copies1228 a h).2.2.1
  have c39299 := (publicBatchWrapper89_copies1228 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_14 := arithEq_of_rows h (row := 621) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_14
  simp only [← c39294, ← c39295, ← c39296] at e_621_14
  have e_621_15 := arithEq_of_rows h (row := 621) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_15
  simp only [← c39297, ← c39298, k2, ← c39299] at e_621_15
  have hr := e_621_15
  simp only [e_621_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3463)) := by
  have c39300 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.1
  have c39301 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.1
  have c39302 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.1
  have c39303 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.1
  have c39304 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.1
  have c39305 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_16 := arithEq_of_rows h (row := 621) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_16
  simp only [← c39300, ← c39301, ← c39302] at e_621_16
  have e_621_17 := arithEq_of_rows h (row := 621) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_17
  simp only [← c39303, ← c39304, k2, ← c39305] at e_621_17
  have hr := e_621_17
  simp only [e_621_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 621 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3464)) := by
  have c39306 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39307 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39308 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39309 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39310 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39311 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_621_18 := arithEq_of_rows h (row := 621) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_18
  simp only [← c39306, ← c39307, ← c39308] at e_621_18
  have e_621_19 := arithEq_of_rows h (row := 621) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_19
  simp only [← c39309, ← c39310, k2, ← c39311] at e_621_19
  have hr := e_621_19
  simp only [e_621_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3465)) := by
  have c39312 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39313 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39314 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39315 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39316 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39317 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_0 := arithEq_of_rows h (row := 622) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_0
  simp only [← c39312, ← c39313, ← c39314] at e_622_0
  have e_622_1 := arithEq_of_rows h (row := 622) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_1
  simp only [← c39315, ← c39316, k2, ← c39317] at e_622_1
  have hr := e_622_1
  simp only [e_622_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3466)) := by
  have c39318 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39319 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39320 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39321 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39322 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39323 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_2 := arithEq_of_rows h (row := 622) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_2
  simp only [← c39318, ← c39319, ← c39320] at e_622_2
  have e_622_3 := arithEq_of_rows h (row := 622) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_3
  simp only [← c39321, ← c39322, k2, ← c39323] at e_622_3
  have hr := e_622_3
  simp only [e_622_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3467)) := by
  have c39324 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39325 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39326 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39327 := (publicBatchWrapper89_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39328 := (publicBatchWrapper89_copies1229 a h).1
  have c39329 := (publicBatchWrapper89_copies1229 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_4 := arithEq_of_rows h (row := 622) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_4
  simp only [← c39324, ← c39325, ← c39326] at e_622_4
  have e_622_5 := arithEq_of_rows h (row := 622) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_5
  simp only [← c39327, ← c39328, k2, ← c39329] at e_622_5
  have hr := e_622_5
  simp only [e_622_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3468)) := by
  have c39330 := (publicBatchWrapper89_copies1229 a h).2.2.1
  have c39331 := (publicBatchWrapper89_copies1229 a h).2.2.2.1
  have c39332 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.1
  have c39333 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.1
  have c39334 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.1
  have c39335 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_6 := arithEq_of_rows h (row := 622) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_6
  simp only [← c39330, ← c39331, ← c39332] at e_622_6
  have e_622_7 := arithEq_of_rows h (row := 622) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_7
  simp only [← c39333, ← c39334, k2, ← c39335] at e_622_7
  have hr := e_622_7
  simp only [e_622_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3469)) := by
  have c39336 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.1
  have c39337 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.1
  have c39338 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39339 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39340 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39341 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_8 := arithEq_of_rows h (row := 622) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_8
  simp only [← c39336, ← c39337, ← c39338] at e_622_8
  have e_622_9 := arithEq_of_rows h (row := 622) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_9
  simp only [← c39339, ← c39340, k2, ← c39341] at e_622_9
  have hr := e_622_9
  simp only [e_622_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3470)) := by
  have c39342 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39343 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39344 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39345 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39346 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39347 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_10 := arithEq_of_rows h (row := 622) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_10
  simp only [← c39342, ← c39343, ← c39344] at e_622_10
  have e_622_11 := arithEq_of_rows h (row := 622) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_11
  simp only [← c39345, ← c39346, k2, ← c39347] at e_622_11
  have hr := e_622_11
  simp only [e_622_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3471)) := by
  have c39348 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39349 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39350 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39351 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39352 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39353 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_12 := arithEq_of_rows h (row := 622) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_12
  simp only [← c39348, ← c39349, ← c39350] at e_622_12
  have e_622_13 := arithEq_of_rows h (row := 622) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_13
  simp only [← c39351, ← c39352, k2, ← c39353] at e_622_13
  have hr := e_622_13
  simp only [e_622_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3472)) := by
  have c39354 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39355 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39356 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39357 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39358 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39359 := (publicBatchWrapper89_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_14 := arithEq_of_rows h (row := 622) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_14
  simp only [← c39354, ← c39355, ← c39356] at e_622_14
  have e_622_15 := arithEq_of_rows h (row := 622) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_15
  simp only [← c39357, ← c39358, k2, ← c39359] at e_622_15
  have hr := e_622_15
  simp only [e_622_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3473)) := by
  have c39360 := (publicBatchWrapper89_copies1230 a h).1
  have c39361 := (publicBatchWrapper89_copies1230 a h).2.1
  have c39362 := (publicBatchWrapper89_copies1230 a h).2.2.1
  have c39363 := (publicBatchWrapper89_copies1230 a h).2.2.2.1
  have c39364 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.1
  have c39365 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_16 := arithEq_of_rows h (row := 622) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_16
  simp only [← c39360, ← c39361, ← c39362] at e_622_16
  have e_622_17 := arithEq_of_rows h (row := 622) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_17
  simp only [← c39363, ← c39364, k2, ← c39365] at e_622_17
  have hr := e_622_17
  simp only [e_622_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 622 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3474)) := by
  have c39366 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.1
  have c39367 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.1
  have c39368 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.1
  have c39369 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.1
  have c39370 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39371 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_622_18 := arithEq_of_rows h (row := 622) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_18
  simp only [← c39366, ← c39367, ← c39368] at e_622_18
  have e_622_19 := arithEq_of_rows h (row := 622) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_19
  simp only [← c39369, ← c39370, k2, ← c39371] at e_622_19
  have hr := e_622_19
  simp only [e_622_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3475)) := by
  have c39372 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39373 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39374 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39375 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39376 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39377 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_0 := arithEq_of_rows h (row := 623) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_0
  simp only [← c39372, ← c39373, ← c39374] at e_623_0
  have e_623_1 := arithEq_of_rows h (row := 623) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_1
  simp only [← c39375, ← c39376, k2, ← c39377] at e_623_1
  have hr := e_623_1
  simp only [e_623_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3476)) := by
  have c39378 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39379 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39380 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39381 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39382 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39383 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_2 := arithEq_of_rows h (row := 623) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_2
  simp only [← c39378, ← c39379, ← c39380] at e_623_2
  have e_623_3 := arithEq_of_rows h (row := 623) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_3
  simp only [← c39381, ← c39382, k2, ← c39383] at e_623_3
  have hr := e_623_3
  simp only [e_623_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3477)) := by
  have c39384 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39385 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39386 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39387 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39388 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39389 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_4 := arithEq_of_rows h (row := 623) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_4
  simp only [← c39384, ← c39385, ← c39386] at e_623_4
  have e_623_5 := arithEq_of_rows h (row := 623) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_5
  simp only [← c39387, ← c39388, k2, ← c39389] at e_623_5
  have hr := e_623_5
  simp only [e_623_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3478)) := by
  have c39390 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39391 := (publicBatchWrapper89_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39392 := (publicBatchWrapper89_copies1231 a h).1
  have c39393 := (publicBatchWrapper89_copies1231 a h).2.1
  have c39394 := (publicBatchWrapper89_copies1231 a h).2.2.1
  have c39395 := (publicBatchWrapper89_copies1231 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_6 := arithEq_of_rows h (row := 623) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_6
  simp only [← c39390, ← c39391, ← c39392] at e_623_6
  have e_623_7 := arithEq_of_rows h (row := 623) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_7
  simp only [← c39393, ← c39394, k2, ← c39395] at e_623_7
  have hr := e_623_7
  simp only [e_623_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3479)) := by
  have c39396 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.1
  have c39397 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.1
  have c39398 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.1
  have c39399 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.1
  have c39400 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.1
  have c39401 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_8 := arithEq_of_rows h (row := 623) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_8
  simp only [← c39396, ← c39397, ← c39398] at e_623_8
  have e_623_9 := arithEq_of_rows h (row := 623) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_9
  simp only [← c39399, ← c39400, k2, ← c39401] at e_623_9
  have hr := e_623_9
  simp only [e_623_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3480)) := by
  have c39402 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39403 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39404 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39405 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39406 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39407 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_10 := arithEq_of_rows h (row := 623) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_10
  simp only [← c39402, ← c39403, ← c39404] at e_623_10
  have e_623_11 := arithEq_of_rows h (row := 623) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_11
  simp only [← c39405, ← c39406, k2, ← c39407] at e_623_11
  have hr := e_623_11
  simp only [e_623_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3481)) := by
  have c39408 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39409 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39410 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39411 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39412 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39413 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_12 := arithEq_of_rows h (row := 623) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_12
  simp only [← c39408, ← c39409, ← c39410] at e_623_12
  have e_623_13 := arithEq_of_rows h (row := 623) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_13
  simp only [← c39411, ← c39412, k2, ← c39413] at e_623_13
  have hr := e_623_13
  simp only [e_623_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3482)) := by
  have c39414 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39415 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39416 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39417 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39418 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39419 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_14 := arithEq_of_rows h (row := 623) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_14
  simp only [← c39414, ← c39415, ← c39416] at e_623_14
  have e_623_15 := arithEq_of_rows h (row := 623) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_15
  simp only [← c39417, ← c39418, k2, ← c39419] at e_623_15
  have hr := e_623_15
  simp only [e_623_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3483)) := by
  have c39420 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39421 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39422 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39423 := (publicBatchWrapper89_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39424 := (publicBatchWrapper89_copies1232 a h).1
  have c39425 := (publicBatchWrapper89_copies1232 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_16 := arithEq_of_rows h (row := 623) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_16
  simp only [← c39420, ← c39421, ← c39422] at e_623_16
  have e_623_17 := arithEq_of_rows h (row := 623) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_17
  simp only [← c39423, ← c39424, k2, ← c39425] at e_623_17
  have hr := e_623_17
  simp only [e_623_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 623 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3484)) := by
  have c39426 := (publicBatchWrapper89_copies1232 a h).2.2.1
  have c39427 := (publicBatchWrapper89_copies1232 a h).2.2.2.1
  have c39428 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.1
  have c39429 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.1
  have c39430 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.1
  have c39431 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_623_18 := arithEq_of_rows h (row := 623) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_18
  simp only [← c39426, ← c39427, ← c39428] at e_623_18
  have e_623_19 := arithEq_of_rows h (row := 623) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_19
  simp only [← c39429, ← c39430, k2, ← c39431] at e_623_19
  have hr := e_623_19
  simp only [e_623_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3485)) := by
  have c39432 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.1
  have c39433 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.1
  have c39434 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39435 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39436 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39437 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_0 := arithEq_of_rows h (row := 624) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_0
  simp only [← c39432, ← c39433, ← c39434] at e_624_0
  have e_624_1 := arithEq_of_rows h (row := 624) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_1
  simp only [← c39435, ← c39436, k2, ← c39437] at e_624_1
  have hr := e_624_1
  simp only [e_624_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3486)) := by
  have c39438 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39439 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39440 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39441 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39442 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39443 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_2 := arithEq_of_rows h (row := 624) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_2
  simp only [← c39438, ← c39439, ← c39440] at e_624_2
  have e_624_3 := arithEq_of_rows h (row := 624) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_3
  simp only [← c39441, ← c39442, k2, ← c39443] at e_624_3
  have hr := e_624_3
  simp only [e_624_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3487)) := by
  have c39444 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39445 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39446 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39447 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39448 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39449 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_4 := arithEq_of_rows h (row := 624) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_4
  simp only [← c39444, ← c39445, ← c39446] at e_624_4
  have e_624_5 := arithEq_of_rows h (row := 624) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_5
  simp only [← c39447, ← c39448, k2, ← c39449] at e_624_5
  have hr := e_624_5
  simp only [e_624_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3488)) := by
  have c39450 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39451 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39452 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39453 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39454 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39455 := (publicBatchWrapper89_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_6 := arithEq_of_rows h (row := 624) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_6
  simp only [← c39450, ← c39451, ← c39452] at e_624_6
  have e_624_7 := arithEq_of_rows h (row := 624) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_7
  simp only [← c39453, ← c39454, k2, ← c39455] at e_624_7
  have hr := e_624_7
  simp only [e_624_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3489)) := by
  have c39456 := (publicBatchWrapper89_copies1233 a h).1
  have c39457 := (publicBatchWrapper89_copies1233 a h).2.1
  have c39458 := (publicBatchWrapper89_copies1233 a h).2.2.1
  have c39459 := (publicBatchWrapper89_copies1233 a h).2.2.2.1
  have c39460 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.1
  have c39461 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_8 := arithEq_of_rows h (row := 624) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_8
  simp only [← c39456, ← c39457, ← c39458] at e_624_8
  have e_624_9 := arithEq_of_rows h (row := 624) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_9
  simp only [← c39459, ← c39460, k2, ← c39461] at e_624_9
  have hr := e_624_9
  simp only [e_624_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3490)) := by
  have c39462 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.1
  have c39463 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.1
  have c39464 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.1
  have c39465 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.1
  have c39466 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39467 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_10 := arithEq_of_rows h (row := 624) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_10
  simp only [← c39462, ← c39463, ← c39464] at e_624_10
  have e_624_11 := arithEq_of_rows h (row := 624) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_11
  simp only [← c39465, ← c39466, k2, ← c39467] at e_624_11
  have hr := e_624_11
  simp only [e_624_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3491)) := by
  have c39468 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39469 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39470 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39471 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39472 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39473 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_12 := arithEq_of_rows h (row := 624) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_12
  simp only [← c39468, ← c39469, ← c39470] at e_624_12
  have e_624_13 := arithEq_of_rows h (row := 624) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_13
  simp only [← c39471, ← c39472, k2, ← c39473] at e_624_13
  have hr := e_624_13
  simp only [e_624_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3492)) := by
  have c39474 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39475 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39476 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39477 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39478 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39479 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_14 := arithEq_of_rows h (row := 624) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_14
  simp only [← c39474, ← c39475, ← c39476] at e_624_14
  have e_624_15 := arithEq_of_rows h (row := 624) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_15
  simp only [← c39477, ← c39478, k2, ← c39479] at e_624_15
  have hr := e_624_15
  simp only [e_624_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3493)) := by
  have c39480 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39481 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39482 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39483 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39484 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39485 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_16 := arithEq_of_rows h (row := 624) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_16
  simp only [← c39480, ← c39481, ← c39482] at e_624_16
  have e_624_17 := arithEq_of_rows h (row := 624) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_17
  simp only [← c39483, ← c39484, k2, ← c39485] at e_624_17
  have hr := e_624_17
  simp only [e_624_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 624 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3494)) := by
  have c39486 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39487 := (publicBatchWrapper89_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39488 := (publicBatchWrapper89_copies1234 a h).1
  have c39489 := (publicBatchWrapper89_copies1234 a h).2.1
  have c39490 := (publicBatchWrapper89_copies1234 a h).2.2.1
  have c39491 := (publicBatchWrapper89_copies1234 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_624_18 := arithEq_of_rows h (row := 624) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_18
  simp only [← c39486, ← c39487, ← c39488] at e_624_18
  have e_624_19 := arithEq_of_rows h (row := 624) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_19
  simp only [← c39489, ← c39490, k2, ← c39491] at e_624_19
  have hr := e_624_19
  simp only [e_624_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3495)) := by
  have c39492 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.1
  have c39493 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.1
  have c39494 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.1
  have c39495 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.1
  have c39496 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.1
  have c39497 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_0 := arithEq_of_rows h (row := 625) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_0
  simp only [← c39492, ← c39493, ← c39494] at e_625_0
  have e_625_1 := arithEq_of_rows h (row := 625) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_1
  simp only [← c39495, ← c39496, k2, ← c39497] at e_625_1
  have hr := e_625_1
  simp only [e_625_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3496)) := by
  have c39498 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39499 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39500 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39501 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39502 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39503 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_2 := arithEq_of_rows h (row := 625) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_2
  simp only [← c39498, ← c39499, ← c39500] at e_625_2
  have e_625_3 := arithEq_of_rows h (row := 625) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_3
  simp only [← c39501, ← c39502, k2, ← c39503] at e_625_3
  have hr := e_625_3
  simp only [e_625_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3497)) := by
  have c39504 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39505 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39506 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39507 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39508 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39509 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_4 := arithEq_of_rows h (row := 625) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_4
  simp only [← c39504, ← c39505, ← c39506] at e_625_4
  have e_625_5 := arithEq_of_rows h (row := 625) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_5
  simp only [← c39507, ← c39508, k2, ← c39509] at e_625_5
  have hr := e_625_5
  simp only [e_625_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3498)) := by
  have c39510 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39511 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39512 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39513 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39514 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39515 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_6 := arithEq_of_rows h (row := 625) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_6
  simp only [← c39510, ← c39511, ← c39512] at e_625_6
  have e_625_7 := arithEq_of_rows h (row := 625) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_7
  simp only [← c39513, ← c39514, k2, ← c39515] at e_625_7
  have hr := e_625_7
  simp only [e_625_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3499)) := by
  have c39516 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39517 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39518 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39519 := (publicBatchWrapper89_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39520 := (publicBatchWrapper89_copies1235 a h).1
  have c39521 := (publicBatchWrapper89_copies1235 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_8 := arithEq_of_rows h (row := 625) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_8
  simp only [← c39516, ← c39517, ← c39518] at e_625_8
  have e_625_9 := arithEq_of_rows h (row := 625) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_9
  simp only [← c39519, ← c39520, k2, ← c39521] at e_625_9
  have hr := e_625_9
  simp only [e_625_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3500)) := by
  have c39522 := (publicBatchWrapper89_copies1235 a h).2.2.1
  have c39523 := (publicBatchWrapper89_copies1235 a h).2.2.2.1
  have c39524 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.1
  have c39525 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.1
  have c39526 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.1
  have c39527 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_10 := arithEq_of_rows h (row := 625) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_10
  simp only [← c39522, ← c39523, ← c39524] at e_625_10
  have e_625_11 := arithEq_of_rows h (row := 625) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_11
  simp only [← c39525, ← c39526, k2, ← c39527] at e_625_11
  have hr := e_625_11
  simp only [e_625_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3501)) := by
  have c39528 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.1
  have c39529 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.1
  have c39530 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39531 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39532 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39533 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_12 := arithEq_of_rows h (row := 625) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_12
  simp only [← c39528, ← c39529, ← c39530] at e_625_12
  have e_625_13 := arithEq_of_rows h (row := 625) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_13
  simp only [← c39531, ← c39532, k2, ← c39533] at e_625_13
  have hr := e_625_13
  simp only [e_625_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3502)) := by
  have c39534 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39535 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39536 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39537 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39538 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39539 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_14 := arithEq_of_rows h (row := 625) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_14
  simp only [← c39534, ← c39535, ← c39536] at e_625_14
  have e_625_15 := arithEq_of_rows h (row := 625) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_15
  simp only [← c39537, ← c39538, k2, ← c39539] at e_625_15
  have hr := e_625_15
  simp only [e_625_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3503)) := by
  have c39540 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39541 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39542 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39543 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39544 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39545 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_16 := arithEq_of_rows h (row := 625) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_16
  simp only [← c39540, ← c39541, ← c39542] at e_625_16
  have e_625_17 := arithEq_of_rows h (row := 625) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_17
  simp only [← c39543, ← c39544, k2, ← c39545] at e_625_17
  have hr := e_625_17
  simp only [e_625_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 625 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3504)) := by
  have c39546 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39547 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39548 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39549 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39550 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39551 := (publicBatchWrapper89_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_625_18 := arithEq_of_rows h (row := 625) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_18
  simp only [← c39546, ← c39547, ← c39548] at e_625_18
  have e_625_19 := arithEq_of_rows h (row := 625) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_19
  simp only [← c39549, ← c39550, k2, ← c39551] at e_625_19
  have hr := e_625_19
  simp only [e_625_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3505)) := by
  have c39552 := (publicBatchWrapper89_copies1236 a h).1
  have c39553 := (publicBatchWrapper89_copies1236 a h).2.1
  have c39554 := (publicBatchWrapper89_copies1236 a h).2.2.1
  have c39555 := (publicBatchWrapper89_copies1236 a h).2.2.2.1
  have c39556 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.1
  have c39557 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_0 := arithEq_of_rows h (row := 626) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_0
  simp only [← c39552, ← c39553, ← c39554] at e_626_0
  have e_626_1 := arithEq_of_rows h (row := 626) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_1
  simp only [← c39555, ← c39556, k2, ← c39557] at e_626_1
  have hr := e_626_1
  simp only [e_626_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3506)) := by
  have c39558 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.1
  have c39559 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.1
  have c39560 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.1
  have c39561 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.1
  have c39562 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39563 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_2 := arithEq_of_rows h (row := 626) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_2
  simp only [← c39558, ← c39559, ← c39560] at e_626_2
  have e_626_3 := arithEq_of_rows h (row := 626) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_3
  simp only [← c39561, ← c39562, k2, ← c39563] at e_626_3
  have hr := e_626_3
  simp only [e_626_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3507)) := by
  have c39564 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39565 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39566 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39567 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39568 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39569 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_4 := arithEq_of_rows h (row := 626) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_4
  simp only [← c39564, ← c39565, ← c39566] at e_626_4
  have e_626_5 := arithEq_of_rows h (row := 626) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_5
  simp only [← c39567, ← c39568, k2, ← c39569] at e_626_5
  have hr := e_626_5
  simp only [e_626_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3508)) := by
  have c39570 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39571 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39572 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39573 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39574 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39575 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_6 := arithEq_of_rows h (row := 626) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_6
  simp only [← c39570, ← c39571, ← c39572] at e_626_6
  have e_626_7 := arithEq_of_rows h (row := 626) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_7
  simp only [← c39573, ← c39574, k2, ← c39575] at e_626_7
  have hr := e_626_7
  simp only [e_626_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3509)) := by
  have c39576 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39577 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39578 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39579 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39580 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39581 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_8 := arithEq_of_rows h (row := 626) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_8
  simp only [← c39576, ← c39577, ← c39578] at e_626_8
  have e_626_9 := arithEq_of_rows h (row := 626) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_9
  simp only [← c39579, ← c39580, k2, ← c39581] at e_626_9
  have hr := e_626_9
  simp only [e_626_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3510)) := by
  have c39582 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39583 := (publicBatchWrapper89_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39584 := (publicBatchWrapper89_copies1237 a h).1
  have c39585 := (publicBatchWrapper89_copies1237 a h).2.1
  have c39586 := (publicBatchWrapper89_copies1237 a h).2.2.1
  have c39587 := (publicBatchWrapper89_copies1237 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_10 := arithEq_of_rows h (row := 626) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_10
  simp only [← c39582, ← c39583, ← c39584] at e_626_10
  have e_626_11 := arithEq_of_rows h (row := 626) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_11
  simp only [← c39585, ← c39586, k2, ← c39587] at e_626_11
  have hr := e_626_11
  simp only [e_626_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3511)) := by
  have c39588 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.1
  have c39589 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.1
  have c39590 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.1
  have c39591 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.1
  have c39592 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.1
  have c39593 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_12 := arithEq_of_rows h (row := 626) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_12
  simp only [← c39588, ← c39589, ← c39590] at e_626_12
  have e_626_13 := arithEq_of_rows h (row := 626) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_13
  simp only [← c39591, ← c39592, k2, ← c39593] at e_626_13
  have hr := e_626_13
  simp only [e_626_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3512)) := by
  have c39594 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39595 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39596 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39597 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39598 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39599 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_14 := arithEq_of_rows h (row := 626) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_14
  simp only [← c39594, ← c39595, ← c39596] at e_626_14
  have e_626_15 := arithEq_of_rows h (row := 626) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_15
  simp only [← c39597, ← c39598, k2, ← c39599] at e_626_15
  have hr := e_626_15
  simp only [e_626_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3513)) := by
  have c39600 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39601 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39602 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39603 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39604 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39605 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_16 := arithEq_of_rows h (row := 626) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_16
  simp only [← c39600, ← c39601, ← c39602] at e_626_16
  have e_626_17 := arithEq_of_rows h (row := 626) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_17
  simp only [← c39603, ← c39604, k2, ← c39605] at e_626_17
  have hr := e_626_17
  simp only [e_626_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 626 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3514)) := by
  have c39606 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39607 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39608 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39609 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39610 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39611 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_626_18 := arithEq_of_rows h (row := 626) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_18
  simp only [← c39606, ← c39607, ← c39608] at e_626_18
  have e_626_19 := arithEq_of_rows h (row := 626) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_19
  simp only [← c39609, ← c39610, k2, ← c39611] at e_626_19
  have hr := e_626_19
  simp only [e_626_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3515)) := by
  have c39612 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39613 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39614 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39615 := (publicBatchWrapper89_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39616 := (publicBatchWrapper89_copies1238 a h).1
  have c39617 := (publicBatchWrapper89_copies1238 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_0 := arithEq_of_rows h (row := 627) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_0
  simp only [← c39612, ← c39613, ← c39614] at e_627_0
  have e_627_1 := arithEq_of_rows h (row := 627) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_1
  simp only [← c39615, ← c39616, k2, ← c39617] at e_627_1
  have hr := e_627_1
  simp only [e_627_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3516)) := by
  have c39618 := (publicBatchWrapper89_copies1238 a h).2.2.1
  have c39619 := (publicBatchWrapper89_copies1238 a h).2.2.2.1
  have c39620 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.1
  have c39621 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.1
  have c39622 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.1
  have c39623 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_2 := arithEq_of_rows h (row := 627) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_2
  simp only [← c39618, ← c39619, ← c39620] at e_627_2
  have e_627_3 := arithEq_of_rows h (row := 627) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_3
  simp only [← c39621, ← c39622, k2, ← c39623] at e_627_3
  have hr := e_627_3
  simp only [e_627_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3517)) := by
  have c39624 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.1
  have c39625 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.1
  have c39626 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39627 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39628 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39629 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_4 := arithEq_of_rows h (row := 627) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_4
  simp only [← c39624, ← c39625, ← c39626] at e_627_4
  have e_627_5 := arithEq_of_rows h (row := 627) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_5
  simp only [← c39627, ← c39628, k2, ← c39629] at e_627_5
  have hr := e_627_5
  simp only [e_627_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3518)) := by
  have c39630 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39631 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39632 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39633 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39634 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39635 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_6 := arithEq_of_rows h (row := 627) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_6
  simp only [← c39630, ← c39631, ← c39632] at e_627_6
  have e_627_7 := arithEq_of_rows h (row := 627) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_7
  simp only [← c39633, ← c39634, k2, ← c39635] at e_627_7
  have hr := e_627_7
  simp only [e_627_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3519)) := by
  have c39636 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39637 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39638 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39639 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39640 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39641 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_8 := arithEq_of_rows h (row := 627) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_8
  simp only [← c39636, ← c39637, ← c39638] at e_627_8
  have e_627_9 := arithEq_of_rows h (row := 627) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_9
  simp only [← c39639, ← c39640, k2, ← c39641] at e_627_9
  have hr := e_627_9
  simp only [e_627_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3520)) := by
  have c39642 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39643 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39644 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39645 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39646 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39647 := (publicBatchWrapper89_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_10 := arithEq_of_rows h (row := 627) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_10
  simp only [← c39642, ← c39643, ← c39644] at e_627_10
  have e_627_11 := arithEq_of_rows h (row := 627) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_11
  simp only [← c39645, ← c39646, k2, ← c39647] at e_627_11
  have hr := e_627_11
  simp only [e_627_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3521)) := by
  have c39648 := (publicBatchWrapper89_copies1239 a h).1
  have c39649 := (publicBatchWrapper89_copies1239 a h).2.1
  have c39650 := (publicBatchWrapper89_copies1239 a h).2.2.1
  have c39651 := (publicBatchWrapper89_copies1239 a h).2.2.2.1
  have c39652 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.1
  have c39653 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_12 := arithEq_of_rows h (row := 627) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_12
  simp only [← c39648, ← c39649, ← c39650] at e_627_12
  have e_627_13 := arithEq_of_rows h (row := 627) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_13
  simp only [← c39651, ← c39652, k2, ← c39653] at e_627_13
  have hr := e_627_13
  simp only [e_627_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3522)) := by
  have c39654 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.1
  have c39655 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.1
  have c39656 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.1
  have c39657 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.1
  have c39658 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39659 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_14 := arithEq_of_rows h (row := 627) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_14
  simp only [← c39654, ← c39655, ← c39656] at e_627_14
  have e_627_15 := arithEq_of_rows h (row := 627) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_15
  simp only [← c39657, ← c39658, k2, ← c39659] at e_627_15
  have hr := e_627_15
  simp only [e_627_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3523)) := by
  have c39660 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39661 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39662 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39663 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39664 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39665 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_16 := arithEq_of_rows h (row := 627) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_16
  simp only [← c39660, ← c39661, ← c39662] at e_627_16
  have e_627_17 := arithEq_of_rows h (row := 627) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_17
  simp only [← c39663, ← c39664, k2, ← c39665] at e_627_17
  have hr := e_627_17
  simp only [e_627_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 627 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3524)) := by
  have c39666 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39667 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39668 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39669 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39670 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39671 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_627_18 := arithEq_of_rows h (row := 627) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_18
  simp only [← c39666, ← c39667, ← c39668] at e_627_18
  have e_627_19 := arithEq_of_rows h (row := 627) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_19
  simp only [← c39669, ← c39670, k2, ← c39671] at e_627_19
  have hr := e_627_19
  simp only [e_627_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3525)) := by
  have c39672 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39673 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39674 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39675 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39676 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39677 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_0 := arithEq_of_rows h (row := 628) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_0
  simp only [← c39672, ← c39673, ← c39674] at e_628_0
  have e_628_1 := arithEq_of_rows h (row := 628) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_1
  simp only [← c39675, ← c39676, k2, ← c39677] at e_628_1
  have hr := e_628_1
  simp only [e_628_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3526)) := by
  have c39678 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39679 := (publicBatchWrapper89_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39680 := (publicBatchWrapper89_copies1240 a h).1
  have c39681 := (publicBatchWrapper89_copies1240 a h).2.1
  have c39682 := (publicBatchWrapper89_copies1240 a h).2.2.1
  have c39683 := (publicBatchWrapper89_copies1240 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_2 := arithEq_of_rows h (row := 628) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_2
  simp only [← c39678, ← c39679, ← c39680] at e_628_2
  have e_628_3 := arithEq_of_rows h (row := 628) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_3
  simp only [← c39681, ← c39682, k2, ← c39683] at e_628_3
  have hr := e_628_3
  simp only [e_628_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3527)) := by
  have c39684 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.1
  have c39685 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.1
  have c39686 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.1
  have c39687 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.1
  have c39688 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.1
  have c39689 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_4 := arithEq_of_rows h (row := 628) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_4
  simp only [← c39684, ← c39685, ← c39686] at e_628_4
  have e_628_5 := arithEq_of_rows h (row := 628) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_5
  simp only [← c39687, ← c39688, k2, ← c39689] at e_628_5
  have hr := e_628_5
  simp only [e_628_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3528)) := by
  have c39690 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39691 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39692 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39693 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39694 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39695 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_6 := arithEq_of_rows h (row := 628) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_6
  simp only [← c39690, ← c39691, ← c39692] at e_628_6
  have e_628_7 := arithEq_of_rows h (row := 628) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_7
  simp only [← c39693, ← c39694, k2, ← c39695] at e_628_7
  have hr := e_628_7
  simp only [e_628_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3529)) := by
  have c39696 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39697 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39698 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39699 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39700 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39701 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_8 := arithEq_of_rows h (row := 628) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_8
  simp only [← c39696, ← c39697, ← c39698] at e_628_8
  have e_628_9 := arithEq_of_rows h (row := 628) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_9
  simp only [← c39699, ← c39700, k2, ← c39701] at e_628_9
  have hr := e_628_9
  simp only [e_628_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3530)) := by
  have c39702 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39703 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39704 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39705 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39706 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39707 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_10 := arithEq_of_rows h (row := 628) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_10
  simp only [← c39702, ← c39703, ← c39704] at e_628_10
  have e_628_11 := arithEq_of_rows h (row := 628) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_11
  simp only [← c39705, ← c39706, k2, ← c39707] at e_628_11
  have hr := e_628_11
  simp only [e_628_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3531)) := by
  have c39708 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39709 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39710 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39711 := (publicBatchWrapper89_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39712 := (publicBatchWrapper89_copies1241 a h).1
  have c39713 := (publicBatchWrapper89_copies1241 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_12 := arithEq_of_rows h (row := 628) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_12
  simp only [← c39708, ← c39709, ← c39710] at e_628_12
  have e_628_13 := arithEq_of_rows h (row := 628) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_13
  simp only [← c39711, ← c39712, k2, ← c39713] at e_628_13
  have hr := e_628_13
  simp only [e_628_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3532)) := by
  have c39714 := (publicBatchWrapper89_copies1241 a h).2.2.1
  have c39715 := (publicBatchWrapper89_copies1241 a h).2.2.2.1
  have c39716 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.1
  have c39717 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.1
  have c39718 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.1
  have c39719 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_14 := arithEq_of_rows h (row := 628) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_14
  simp only [← c39714, ← c39715, ← c39716] at e_628_14
  have e_628_15 := arithEq_of_rows h (row := 628) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_15
  simp only [← c39717, ← c39718, k2, ← c39719] at e_628_15
  have hr := e_628_15
  simp only [e_628_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3533)) := by
  have c39720 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.1
  have c39721 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.1
  have c39722 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39723 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39724 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39725 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_16 := arithEq_of_rows h (row := 628) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_16
  simp only [← c39720, ← c39721, ← c39722] at e_628_16
  have e_628_17 := arithEq_of_rows h (row := 628) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_17
  simp only [← c39723, ← c39724, k2, ← c39725] at e_628_17
  have hr := e_628_17
  simp only [e_628_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 628 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3534)) := by
  have c39726 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39727 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39728 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39729 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39730 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39731 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_628_18 := arithEq_of_rows h (row := 628) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_18
  simp only [← c39726, ← c39727, ← c39728] at e_628_18
  have e_628_19 := arithEq_of_rows h (row := 628) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_19
  simp only [← c39729, ← c39730, k2, ← c39731] at e_628_19
  have hr := e_628_19
  simp only [e_628_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3535)) := by
  have c39732 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39733 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39734 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39735 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39736 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39737 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_0 := arithEq_of_rows h (row := 629) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_0
  simp only [← c39732, ← c39733, ← c39734] at e_629_0
  have e_629_1 := arithEq_of_rows h (row := 629) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_1
  simp only [← c39735, ← c39736, k2, ← c39737] at e_629_1
  have hr := e_629_1
  simp only [e_629_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3536)) := by
  have c39738 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39739 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39740 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39741 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39742 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39743 := (publicBatchWrapper89_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_2 := arithEq_of_rows h (row := 629) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_2
  simp only [← c39738, ← c39739, ← c39740] at e_629_2
  have e_629_3 := arithEq_of_rows h (row := 629) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_3
  simp only [← c39741, ← c39742, k2, ← c39743] at e_629_3
  have hr := e_629_3
  simp only [e_629_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3537)) := by
  have c39744 := (publicBatchWrapper89_copies1242 a h).1
  have c39745 := (publicBatchWrapper89_copies1242 a h).2.1
  have c39746 := (publicBatchWrapper89_copies1242 a h).2.2.1
  have c39747 := (publicBatchWrapper89_copies1242 a h).2.2.2.1
  have c39748 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.1
  have c39749 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_4 := arithEq_of_rows h (row := 629) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_4
  simp only [← c39744, ← c39745, ← c39746] at e_629_4
  have e_629_5 := arithEq_of_rows h (row := 629) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_5
  simp only [← c39747, ← c39748, k2, ← c39749] at e_629_5
  have hr := e_629_5
  simp only [e_629_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3538)) := by
  have c39750 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.1
  have c39751 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.1
  have c39752 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.1
  have c39753 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.1
  have c39754 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39755 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_6 := arithEq_of_rows h (row := 629) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_6
  simp only [← c39750, ← c39751, ← c39752] at e_629_6
  have e_629_7 := arithEq_of_rows h (row := 629) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_7
  simp only [← c39753, ← c39754, k2, ← c39755] at e_629_7
  have hr := e_629_7
  simp only [e_629_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3539)) := by
  have c39756 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39757 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39758 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39759 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39760 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39761 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_8 := arithEq_of_rows h (row := 629) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_8
  simp only [← c39756, ← c39757, ← c39758] at e_629_8
  have e_629_9 := arithEq_of_rows h (row := 629) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_9
  simp only [← c39759, ← c39760, k2, ← c39761] at e_629_9
  have hr := e_629_9
  simp only [e_629_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3540)) := by
  have c39762 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39763 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39764 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39765 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39766 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39767 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_10 := arithEq_of_rows h (row := 629) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_10
  simp only [← c39762, ← c39763, ← c39764] at e_629_10
  have e_629_11 := arithEq_of_rows h (row := 629) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_11
  simp only [← c39765, ← c39766, k2, ← c39767] at e_629_11
  have hr := e_629_11
  simp only [e_629_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3541)) := by
  have c39768 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39769 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39770 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39771 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39772 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39773 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_12 := arithEq_of_rows h (row := 629) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_12
  simp only [← c39768, ← c39769, ← c39770] at e_629_12
  have e_629_13 := arithEq_of_rows h (row := 629) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_13
  simp only [← c39771, ← c39772, k2, ← c39773] at e_629_13
  have hr := e_629_13
  simp only [e_629_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3542)) := by
  have c39774 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39775 := (publicBatchWrapper89_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39776 := (publicBatchWrapper89_copies1243 a h).1
  have c39777 := (publicBatchWrapper89_copies1243 a h).2.1
  have c39778 := (publicBatchWrapper89_copies1243 a h).2.2.1
  have c39779 := (publicBatchWrapper89_copies1243 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_14 := arithEq_of_rows h (row := 629) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_14
  simp only [← c39774, ← c39775, ← c39776] at e_629_14
  have e_629_15 := arithEq_of_rows h (row := 629) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_15
  simp only [← c39777, ← c39778, k2, ← c39779] at e_629_15
  have hr := e_629_15
  simp only [e_629_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3543)) := by
  have c39780 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.1
  have c39781 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.1
  have c39782 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.1
  have c39783 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.1
  have c39784 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.1
  have c39785 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_16 := arithEq_of_rows h (row := 629) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_16
  simp only [← c39780, ← c39781, ← c39782] at e_629_16
  have e_629_17 := arithEq_of_rows h (row := 629) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_17
  simp only [← c39783, ← c39784, k2, ← c39785] at e_629_17
  have hr := e_629_17
  simp only [e_629_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 629 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3544)) := by
  have c39786 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39787 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39788 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39789 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39790 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39791 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_629_18 := arithEq_of_rows h (row := 629) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_18
  simp only [← c39786, ← c39787, ← c39788] at e_629_18
  have e_629_19 := arithEq_of_rows h (row := 629) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_19
  simp only [← c39789, ← c39790, k2, ← c39791] at e_629_19
  have hr := e_629_19
  simp only [e_629_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3545)) := by
  have c39792 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39793 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39794 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39795 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39796 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39797 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_0 := arithEq_of_rows h (row := 630) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_0
  simp only [← c39792, ← c39793, ← c39794] at e_630_0
  have e_630_1 := arithEq_of_rows h (row := 630) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_1
  simp only [← c39795, ← c39796, k2, ← c39797] at e_630_1
  have hr := e_630_1
  simp only [e_630_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3546)) := by
  have c39798 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39799 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39800 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39801 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39802 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39803 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_2 := arithEq_of_rows h (row := 630) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_2
  simp only [← c39798, ← c39799, ← c39800] at e_630_2
  have e_630_3 := arithEq_of_rows h (row := 630) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_3
  simp only [← c39801, ← c39802, k2, ← c39803] at e_630_3
  have hr := e_630_3
  simp only [e_630_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3547)) := by
  have c39804 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39805 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39806 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39807 := (publicBatchWrapper89_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39808 := (publicBatchWrapper89_copies1244 a h).1
  have c39809 := (publicBatchWrapper89_copies1244 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_4 := arithEq_of_rows h (row := 630) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_4
  simp only [← c39804, ← c39805, ← c39806] at e_630_4
  have e_630_5 := arithEq_of_rows h (row := 630) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_5
  simp only [← c39807, ← c39808, k2, ← c39809] at e_630_5
  have hr := e_630_5
  simp only [e_630_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3548)) := by
  have c39810 := (publicBatchWrapper89_copies1244 a h).2.2.1
  have c39811 := (publicBatchWrapper89_copies1244 a h).2.2.2.1
  have c39812 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.1
  have c39813 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.1
  have c39814 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.1
  have c39815 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_6 := arithEq_of_rows h (row := 630) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_6
  simp only [← c39810, ← c39811, ← c39812] at e_630_6
  have e_630_7 := arithEq_of_rows h (row := 630) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_7
  simp only [← c39813, ← c39814, k2, ← c39815] at e_630_7
  have hr := e_630_7
  simp only [e_630_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3549)) := by
  have c39816 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.1
  have c39817 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.1
  have c39818 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39819 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39820 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39821 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_8 := arithEq_of_rows h (row := 630) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_8
  simp only [← c39816, ← c39817, ← c39818] at e_630_8
  have e_630_9 := arithEq_of_rows h (row := 630) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_9
  simp only [← c39819, ← c39820, k2, ← c39821] at e_630_9
  have hr := e_630_9
  simp only [e_630_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3550)) := by
  have c39822 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39823 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39824 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39825 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39826 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39827 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_10 := arithEq_of_rows h (row := 630) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_10
  simp only [← c39822, ← c39823, ← c39824] at e_630_10
  have e_630_11 := arithEq_of_rows h (row := 630) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_11
  simp only [← c39825, ← c39826, k2, ← c39827] at e_630_11
  have hr := e_630_11
  simp only [e_630_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3551)) := by
  have c39828 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39829 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39830 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39831 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39832 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39833 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_12 := arithEq_of_rows h (row := 630) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_12
  simp only [← c39828, ← c39829, ← c39830] at e_630_12
  have e_630_13 := arithEq_of_rows h (row := 630) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_13
  simp only [← c39831, ← c39832, k2, ← c39833] at e_630_13
  have hr := e_630_13
  simp only [e_630_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3552)) := by
  have c39834 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39835 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39836 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39837 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39838 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39839 := (publicBatchWrapper89_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_14 := arithEq_of_rows h (row := 630) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_14
  simp only [← c39834, ← c39835, ← c39836] at e_630_14
  have e_630_15 := arithEq_of_rows h (row := 630) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_15
  simp only [← c39837, ← c39838, k2, ← c39839] at e_630_15
  have hr := e_630_15
  simp only [e_630_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3553)) := by
  have c39840 := (publicBatchWrapper89_copies1245 a h).1
  have c39841 := (publicBatchWrapper89_copies1245 a h).2.1
  have c39842 := (publicBatchWrapper89_copies1245 a h).2.2.1
  have c39843 := (publicBatchWrapper89_copies1245 a h).2.2.2.1
  have c39844 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.1
  have c39845 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_16 := arithEq_of_rows h (row := 630) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_16
  simp only [← c39840, ← c39841, ← c39842] at e_630_16
  have e_630_17 := arithEq_of_rows h (row := 630) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_17
  simp only [← c39843, ← c39844, k2, ← c39845] at e_630_17
  have hr := e_630_17
  simp only [e_630_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 630 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3554)) := by
  have c39846 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.1
  have c39847 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.1
  have c39848 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.1
  have c39849 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.1
  have c39850 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39851 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_630_18 := arithEq_of_rows h (row := 630) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_18
  simp only [← c39846, ← c39847, ← c39848] at e_630_18
  have e_630_19 := arithEq_of_rows h (row := 630) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_19
  simp only [← c39849, ← c39850, k2, ← c39851] at e_630_19
  have hr := e_630_19
  simp only [e_630_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3555)) := by
  have c39852 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39853 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39854 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39855 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39856 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39857 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_0 := arithEq_of_rows h (row := 631) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_0
  simp only [← c39852, ← c39853, ← c39854] at e_631_0
  have e_631_1 := arithEq_of_rows h (row := 631) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_1
  simp only [← c39855, ← c39856, k2, ← c39857] at e_631_1
  have hr := e_631_1
  simp only [e_631_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3556)) := by
  have c39858 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39859 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39860 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39861 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39862 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39863 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_2 := arithEq_of_rows h (row := 631) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_2
  simp only [← c39858, ← c39859, ← c39860] at e_631_2
  have e_631_3 := arithEq_of_rows h (row := 631) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_3
  simp only [← c39861, ← c39862, k2, ← c39863] at e_631_3
  have hr := e_631_3
  simp only [e_631_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3557)) := by
  have c39864 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39865 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39866 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39867 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39868 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39869 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_4 := arithEq_of_rows h (row := 631) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_4
  simp only [← c39864, ← c39865, ← c39866] at e_631_4
  have e_631_5 := arithEq_of_rows h (row := 631) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_5
  simp only [← c39867, ← c39868, k2, ← c39869] at e_631_5
  have hr := e_631_5
  simp only [e_631_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3558)) := by
  have c39870 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39871 := (publicBatchWrapper89_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39872 := (publicBatchWrapper89_copies1246 a h).1
  have c39873 := (publicBatchWrapper89_copies1246 a h).2.1
  have c39874 := (publicBatchWrapper89_copies1246 a h).2.2.1
  have c39875 := (publicBatchWrapper89_copies1246 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_6 := arithEq_of_rows h (row := 631) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_6
  simp only [← c39870, ← c39871, ← c39872] at e_631_6
  have e_631_7 := arithEq_of_rows h (row := 631) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_7
  simp only [← c39873, ← c39874, k2, ← c39875] at e_631_7
  have hr := e_631_7
  simp only [e_631_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3559)) := by
  have c39876 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.1
  have c39877 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.1
  have c39878 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.1
  have c39879 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.1
  have c39880 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.1
  have c39881 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_8 := arithEq_of_rows h (row := 631) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_8
  simp only [← c39876, ← c39877, ← c39878] at e_631_8
  have e_631_9 := arithEq_of_rows h (row := 631) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_9
  simp only [← c39879, ← c39880, k2, ← c39881] at e_631_9
  have hr := e_631_9
  simp only [e_631_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3560)) := by
  have c39882 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39883 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39884 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39885 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39886 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39887 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_10 := arithEq_of_rows h (row := 631) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_10
  simp only [← c39882, ← c39883, ← c39884] at e_631_10
  have e_631_11 := arithEq_of_rows h (row := 631) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_11
  simp only [← c39885, ← c39886, k2, ← c39887] at e_631_11
  have hr := e_631_11
  simp only [e_631_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3561)) := by
  have c39888 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39889 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39890 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39891 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39892 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39893 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_12 := arithEq_of_rows h (row := 631) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_12
  simp only [← c39888, ← c39889, ← c39890] at e_631_12
  have e_631_13 := arithEq_of_rows h (row := 631) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_13
  simp only [← c39891, ← c39892, k2, ← c39893] at e_631_13
  have hr := e_631_13
  simp only [e_631_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3562)) := by
  have c39894 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39895 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39896 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39897 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39898 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39899 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_14 := arithEq_of_rows h (row := 631) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_14
  simp only [← c39894, ← c39895, ← c39896] at e_631_14
  have e_631_15 := arithEq_of_rows h (row := 631) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_15
  simp only [← c39897, ← c39898, k2, ← c39899] at e_631_15
  have hr := e_631_15
  simp only [e_631_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3563)) := by
  have c39900 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39901 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39902 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39903 := (publicBatchWrapper89_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39904 := (publicBatchWrapper89_copies1247 a h).1
  have c39905 := (publicBatchWrapper89_copies1247 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_16 := arithEq_of_rows h (row := 631) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_16
  simp only [← c39900, ← c39901, ← c39902] at e_631_16
  have e_631_17 := arithEq_of_rows h (row := 631) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_17
  simp only [← c39903, ← c39904, k2, ← c39905] at e_631_17
  have hr := e_631_17
  simp only [e_631_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 631 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3564)) := by
  have c39906 := (publicBatchWrapper89_copies1247 a h).2.2.1
  have c39907 := (publicBatchWrapper89_copies1247 a h).2.2.2.1
  have c39908 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.1
  have c39909 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.1
  have c39910 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.1
  have c39911 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_631_18 := arithEq_of_rows h (row := 631) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_18
  simp only [← c39906, ← c39907, ← c39908] at e_631_18
  have e_631_19 := arithEq_of_rows h (row := 631) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_19
  simp only [← c39909, ← c39910, k2, ← c39911] at e_631_19
  have hr := e_631_19
  simp only [e_631_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3565)) := by
  have c39912 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.1
  have c39913 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.1
  have c39914 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39915 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39916 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39917 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_0 := arithEq_of_rows h (row := 632) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_0
  simp only [← c39912, ← c39913, ← c39914] at e_632_0
  have e_632_1 := arithEq_of_rows h (row := 632) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_1
  simp only [← c39915, ← c39916, k2, ← c39917] at e_632_1
  have hr := e_632_1
  simp only [e_632_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3566)) := by
  have c39918 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39919 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39920 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39921 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39922 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39923 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_2 := arithEq_of_rows h (row := 632) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_2
  simp only [← c39918, ← c39919, ← c39920] at e_632_2
  have e_632_3 := arithEq_of_rows h (row := 632) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_3
  simp only [← c39921, ← c39922, k2, ← c39923] at e_632_3
  have hr := e_632_3
  simp only [e_632_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3567)) := by
  have c39924 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39925 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39926 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39927 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39928 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39929 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_4 := arithEq_of_rows h (row := 632) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_4
  simp only [← c39924, ← c39925, ← c39926] at e_632_4
  have e_632_5 := arithEq_of_rows h (row := 632) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_5
  simp only [← c39927, ← c39928, k2, ← c39929] at e_632_5
  have hr := e_632_5
  simp only [e_632_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3568)) := by
  have c39930 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39931 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39932 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39933 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39934 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39935 := (publicBatchWrapper89_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_6 := arithEq_of_rows h (row := 632) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_6
  simp only [← c39930, ← c39931, ← c39932] at e_632_6
  have e_632_7 := arithEq_of_rows h (row := 632) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_7
  simp only [← c39933, ← c39934, k2, ← c39935] at e_632_7
  have hr := e_632_7
  simp only [e_632_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3569)) := by
  have c39936 := (publicBatchWrapper89_copies1248 a h).1
  have c39937 := (publicBatchWrapper89_copies1248 a h).2.1
  have c39938 := (publicBatchWrapper89_copies1248 a h).2.2.1
  have c39939 := (publicBatchWrapper89_copies1248 a h).2.2.2.1
  have c39940 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.1
  have c39941 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_8 := arithEq_of_rows h (row := 632) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_8
  simp only [← c39936, ← c39937, ← c39938] at e_632_8
  have e_632_9 := arithEq_of_rows h (row := 632) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_9
  simp only [← c39939, ← c39940, k2, ← c39941] at e_632_9
  have hr := e_632_9
  simp only [e_632_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3610)) := by
  have c39942 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.1
  have c39943 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.1
  have c39944 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.1
  have c39945 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.1
  have c39946 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39947 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_10 := arithEq_of_rows h (row := 632) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_10
  simp only [← c39942, ← c39943, ← c39944] at e_632_10
  have e_632_11 := arithEq_of_rows h (row := 632) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_11
  simp only [← c39945, ← c39946, k2, ← c39947] at e_632_11
  have hr := e_632_11
  simp only [e_632_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3611)) := by
  have c39948 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39949 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39950 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39951 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39952 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39953 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_12 := arithEq_of_rows h (row := 632) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_12
  simp only [← c39948, ← c39949, ← c39950] at e_632_12
  have e_632_13 := arithEq_of_rows h (row := 632) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_13
  simp only [← c39951, ← c39952, k2, ← c39953] at e_632_13
  have hr := e_632_13
  simp only [e_632_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3612)) := by
  have c39954 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39955 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39956 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39957 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39958 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39959 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_14 := arithEq_of_rows h (row := 632) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_14
  simp only [← c39954, ← c39955, ← c39956] at e_632_14
  have e_632_15 := arithEq_of_rows h (row := 632) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_15
  simp only [← c39957, ← c39958, k2, ← c39959] at e_632_15
  have hr := e_632_15
  simp only [e_632_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3613)) := by
  have c39960 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39961 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39962 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39963 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39964 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39965 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_16 := arithEq_of_rows h (row := 632) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_16
  simp only [← c39960, ← c39961, ← c39962] at e_632_16
  have e_632_17 := arithEq_of_rows h (row := 632) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_17
  simp only [← c39963, ← c39964, k2, ← c39965] at e_632_17
  have hr := e_632_17
  simp only [e_632_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 632 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3614)) := by
  have c39966 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39967 := (publicBatchWrapper89_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39968 := (publicBatchWrapper89_copies1249 a h).1
  have c39969 := (publicBatchWrapper89_copies1249 a h).2.1
  have c39970 := (publicBatchWrapper89_copies1249 a h).2.2.1
  have c39971 := (publicBatchWrapper89_copies1249 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_632_18 := arithEq_of_rows h (row := 632) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_18
  simp only [← c39966, ← c39967, ← c39968] at e_632_18
  have e_632_19 := arithEq_of_rows h (row := 632) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_19
  simp only [← c39969, ← c39970, k2, ← c39971] at e_632_19
  have hr := e_632_19
  simp only [e_632_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3615)) := by
  have c39972 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.1
  have c39973 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.1
  have c39974 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.1
  have c39975 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.1
  have c39976 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.1
  have c39977 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_0 := arithEq_of_rows h (row := 633) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_0
  simp only [← c39972, ← c39973, ← c39974] at e_633_0
  have e_633_1 := arithEq_of_rows h (row := 633) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_1
  simp only [← c39975, ← c39976, k2, ← c39977] at e_633_1
  have hr := e_633_1
  simp only [e_633_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3616)) := by
  have c39978 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39979 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39980 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39981 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39982 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39983 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_2 := arithEq_of_rows h (row := 633) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_2
  simp only [← c39978, ← c39979, ← c39980] at e_633_2
  have e_633_3 := arithEq_of_rows h (row := 633) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_3
  simp only [← c39981, ← c39982, k2, ← c39983] at e_633_3
  have hr := e_633_3
  simp only [e_633_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3617)) := by
  have c39984 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39985 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39986 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39987 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39988 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39989 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_4 := arithEq_of_rows h (row := 633) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_4
  simp only [← c39984, ← c39985, ← c39986] at e_633_4
  have e_633_5 := arithEq_of_rows h (row := 633) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_5
  simp only [← c39987, ← c39988, k2, ← c39989] at e_633_5
  have hr := e_633_5
  simp only [e_633_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3618)) := by
  have c39990 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39991 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39992 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39993 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39994 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39995 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_6 := arithEq_of_rows h (row := 633) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_6
  simp only [← c39990, ← c39991, ← c39992] at e_633_6
  have e_633_7 := arithEq_of_rows h (row := 633) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_7
  simp only [← c39993, ← c39994, k2, ← c39995] at e_633_7
  have hr := e_633_7
  simp only [e_633_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3619)) := by
  have c39996 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39997 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39998 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39999 := (publicBatchWrapper89_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40000 := (publicBatchWrapper89_copies1250 a h).1
  have c40001 := (publicBatchWrapper89_copies1250 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_8 := arithEq_of_rows h (row := 633) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_8
  simp only [← c39996, ← c39997, ← c39998] at e_633_8
  have e_633_9 := arithEq_of_rows h (row := 633) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_9
  simp only [← c39999, ← c40000, k2, ← c40001] at e_633_9
  have hr := e_633_9
  simp only [e_633_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3620)) := by
  have c40002 := (publicBatchWrapper89_copies1250 a h).2.2.1
  have c40003 := (publicBatchWrapper89_copies1250 a h).2.2.2.1
  have c40004 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.1
  have c40005 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.1
  have c40006 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.1
  have c40007 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_10 := arithEq_of_rows h (row := 633) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_10
  simp only [← c40002, ← c40003, ← c40004] at e_633_10
  have e_633_11 := arithEq_of_rows h (row := 633) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_11
  simp only [← c40005, ← c40006, k2, ← c40007] at e_633_11
  have hr := e_633_11
  simp only [e_633_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3621)) := by
  have c40008 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.1
  have c40009 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.1
  have c40010 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40011 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40012 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40013 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_12 := arithEq_of_rows h (row := 633) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_12
  simp only [← c40008, ← c40009, ← c40010] at e_633_12
  have e_633_13 := arithEq_of_rows h (row := 633) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_13
  simp only [← c40011, ← c40012, k2, ← c40013] at e_633_13
  have hr := e_633_13
  simp only [e_633_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3622)) := by
  have c40014 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40015 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40016 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40017 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40018 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40019 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_14 := arithEq_of_rows h (row := 633) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_14
  simp only [← c40014, ← c40015, ← c40016] at e_633_14
  have e_633_15 := arithEq_of_rows h (row := 633) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_15
  simp only [← c40017, ← c40018, k2, ← c40019] at e_633_15
  have hr := e_633_15
  simp only [e_633_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3623)) := by
  have c40020 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40021 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40022 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40023 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40024 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40025 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_16 := arithEq_of_rows h (row := 633) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_16
  simp only [← c40020, ← c40021, ← c40022] at e_633_16
  have e_633_17 := arithEq_of_rows h (row := 633) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_17
  simp only [← c40023, ← c40024, k2, ← c40025] at e_633_17
  have hr := e_633_17
  simp only [e_633_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 633 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3624)) := by
  have c40026 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40027 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40028 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40029 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40030 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40031 := (publicBatchWrapper89_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_633_18 := arithEq_of_rows h (row := 633) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_18
  simp only [← c40026, ← c40027, ← c40028] at e_633_18
  have e_633_19 := arithEq_of_rows h (row := 633) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_19
  simp only [← c40029, ← c40030, k2, ← c40031] at e_633_19
  have hr := e_633_19
  simp only [e_633_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3625)) := by
  have c40032 := (publicBatchWrapper89_copies1251 a h).1
  have c40033 := (publicBatchWrapper89_copies1251 a h).2.1
  have c40034 := (publicBatchWrapper89_copies1251 a h).2.2.1
  have c40035 := (publicBatchWrapper89_copies1251 a h).2.2.2.1
  have c40036 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.1
  have c40037 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_0 := arithEq_of_rows h (row := 634) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_0
  simp only [← c40032, ← c40033, ← c40034] at e_634_0
  have e_634_1 := arithEq_of_rows h (row := 634) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_1
  simp only [← c40035, ← c40036, k2, ← c40037] at e_634_1
  have hr := e_634_1
  simp only [e_634_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3626)) := by
  have c40038 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.1
  have c40039 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.1
  have c40040 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.1
  have c40041 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.1
  have c40042 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40043 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_2 := arithEq_of_rows h (row := 634) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_2
  simp only [← c40038, ← c40039, ← c40040] at e_634_2
  have e_634_3 := arithEq_of_rows h (row := 634) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_3
  simp only [← c40041, ← c40042, k2, ← c40043] at e_634_3
  have hr := e_634_3
  simp only [e_634_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3627)) := by
  have c40044 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40045 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40046 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40047 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40048 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40049 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_4 := arithEq_of_rows h (row := 634) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_4
  simp only [← c40044, ← c40045, ← c40046] at e_634_4
  have e_634_5 := arithEq_of_rows h (row := 634) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_5
  simp only [← c40047, ← c40048, k2, ← c40049] at e_634_5
  have hr := e_634_5
  simp only [e_634_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3628)) := by
  have c40050 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40051 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40052 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40053 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40054 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40055 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_6 := arithEq_of_rows h (row := 634) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_6
  simp only [← c40050, ← c40051, ← c40052] at e_634_6
  have e_634_7 := arithEq_of_rows h (row := 634) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_7
  simp only [← c40053, ← c40054, k2, ← c40055] at e_634_7
  have hr := e_634_7
  simp only [e_634_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3629)) := by
  have c40056 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40057 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40058 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40059 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40060 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40061 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_8 := arithEq_of_rows h (row := 634) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_8
  simp only [← c40056, ← c40057, ← c40058] at e_634_8
  have e_634_9 := arithEq_of_rows h (row := 634) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_9
  simp only [← c40059, ← c40060, k2, ← c40061] at e_634_9
  have hr := e_634_9
  simp only [e_634_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3630)) := by
  have c40062 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40063 := (publicBatchWrapper89_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40064 := (publicBatchWrapper89_copies1252 a h).1
  have c40065 := (publicBatchWrapper89_copies1252 a h).2.1
  have c40066 := (publicBatchWrapper89_copies1252 a h).2.2.1
  have c40067 := (publicBatchWrapper89_copies1252 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_10 := arithEq_of_rows h (row := 634) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_10
  simp only [← c40062, ← c40063, ← c40064] at e_634_10
  have e_634_11 := arithEq_of_rows h (row := 634) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_11
  simp only [← c40065, ← c40066, k2, ← c40067] at e_634_11
  have hr := e_634_11
  simp only [e_634_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3631)) := by
  have c40068 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.1
  have c40069 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.1
  have c40070 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.1
  have c40071 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.1
  have c40072 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.1
  have c40073 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_12 := arithEq_of_rows h (row := 634) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_12
  simp only [← c40068, ← c40069, ← c40070] at e_634_12
  have e_634_13 := arithEq_of_rows h (row := 634) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_13
  simp only [← c40071, ← c40072, k2, ← c40073] at e_634_13
  have hr := e_634_13
  simp only [e_634_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3632)) := by
  have c40074 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40075 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40076 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40077 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40078 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40079 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_14 := arithEq_of_rows h (row := 634) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_14
  simp only [← c40074, ← c40075, ← c40076] at e_634_14
  have e_634_15 := arithEq_of_rows h (row := 634) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_15
  simp only [← c40077, ← c40078, k2, ← c40079] at e_634_15
  have hr := e_634_15
  simp only [e_634_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3633)) := by
  have c40080 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40081 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40082 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40083 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40084 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40085 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_16 := arithEq_of_rows h (row := 634) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_16
  simp only [← c40080, ← c40081, ← c40082] at e_634_16
  have e_634_17 := arithEq_of_rows h (row := 634) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_17
  simp only [← c40083, ← c40084, k2, ← c40085] at e_634_17
  have hr := e_634_17
  simp only [e_634_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 634 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3634)) := by
  have c40086 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40087 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40088 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40089 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40090 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40091 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_634_18 := arithEq_of_rows h (row := 634) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_18
  simp only [← c40086, ← c40087, ← c40088] at e_634_18
  have e_634_19 := arithEq_of_rows h (row := 634) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_19
  simp only [← c40089, ← c40090, k2, ← c40091] at e_634_19
  have hr := e_634_19
  simp only [e_634_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3635)) := by
  have c40092 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40093 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40094 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40095 := (publicBatchWrapper89_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40096 := (publicBatchWrapper89_copies1253 a h).1
  have c40097 := (publicBatchWrapper89_copies1253 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_0 := arithEq_of_rows h (row := 635) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_0
  simp only [← c40092, ← c40093, ← c40094] at e_635_0
  have e_635_1 := arithEq_of_rows h (row := 635) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_1
  simp only [← c40095, ← c40096, k2, ← c40097] at e_635_1
  have hr := e_635_1
  simp only [e_635_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3636)) := by
  have c40098 := (publicBatchWrapper89_copies1253 a h).2.2.1
  have c40099 := (publicBatchWrapper89_copies1253 a h).2.2.2.1
  have c40100 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.1
  have c40101 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.1
  have c40102 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.1
  have c40103 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_2 := arithEq_of_rows h (row := 635) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_2
  simp only [← c40098, ← c40099, ← c40100] at e_635_2
  have e_635_3 := arithEq_of_rows h (row := 635) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_3
  simp only [← c40101, ← c40102, k2, ← c40103] at e_635_3
  have hr := e_635_3
  simp only [e_635_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3637)) := by
  have c40104 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.1
  have c40105 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.1
  have c40106 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40107 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40108 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40109 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_4 := arithEq_of_rows h (row := 635) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_4
  simp only [← c40104, ← c40105, ← c40106] at e_635_4
  have e_635_5 := arithEq_of_rows h (row := 635) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_5
  simp only [← c40107, ← c40108, k2, ← c40109] at e_635_5
  have hr := e_635_5
  simp only [e_635_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3638)) := by
  have c40110 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40111 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40112 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40113 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40114 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40115 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_6 := arithEq_of_rows h (row := 635) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_6
  simp only [← c40110, ← c40111, ← c40112] at e_635_6
  have e_635_7 := arithEq_of_rows h (row := 635) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_7
  simp only [← c40113, ← c40114, k2, ← c40115] at e_635_7
  have hr := e_635_7
  simp only [e_635_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3639)) := by
  have c40116 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40117 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40118 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40119 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40120 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40121 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_8 := arithEq_of_rows h (row := 635) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_8
  simp only [← c40116, ← c40117, ← c40118] at e_635_8
  have e_635_9 := arithEq_of_rows h (row := 635) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_9
  simp only [← c40119, ← c40120, k2, ← c40121] at e_635_9
  have hr := e_635_9
  simp only [e_635_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3640)) := by
  have c40122 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40123 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40124 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40125 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40126 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40127 := (publicBatchWrapper89_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_10 := arithEq_of_rows h (row := 635) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_10
  simp only [← c40122, ← c40123, ← c40124] at e_635_10
  have e_635_11 := arithEq_of_rows h (row := 635) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_11
  simp only [← c40125, ← c40126, k2, ← c40127] at e_635_11
  have hr := e_635_11
  simp only [e_635_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3641)) := by
  have c40128 := (publicBatchWrapper89_copies1254 a h).1
  have c40129 := (publicBatchWrapper89_copies1254 a h).2.1
  have c40130 := (publicBatchWrapper89_copies1254 a h).2.2.1
  have c40131 := (publicBatchWrapper89_copies1254 a h).2.2.2.1
  have c40132 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.1
  have c40133 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_12 := arithEq_of_rows h (row := 635) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_12
  simp only [← c40128, ← c40129, ← c40130] at e_635_12
  have e_635_13 := arithEq_of_rows h (row := 635) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_13
  simp only [← c40131, ← c40132, k2, ← c40133] at e_635_13
  have hr := e_635_13
  simp only [e_635_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3642)) := by
  have c40134 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.1
  have c40135 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.1
  have c40136 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.1
  have c40137 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.1
  have c40138 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40139 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_14 := arithEq_of_rows h (row := 635) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_14
  simp only [← c40134, ← c40135, ← c40136] at e_635_14
  have e_635_15 := arithEq_of_rows h (row := 635) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_15
  simp only [← c40137, ← c40138, k2, ← c40139] at e_635_15
  have hr := e_635_15
  simp only [e_635_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3643)) := by
  have c40140 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40141 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40142 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40143 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40144 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40145 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_16 := arithEq_of_rows h (row := 635) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_16
  simp only [← c40140, ← c40141, ← c40142] at e_635_16
  have e_635_17 := arithEq_of_rows h (row := 635) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_17
  simp only [← c40143, ← c40144, k2, ← c40145] at e_635_17
  have hr := e_635_17
  simp only [e_635_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 635 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3644)) := by
  have c40146 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40147 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40148 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40149 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40150 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40151 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_635_18 := arithEq_of_rows h (row := 635) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_18
  simp only [← c40146, ← c40147, ← c40148] at e_635_18
  have e_635_19 := arithEq_of_rows h (row := 635) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_19
  simp only [← c40149, ← c40150, k2, ← c40151] at e_635_19
  have hr := e_635_19
  simp only [e_635_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3645)) := by
  have c40152 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40153 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40154 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40155 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40156 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40157 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_0 := arithEq_of_rows h (row := 636) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_0
  simp only [← c40152, ← c40153, ← c40154] at e_636_0
  have e_636_1 := arithEq_of_rows h (row := 636) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_1
  simp only [← c40155, ← c40156, k2, ← c40157] at e_636_1
  have hr := e_636_1
  simp only [e_636_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3646)) := by
  have c40158 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40159 := (publicBatchWrapper89_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40160 := (publicBatchWrapper89_copies1255 a h).1
  have c40161 := (publicBatchWrapper89_copies1255 a h).2.1
  have c40162 := (publicBatchWrapper89_copies1255 a h).2.2.1
  have c40163 := (publicBatchWrapper89_copies1255 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_2 := arithEq_of_rows h (row := 636) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_2
  simp only [← c40158, ← c40159, ← c40160] at e_636_2
  have e_636_3 := arithEq_of_rows h (row := 636) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_3
  simp only [← c40161, ← c40162, k2, ← c40163] at e_636_3
  have hr := e_636_3
  simp only [e_636_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3647)) := by
  have c40164 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.1
  have c40165 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.1
  have c40166 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.1
  have c40167 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.1
  have c40168 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.1
  have c40169 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_4 := arithEq_of_rows h (row := 636) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_4
  simp only [← c40164, ← c40165, ← c40166] at e_636_4
  have e_636_5 := arithEq_of_rows h (row := 636) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_5
  simp only [← c40167, ← c40168, k2, ← c40169] at e_636_5
  have hr := e_636_5
  simp only [e_636_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3648)) := by
  have c40170 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40171 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40172 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40173 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40174 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40175 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_6 := arithEq_of_rows h (row := 636) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_6
  simp only [← c40170, ← c40171, ← c40172] at e_636_6
  have e_636_7 := arithEq_of_rows h (row := 636) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_7
  simp only [← c40173, ← c40174, k2, ← c40175] at e_636_7
  have hr := e_636_7
  simp only [e_636_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3649)) := by
  have c40176 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40177 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40178 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40179 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40180 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40181 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_8 := arithEq_of_rows h (row := 636) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_8
  simp only [← c40176, ← c40177, ← c40178] at e_636_8
  have e_636_9 := arithEq_of_rows h (row := 636) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_9
  simp only [← c40179, ← c40180, k2, ← c40181] at e_636_9
  have hr := e_636_9
  simp only [e_636_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3650)) := by
  have c40182 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40183 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40184 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40185 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40186 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40187 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_10 := arithEq_of_rows h (row := 636) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_10
  simp only [← c40182, ← c40183, ← c40184] at e_636_10
  have e_636_11 := arithEq_of_rows h (row := 636) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_11
  simp only [← c40185, ← c40186, k2, ← c40187] at e_636_11
  have hr := e_636_11
  simp only [e_636_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3651)) := by
  have c40188 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40189 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40190 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40191 := (publicBatchWrapper89_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40192 := (publicBatchWrapper89_copies1256 a h).1
  have c40193 := (publicBatchWrapper89_copies1256 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_12 := arithEq_of_rows h (row := 636) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_12
  simp only [← c40188, ← c40189, ← c40190] at e_636_12
  have e_636_13 := arithEq_of_rows h (row := 636) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_13
  simp only [← c40191, ← c40192, k2, ← c40193] at e_636_13
  have hr := e_636_13
  simp only [e_636_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3652)) := by
  have c40194 := (publicBatchWrapper89_copies1256 a h).2.2.1
  have c40195 := (publicBatchWrapper89_copies1256 a h).2.2.2.1
  have c40196 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.1
  have c40197 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.1
  have c40198 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.1
  have c40199 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_14 := arithEq_of_rows h (row := 636) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_14
  simp only [← c40194, ← c40195, ← c40196] at e_636_14
  have e_636_15 := arithEq_of_rows h (row := 636) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_15
  simp only [← c40197, ← c40198, k2, ← c40199] at e_636_15
  have hr := e_636_15
  simp only [e_636_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3653)) := by
  have c40200 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.1
  have c40201 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.1
  have c40202 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40203 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40204 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40205 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_16 := arithEq_of_rows h (row := 636) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_16
  simp only [← c40200, ← c40201, ← c40202] at e_636_16
  have e_636_17 := arithEq_of_rows h (row := 636) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_17
  simp only [← c40203, ← c40204, k2, ← c40205] at e_636_17
  have hr := e_636_17
  simp only [e_636_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 636 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3654)) := by
  have c40206 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40207 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40208 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40209 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40210 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40211 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_636_18 := arithEq_of_rows h (row := 636) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_18
  simp only [← c40206, ← c40207, ← c40208] at e_636_18
  have e_636_19 := arithEq_of_rows h (row := 636) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_19
  simp only [← c40209, ← c40210, k2, ← c40211] at e_636_19
  have hr := e_636_19
  simp only [e_636_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3655)) := by
  have c40212 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40213 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40214 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40215 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40216 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40217 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_0 := arithEq_of_rows h (row := 637) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_0
  simp only [← c40212, ← c40213, ← c40214] at e_637_0
  have e_637_1 := arithEq_of_rows h (row := 637) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_1
  simp only [← c40215, ← c40216, k2, ← c40217] at e_637_1
  have hr := e_637_1
  simp only [e_637_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3656)) := by
  have c40218 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40219 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40220 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40221 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40222 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40223 := (publicBatchWrapper89_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_2 := arithEq_of_rows h (row := 637) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_2
  simp only [← c40218, ← c40219, ← c40220] at e_637_2
  have e_637_3 := arithEq_of_rows h (row := 637) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_3
  simp only [← c40221, ← c40222, k2, ← c40223] at e_637_3
  have hr := e_637_3
  simp only [e_637_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3657)) := by
  have c40224 := (publicBatchWrapper89_copies1257 a h).1
  have c40225 := (publicBatchWrapper89_copies1257 a h).2.1
  have c40226 := (publicBatchWrapper89_copies1257 a h).2.2.1
  have c40227 := (publicBatchWrapper89_copies1257 a h).2.2.2.1
  have c40228 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.1
  have c40229 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_4 := arithEq_of_rows h (row := 637) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_4
  simp only [← c40224, ← c40225, ← c40226] at e_637_4
  have e_637_5 := arithEq_of_rows h (row := 637) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_5
  simp only [← c40227, ← c40228, k2, ← c40229] at e_637_5
  have hr := e_637_5
  simp only [e_637_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3658)) := by
  have c40230 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.1
  have c40231 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.1
  have c40232 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.1
  have c40233 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.1
  have c40234 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40235 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_6 := arithEq_of_rows h (row := 637) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_6
  simp only [← c40230, ← c40231, ← c40232] at e_637_6
  have e_637_7 := arithEq_of_rows h (row := 637) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_7
  simp only [← c40233, ← c40234, k2, ← c40235] at e_637_7
  have hr := e_637_7
  simp only [e_637_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3659)) := by
  have c40236 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40237 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40238 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40239 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40240 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40241 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_8 := arithEq_of_rows h (row := 637) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_8
  simp only [← c40236, ← c40237, ← c40238] at e_637_8
  have e_637_9 := arithEq_of_rows h (row := 637) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_9
  simp only [← c40239, ← c40240, k2, ← c40241] at e_637_9
  have hr := e_637_9
  simp only [e_637_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3660)) := by
  have c40242 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40243 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40244 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40245 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40246 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40247 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_10 := arithEq_of_rows h (row := 637) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_10
  simp only [← c40242, ← c40243, ← c40244] at e_637_10
  have e_637_11 := arithEq_of_rows h (row := 637) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_11
  simp only [← c40245, ← c40246, k2, ← c40247] at e_637_11
  have hr := e_637_11
  simp only [e_637_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3661)) := by
  have c40248 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40249 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40250 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40251 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40252 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40253 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_12 := arithEq_of_rows h (row := 637) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_12
  simp only [← c40248, ← c40249, ← c40250] at e_637_12
  have e_637_13 := arithEq_of_rows h (row := 637) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_13
  simp only [← c40251, ← c40252, k2, ← c40253] at e_637_13
  have hr := e_637_13
  simp only [e_637_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3662)) := by
  have c40254 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40255 := (publicBatchWrapper89_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40256 := (publicBatchWrapper89_copies1258 a h).1
  have c40257 := (publicBatchWrapper89_copies1258 a h).2.1
  have c40258 := (publicBatchWrapper89_copies1258 a h).2.2.1
  have c40259 := (publicBatchWrapper89_copies1258 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_14 := arithEq_of_rows h (row := 637) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_14
  simp only [← c40254, ← c40255, ← c40256] at e_637_14
  have e_637_15 := arithEq_of_rows h (row := 637) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_15
  simp only [← c40257, ← c40258, k2, ← c40259] at e_637_15
  have hr := e_637_15
  simp only [e_637_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3663)) := by
  have c40260 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.1
  have c40261 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.1
  have c40262 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.1
  have c40263 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.1
  have c40264 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.1
  have c40265 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_16 := arithEq_of_rows h (row := 637) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_16
  simp only [← c40260, ← c40261, ← c40262] at e_637_16
  have e_637_17 := arithEq_of_rows h (row := 637) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_17
  simp only [← c40263, ← c40264, k2, ← c40265] at e_637_17
  have hr := e_637_17
  simp only [e_637_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 637 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3664)) := by
  have c40266 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40267 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40268 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40269 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40270 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40271 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_637_18 := arithEq_of_rows h (row := 637) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_18
  simp only [← c40266, ← c40267, ← c40268] at e_637_18
  have e_637_19 := arithEq_of_rows h (row := 637) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_19
  simp only [← c40269, ← c40270, k2, ← c40271] at e_637_19
  have hr := e_637_19
  simp only [e_637_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3665)) := by
  have c40272 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40273 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40274 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40275 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40276 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40277 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_0 := arithEq_of_rows h (row := 638) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_0
  simp only [← c40272, ← c40273, ← c40274] at e_638_0
  have e_638_1 := arithEq_of_rows h (row := 638) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_1
  simp only [← c40275, ← c40276, k2, ← c40277] at e_638_1
  have hr := e_638_1
  simp only [e_638_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3666)) := by
  have c40278 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40279 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40280 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40281 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40282 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40283 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_2 := arithEq_of_rows h (row := 638) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_2
  simp only [← c40278, ← c40279, ← c40280] at e_638_2
  have e_638_3 := arithEq_of_rows h (row := 638) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_3
  simp only [← c40281, ← c40282, k2, ← c40283] at e_638_3
  have hr := e_638_3
  simp only [e_638_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3667)) := by
  have c40284 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40285 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40286 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40287 := (publicBatchWrapper89_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40288 := (publicBatchWrapper89_copies1259 a h).1
  have c40289 := (publicBatchWrapper89_copies1259 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_4 := arithEq_of_rows h (row := 638) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_4
  simp only [← c40284, ← c40285, ← c40286] at e_638_4
  have e_638_5 := arithEq_of_rows h (row := 638) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_5
  simp only [← c40287, ← c40288, k2, ← c40289] at e_638_5
  have hr := e_638_5
  simp only [e_638_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3668)) := by
  have c40290 := (publicBatchWrapper89_copies1259 a h).2.2.1
  have c40291 := (publicBatchWrapper89_copies1259 a h).2.2.2.1
  have c40292 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.1
  have c40293 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.1
  have c40294 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.1
  have c40295 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_6 := arithEq_of_rows h (row := 638) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_6
  simp only [← c40290, ← c40291, ← c40292] at e_638_6
  have e_638_7 := arithEq_of_rows h (row := 638) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_7
  simp only [← c40293, ← c40294, k2, ← c40295] at e_638_7
  have hr := e_638_7
  simp only [e_638_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3669)) := by
  have c40296 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.1
  have c40297 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.1
  have c40298 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40299 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40300 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40301 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_8 := arithEq_of_rows h (row := 638) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_8
  simp only [← c40296, ← c40297, ← c40298] at e_638_8
  have e_638_9 := arithEq_of_rows h (row := 638) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_9
  simp only [← c40299, ← c40300, k2, ← c40301] at e_638_9
  have hr := e_638_9
  simp only [e_638_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3670)) := by
  have c40302 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40303 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40304 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40305 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40306 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40307 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_10 := arithEq_of_rows h (row := 638) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_10
  simp only [← c40302, ← c40303, ← c40304] at e_638_10
  have e_638_11 := arithEq_of_rows h (row := 638) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_11
  simp only [← c40305, ← c40306, k2, ← c40307] at e_638_11
  have hr := e_638_11
  simp only [e_638_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3671)) := by
  have c40308 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40309 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40310 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40311 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40312 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40313 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_12 := arithEq_of_rows h (row := 638) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_12
  simp only [← c40308, ← c40309, ← c40310] at e_638_12
  have e_638_13 := arithEq_of_rows h (row := 638) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_13
  simp only [← c40311, ← c40312, k2, ← c40313] at e_638_13
  have hr := e_638_13
  simp only [e_638_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3672)) := by
  have c40314 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40315 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40316 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40317 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40318 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40319 := (publicBatchWrapper89_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_14 := arithEq_of_rows h (row := 638) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_14
  simp only [← c40314, ← c40315, ← c40316] at e_638_14
  have e_638_15 := arithEq_of_rows h (row := 638) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_15
  simp only [← c40317, ← c40318, k2, ← c40319] at e_638_15
  have hr := e_638_15
  simp only [e_638_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3673)) := by
  have c40320 := (publicBatchWrapper89_copies1260 a h).1
  have c40321 := (publicBatchWrapper89_copies1260 a h).2.1
  have c40322 := (publicBatchWrapper89_copies1260 a h).2.2.1
  have c40323 := (publicBatchWrapper89_copies1260 a h).2.2.2.1
  have c40324 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.1
  have c40325 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_16 := arithEq_of_rows h (row := 638) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_16
  simp only [← c40320, ← c40321, ← c40322] at e_638_16
  have e_638_17 := arithEq_of_rows h (row := 638) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_17
  simp only [← c40323, ← c40324, k2, ← c40325] at e_638_17
  have hr := e_638_17
  simp only [e_638_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 638 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3674)) := by
  have c40326 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.1
  have c40327 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.1
  have c40328 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.1
  have c40329 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.1
  have c40330 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40331 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_638_18 := arithEq_of_rows h (row := 638) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_18
  simp only [← c40326, ← c40327, ← c40328] at e_638_18
  have e_638_19 := arithEq_of_rows h (row := 638) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_19
  simp only [← c40329, ← c40330, k2, ← c40331] at e_638_19
  have hr := e_638_19
  simp only [e_638_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3675)) := by
  have c40332 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40333 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40334 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40335 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40336 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40337 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_0 := arithEq_of_rows h (row := 639) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_0
  simp only [← c40332, ← c40333, ← c40334] at e_639_0
  have e_639_1 := arithEq_of_rows h (row := 639) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_1
  simp only [← c40335, ← c40336, k2, ← c40337] at e_639_1
  have hr := e_639_1
  simp only [e_639_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3676)) := by
  have c40338 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40339 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40340 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40341 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40342 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40343 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_2 := arithEq_of_rows h (row := 639) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_2
  simp only [← c40338, ← c40339, ← c40340] at e_639_2
  have e_639_3 := arithEq_of_rows h (row := 639) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_3
  simp only [← c40341, ← c40342, k2, ← c40343] at e_639_3
  have hr := e_639_3
  simp only [e_639_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3677)) := by
  have c40344 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40345 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40346 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40347 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40348 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40349 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_4 := arithEq_of_rows h (row := 639) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_4
  simp only [← c40344, ← c40345, ← c40346] at e_639_4
  have e_639_5 := arithEq_of_rows h (row := 639) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_5
  simp only [← c40347, ← c40348, k2, ← c40349] at e_639_5
  have hr := e_639_5
  simp only [e_639_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3678)) := by
  have c40350 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40351 := (publicBatchWrapper89_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40352 := (publicBatchWrapper89_copies1261 a h).1
  have c40353 := (publicBatchWrapper89_copies1261 a h).2.1
  have c40354 := (publicBatchWrapper89_copies1261 a h).2.2.1
  have c40355 := (publicBatchWrapper89_copies1261 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_6 := arithEq_of_rows h (row := 639) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_6
  simp only [← c40350, ← c40351, ← c40352] at e_639_6
  have e_639_7 := arithEq_of_rows h (row := 639) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_7
  simp only [← c40353, ← c40354, k2, ← c40355] at e_639_7
  have hr := e_639_7
  simp only [e_639_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3679)) := by
  have c40356 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.1
  have c40357 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.1
  have c40358 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.1
  have c40359 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.1
  have c40360 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.1
  have c40361 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_8 := arithEq_of_rows h (row := 639) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_8
  simp only [← c40356, ← c40357, ← c40358] at e_639_8
  have e_639_9 := arithEq_of_rows h (row := 639) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_9
  simp only [← c40359, ← c40360, k2, ← c40361] at e_639_9
  have hr := e_639_9
  simp only [e_639_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3680)) := by
  have c40362 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40363 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40364 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40365 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40366 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40367 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_10 := arithEq_of_rows h (row := 639) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_10
  simp only [← c40362, ← c40363, ← c40364] at e_639_10
  have e_639_11 := arithEq_of_rows h (row := 639) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_11
  simp only [← c40365, ← c40366, k2, ← c40367] at e_639_11
  have hr := e_639_11
  simp only [e_639_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3681)) := by
  have c40368 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40369 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40370 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40371 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40372 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40373 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_12 := arithEq_of_rows h (row := 639) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_12
  simp only [← c40368, ← c40369, ← c40370] at e_639_12
  have e_639_13 := arithEq_of_rows h (row := 639) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_13
  simp only [← c40371, ← c40372, k2, ← c40373] at e_639_13
  have hr := e_639_13
  simp only [e_639_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3682)) := by
  have c40374 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40375 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40376 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40377 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40378 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40379 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_14 := arithEq_of_rows h (row := 639) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_14
  simp only [← c40374, ← c40375, ← c40376] at e_639_14
  have e_639_15 := arithEq_of_rows h (row := 639) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_15
  simp only [← c40377, ← c40378, k2, ← c40379] at e_639_15
  have hr := e_639_15
  simp only [e_639_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3683)) := by
  have c40380 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40381 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40382 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40383 := (publicBatchWrapper89_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40384 := (publicBatchWrapper89_copies1262 a h).1
  have c40385 := (publicBatchWrapper89_copies1262 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_16 := arithEq_of_rows h (row := 639) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_16
  simp only [← c40380, ← c40381, ← c40382] at e_639_16
  have e_639_17 := arithEq_of_rows h (row := 639) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_17
  simp only [← c40383, ← c40384, k2, ← c40385] at e_639_17
  have hr := e_639_17
  simp only [e_639_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 639 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3684)) := by
  have c40386 := (publicBatchWrapper89_copies1262 a h).2.2.1
  have c40387 := (publicBatchWrapper89_copies1262 a h).2.2.2.1
  have c40388 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.1
  have c40389 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.1
  have c40390 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.1
  have c40391 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_639_18 := arithEq_of_rows h (row := 639) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_18
  simp only [← c40386, ← c40387, ← c40388] at e_639_18
  have e_639_19 := arithEq_of_rows h (row := 639) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_19
  simp only [← c40389, ← c40390, k2, ← c40391] at e_639_19
  have hr := e_639_19
  simp only [e_639_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3685)) := by
  have c40392 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.1
  have c40393 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.1
  have c40394 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40395 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40396 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40397 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_0 := arithEq_of_rows h (row := 640) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_0
  simp only [← c40392, ← c40393, ← c40394] at e_640_0
  have e_640_1 := arithEq_of_rows h (row := 640) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_1
  simp only [← c40395, ← c40396, k2, ← c40397] at e_640_1
  have hr := e_640_1
  simp only [e_640_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3686)) := by
  have c40398 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40399 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40400 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40401 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40402 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40403 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_2 := arithEq_of_rows h (row := 640) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_2
  simp only [← c40398, ← c40399, ← c40400] at e_640_2
  have e_640_3 := arithEq_of_rows h (row := 640) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_3
  simp only [← c40401, ← c40402, k2, ← c40403] at e_640_3
  have hr := e_640_3
  simp only [e_640_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3687)) := by
  have c40404 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40405 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40406 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40407 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40408 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40409 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_4 := arithEq_of_rows h (row := 640) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_4
  simp only [← c40404, ← c40405, ← c40406] at e_640_4
  have e_640_5 := arithEq_of_rows h (row := 640) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_5
  simp only [← c40407, ← c40408, k2, ← c40409] at e_640_5
  have hr := e_640_5
  simp only [e_640_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3688)) := by
  have c40410 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40411 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40412 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40413 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40414 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40415 := (publicBatchWrapper89_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_6 := arithEq_of_rows h (row := 640) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_6
  simp only [← c40410, ← c40411, ← c40412] at e_640_6
  have e_640_7 := arithEq_of_rows h (row := 640) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_7
  simp only [← c40413, ← c40414, k2, ← c40415] at e_640_7
  have hr := e_640_7
  simp only [e_640_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3689)) := by
  have c40416 := (publicBatchWrapper89_copies1263 a h).1
  have c40417 := (publicBatchWrapper89_copies1263 a h).2.1
  have c40418 := (publicBatchWrapper89_copies1263 a h).2.2.1
  have c40419 := (publicBatchWrapper89_copies1263 a h).2.2.2.1
  have c40420 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.1
  have c40421 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_8 := arithEq_of_rows h (row := 640) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_8
  simp only [← c40416, ← c40417, ← c40418] at e_640_8
  have e_640_9 := arithEq_of_rows h (row := 640) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_9
  simp only [← c40419, ← c40420, k2, ← c40421] at e_640_9
  have hr := e_640_9
  simp only [e_640_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3690)) := by
  have c40422 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.1
  have c40423 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.1
  have c40424 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.1
  have c40425 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.1
  have c40426 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40427 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_10 := arithEq_of_rows h (row := 640) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_10
  simp only [← c40422, ← c40423, ← c40424] at e_640_10
  have e_640_11 := arithEq_of_rows h (row := 640) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_11
  simp only [← c40425, ← c40426, k2, ← c40427] at e_640_11
  have hr := e_640_11
  simp only [e_640_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3691)) := by
  have c40428 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40429 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40430 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40431 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40432 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40433 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_12 := arithEq_of_rows h (row := 640) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_12
  simp only [← c40428, ← c40429, ← c40430] at e_640_12
  have e_640_13 := arithEq_of_rows h (row := 640) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_13
  simp only [← c40431, ← c40432, k2, ← c40433] at e_640_13
  have hr := e_640_13
  simp only [e_640_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3692)) := by
  have c40434 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40435 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40436 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40437 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40438 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40439 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_14 := arithEq_of_rows h (row := 640) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_14
  simp only [← c40434, ← c40435, ← c40436] at e_640_14
  have e_640_15 := arithEq_of_rows h (row := 640) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_15
  simp only [← c40437, ← c40438, k2, ← c40439] at e_640_15
  have hr := e_640_15
  simp only [e_640_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3693)) := by
  have c40440 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40441 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40442 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40443 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40444 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40445 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_16 := arithEq_of_rows h (row := 640) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_16
  simp only [← c40440, ← c40441, ← c40442] at e_640_16
  have e_640_17 := arithEq_of_rows h (row := 640) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_17
  simp only [← c40443, ← c40444, k2, ← c40445] at e_640_17
  have hr := e_640_17
  simp only [e_640_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 640 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3694)) := by
  have c40446 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40447 := (publicBatchWrapper89_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40448 := (publicBatchWrapper89_copies1264 a h).1
  have c40449 := (publicBatchWrapper89_copies1264 a h).2.1
  have c40450 := (publicBatchWrapper89_copies1264 a h).2.2.1
  have c40451 := (publicBatchWrapper89_copies1264 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_640_18 := arithEq_of_rows h (row := 640) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_18
  simp only [← c40446, ← c40447, ← c40448] at e_640_18
  have e_640_19 := arithEq_of_rows h (row := 640) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_19
  simp only [← c40449, ← c40450, k2, ← c40451] at e_640_19
  have hr := e_640_19
  simp only [e_640_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3695)) := by
  have c40452 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.1
  have c40453 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.1
  have c40454 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.1
  have c40455 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.1
  have c40456 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.1
  have c40457 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_0 := arithEq_of_rows h (row := 641) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_0
  simp only [← c40452, ← c40453, ← c40454] at e_641_0
  have e_641_1 := arithEq_of_rows h (row := 641) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_1
  simp only [← c40455, ← c40456, k2, ← c40457] at e_641_1
  have hr := e_641_1
  simp only [e_641_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3696)) := by
  have c40458 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40459 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40460 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40461 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40462 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40463 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_2 := arithEq_of_rows h (row := 641) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_2
  simp only [← c40458, ← c40459, ← c40460] at e_641_2
  have e_641_3 := arithEq_of_rows h (row := 641) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_3
  simp only [← c40461, ← c40462, k2, ← c40463] at e_641_3
  have hr := e_641_3
  simp only [e_641_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3697)) := by
  have c40464 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40465 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40466 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40467 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40468 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40469 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_4 := arithEq_of_rows h (row := 641) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_4
  simp only [← c40464, ← c40465, ← c40466] at e_641_4
  have e_641_5 := arithEq_of_rows h (row := 641) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_5
  simp only [← c40467, ← c40468, k2, ← c40469] at e_641_5
  have hr := e_641_5
  simp only [e_641_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3698)) := by
  have c40470 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40471 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40472 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40473 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40474 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40475 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_6 := arithEq_of_rows h (row := 641) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_6
  simp only [← c40470, ← c40471, ← c40472] at e_641_6
  have e_641_7 := arithEq_of_rows h (row := 641) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_7
  simp only [← c40473, ← c40474, k2, ← c40475] at e_641_7
  have hr := e_641_7
  simp only [e_641_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3699)) := by
  have c40476 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40477 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40478 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40479 := (publicBatchWrapper89_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40480 := (publicBatchWrapper89_copies1265 a h).1
  have c40481 := (publicBatchWrapper89_copies1265 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_8 := arithEq_of_rows h (row := 641) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_8
  simp only [← c40476, ← c40477, ← c40478] at e_641_8
  have e_641_9 := arithEq_of_rows h (row := 641) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_9
  simp only [← c40479, ← c40480, k2, ← c40481] at e_641_9
  have hr := e_641_9
  simp only [e_641_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3700)) := by
  have c40482 := (publicBatchWrapper89_copies1265 a h).2.2.1
  have c40483 := (publicBatchWrapper89_copies1265 a h).2.2.2.1
  have c40484 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.1
  have c40485 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.1
  have c40486 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.1
  have c40487 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_10 := arithEq_of_rows h (row := 641) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_10
  simp only [← c40482, ← c40483, ← c40484] at e_641_10
  have e_641_11 := arithEq_of_rows h (row := 641) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_11
  simp only [← c40485, ← c40486, k2, ← c40487] at e_641_11
  have hr := e_641_11
  simp only [e_641_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3701)) := by
  have c40488 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.1
  have c40489 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.1
  have c40490 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40491 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40492 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40493 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_12 := arithEq_of_rows h (row := 641) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_12
  simp only [← c40488, ← c40489, ← c40490] at e_641_12
  have e_641_13 := arithEq_of_rows h (row := 641) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_13
  simp only [← c40491, ← c40492, k2, ← c40493] at e_641_13
  have hr := e_641_13
  simp only [e_641_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3702)) := by
  have c40494 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40495 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40496 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40497 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40498 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40499 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_14 := arithEq_of_rows h (row := 641) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_14
  simp only [← c40494, ← c40495, ← c40496] at e_641_14
  have e_641_15 := arithEq_of_rows h (row := 641) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_15
  simp only [← c40497, ← c40498, k2, ← c40499] at e_641_15
  have hr := e_641_15
  simp only [e_641_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f5999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3703)) := by
  have c40500 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40501 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40502 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40503 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40504 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40505 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_16 := arithEq_of_rows h (row := 641) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_16
  simp only [← c40500, ← c40501, ← c40502] at e_641_16
  have e_641_17 := arithEq_of_rows h (row := 641) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_17
  simp only [← c40503, ← c40504, k2, ← c40505] at e_641_17
  have hr := e_641_17
  simp only [e_641_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 641 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3704)) := by
  have c40506 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40507 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40508 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40509 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40510 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40511 := (publicBatchWrapper89_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_641_18 := arithEq_of_rows h (row := 641) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_18
  simp only [← c40506, ← c40507, ← c40508] at e_641_18
  have e_641_19 := arithEq_of_rows h (row := 641) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_19
  simp only [← c40509, ← c40510, k2, ← c40511] at e_641_19
  have hr := e_641_19
  simp only [e_641_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3705)) := by
  have c40512 := (publicBatchWrapper89_copies1266 a h).1
  have c40513 := (publicBatchWrapper89_copies1266 a h).2.1
  have c40514 := (publicBatchWrapper89_copies1266 a h).2.2.1
  have c40515 := (publicBatchWrapper89_copies1266 a h).2.2.2.1
  have c40516 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.1
  have c40517 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_0 := arithEq_of_rows h (row := 642) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_0
  simp only [← c40512, ← c40513, ← c40514] at e_642_0
  have e_642_1 := arithEq_of_rows h (row := 642) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_1
  simp only [← c40515, ← c40516, k2, ← c40517] at e_642_1
  have hr := e_642_1
  simp only [e_642_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3706)) := by
  have c40518 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.1
  have c40519 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.1
  have c40520 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.1
  have c40521 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.1
  have c40522 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40523 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_2 := arithEq_of_rows h (row := 642) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_2
  simp only [← c40518, ← c40519, ← c40520] at e_642_2
  have e_642_3 := arithEq_of_rows h (row := 642) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_3
  simp only [← c40521, ← c40522, k2, ← c40523] at e_642_3
  have hr := e_642_3
  simp only [e_642_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3707)) := by
  have c40524 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40525 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40526 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40527 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40528 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40529 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_4 := arithEq_of_rows h (row := 642) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_4
  simp only [← c40524, ← c40525, ← c40526] at e_642_4
  have e_642_5 := arithEq_of_rows h (row := 642) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_5
  simp only [← c40527, ← c40528, k2, ← c40529] at e_642_5
  have hr := e_642_5
  simp only [e_642_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3708)) := by
  have c40530 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40531 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40532 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40533 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40534 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40535 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_6 := arithEq_of_rows h (row := 642) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_6
  simp only [← c40530, ← c40531, ← c40532] at e_642_6
  have e_642_7 := arithEq_of_rows h (row := 642) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_7
  simp only [← c40533, ← c40534, k2, ← c40535] at e_642_7
  have hr := e_642_7
  simp only [e_642_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3709)) := by
  have c40536 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40537 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40538 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40539 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40540 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40541 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_8 := arithEq_of_rows h (row := 642) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_8
  simp only [← c40536, ← c40537, ← c40538] at e_642_8
  have e_642_9 := arithEq_of_rows h (row := 642) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_9
  simp only [← c40539, ← c40540, k2, ← c40541] at e_642_9
  have hr := e_642_9
  simp only [e_642_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3710)) := by
  have c40542 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40543 := (publicBatchWrapper89_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40544 := (publicBatchWrapper89_copies1267 a h).1
  have c40545 := (publicBatchWrapper89_copies1267 a h).2.1
  have c40546 := (publicBatchWrapper89_copies1267 a h).2.2.1
  have c40547 := (publicBatchWrapper89_copies1267 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_10 := arithEq_of_rows h (row := 642) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_10
  simp only [← c40542, ← c40543, ← c40544] at e_642_10
  have e_642_11 := arithEq_of_rows h (row := 642) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_11
  simp only [← c40545, ← c40546, k2, ← c40547] at e_642_11
  have hr := e_642_11
  simp only [e_642_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3711)) := by
  have c40548 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.1
  have c40549 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.1
  have c40550 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.1
  have c40551 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.1
  have c40552 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.1
  have c40553 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_12 := arithEq_of_rows h (row := 642) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_12
  simp only [← c40548, ← c40549, ← c40550] at e_642_12
  have e_642_13 := arithEq_of_rows h (row := 642) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_13
  simp only [← c40551, ← c40552, k2, ← c40553] at e_642_13
  have hr := e_642_13
  simp only [e_642_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3712)) := by
  have c40554 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40555 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40556 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40557 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40558 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40559 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_14 := arithEq_of_rows h (row := 642) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_14
  simp only [← c40554, ← c40555, ← c40556] at e_642_14
  have e_642_15 := arithEq_of_rows h (row := 642) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_15
  simp only [← c40557, ← c40558, k2, ← c40559] at e_642_15
  have hr := e_642_15
  simp only [e_642_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3713)) := by
  have c40560 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40561 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40562 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40563 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40564 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40565 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_16 := arithEq_of_rows h (row := 642) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_16
  simp only [← c40560, ← c40561, ← c40562] at e_642_16
  have e_642_17 := arithEq_of_rows h (row := 642) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_17
  simp only [← c40563, ← c40564, k2, ← c40565] at e_642_17
  have hr := e_642_17
  simp only [e_642_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 642 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3714)) := by
  have c40566 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40567 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40568 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40569 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40570 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40571 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_642_18 := arithEq_of_rows h (row := 642) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_18
  simp only [← c40566, ← c40567, ← c40568] at e_642_18
  have e_642_19 := arithEq_of_rows h (row := 642) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_19
  simp only [← c40569, ← c40570, k2, ← c40571] at e_642_19
  have hr := e_642_19
  simp only [e_642_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3715)) := by
  have c40572 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40573 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40574 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40575 := (publicBatchWrapper89_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40576 := (publicBatchWrapper89_copies1268 a h).1
  have c40577 := (publicBatchWrapper89_copies1268 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_0 := arithEq_of_rows h (row := 643) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_0
  simp only [← c40572, ← c40573, ← c40574] at e_643_0
  have e_643_1 := arithEq_of_rows h (row := 643) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_1
  simp only [← c40575, ← c40576, k2, ← c40577] at e_643_1
  have hr := e_643_1
  simp only [e_643_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3716)) := by
  have c40578 := (publicBatchWrapper89_copies1268 a h).2.2.1
  have c40579 := (publicBatchWrapper89_copies1268 a h).2.2.2.1
  have c40580 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.1
  have c40581 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.1
  have c40582 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.1
  have c40583 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_2 := arithEq_of_rows h (row := 643) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_2
  simp only [← c40578, ← c40579, ← c40580] at e_643_2
  have e_643_3 := arithEq_of_rows h (row := 643) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_3
  simp only [← c40581, ← c40582, k2, ← c40583] at e_643_3
  have hr := e_643_3
  simp only [e_643_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
