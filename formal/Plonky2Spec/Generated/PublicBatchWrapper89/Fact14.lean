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

theorem publicBatchWrapper89_f10559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9423)) := by
  have c67860 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67861 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67862 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67863 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67864 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67865 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_16
  simp only [← c67860, ← c67861, ← c67862] at e_1097_16
  have e_1097_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_17
  simp only [← c67863, ← c67864, k2, ← c67865] at e_1097_17
  have hr := e_1097_17
  simp only [e_1097_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9424)) := by
  have c67866 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67867 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67868 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67869 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67870 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67871 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_18
  simp only [← c67866, ← c67867, ← c67868] at e_1097_18
  have e_1097_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_19
  simp only [← c67869, ← c67870, k2, ← c67871] at e_1097_19
  have hr := e_1097_19
  simp only [e_1097_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9425)) := by
  have c67872 := (publicBatchWrapper89_copies2121 a h).1
  have c67873 := (publicBatchWrapper89_copies2121 a h).2.1
  have c67874 := (publicBatchWrapper89_copies2121 a h).2.2.1
  have c67875 := (publicBatchWrapper89_copies2121 a h).2.2.2.1
  have c67876 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.1
  have c67877 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_0
  simp only [← c67872, ← c67873, ← c67874] at e_1098_0
  have e_1098_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_1
  simp only [← c67875, ← c67876, k2, ← c67877] at e_1098_1
  have hr := e_1098_1
  simp only [e_1098_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9426)) := by
  have c67878 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.1
  have c67879 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.1
  have c67880 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.1
  have c67881 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.1
  have c67882 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67883 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_2
  simp only [← c67878, ← c67879, ← c67880] at e_1098_2
  have e_1098_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_3
  simp only [← c67881, ← c67882, k2, ← c67883] at e_1098_3
  have hr := e_1098_3
  simp only [e_1098_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9427)) := by
  have c67884 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67885 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67886 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67887 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67888 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67889 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_4
  simp only [← c67884, ← c67885, ← c67886] at e_1098_4
  have e_1098_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_5
  simp only [← c67887, ← c67888, k2, ← c67889] at e_1098_5
  have hr := e_1098_5
  simp only [e_1098_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9428)) := by
  have c67890 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67891 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67892 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67893 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67894 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67895 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_6
  simp only [← c67890, ← c67891, ← c67892] at e_1098_6
  have e_1098_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_7
  simp only [← c67893, ← c67894, k2, ← c67895] at e_1098_7
  have hr := e_1098_7
  simp only [e_1098_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9429)) := by
  have c67896 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67897 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67898 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67899 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67900 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67901 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_8
  simp only [← c67896, ← c67897, ← c67898] at e_1098_8
  have e_1098_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_9
  simp only [← c67899, ← c67900, k2, ← c67901] at e_1098_9
  have hr := e_1098_9
  simp only [e_1098_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9430)) := by
  have c67902 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67903 := (publicBatchWrapper89_copies2121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67904 := (publicBatchWrapper89_copies2122 a h).1
  have c67905 := (publicBatchWrapper89_copies2122 a h).2.1
  have c67906 := (publicBatchWrapper89_copies2122 a h).2.2.1
  have c67907 := (publicBatchWrapper89_copies2122 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_10
  simp only [← c67902, ← c67903, ← c67904] at e_1098_10
  have e_1098_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_11
  simp only [← c67905, ← c67906, k2, ← c67907] at e_1098_11
  have hr := e_1098_11
  simp only [e_1098_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9431)) := by
  have c67908 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.1
  have c67909 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.1
  have c67910 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.1
  have c67911 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.1
  have c67912 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.1
  have c67913 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_12
  simp only [← c67908, ← c67909, ← c67910] at e_1098_12
  have e_1098_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_13
  simp only [← c67911, ← c67912, k2, ← c67913] at e_1098_13
  have hr := e_1098_13
  simp only [e_1098_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9432)) := by
  have c67914 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67915 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67916 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67917 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67918 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67919 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_14
  simp only [← c67914, ← c67915, ← c67916] at e_1098_14
  have e_1098_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_15
  simp only [← c67917, ← c67918, k2, ← c67919] at e_1098_15
  have hr := e_1098_15
  simp only [e_1098_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9433)) := by
  have c67920 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67921 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67922 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67923 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67924 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67925 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_16
  simp only [← c67920, ← c67921, ← c67922] at e_1098_16
  have e_1098_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_17
  simp only [← c67923, ← c67924, k2, ← c67925] at e_1098_17
  have hr := e_1098_17
  simp only [e_1098_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1098 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9434)) := by
  have c67926 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67927 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67928 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67929 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67930 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67931 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1098_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_18
  simp only [← c67926, ← c67927, ← c67928] at e_1098_18
  have e_1098_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1098 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1098_19
  simp only [← c67929, ← c67930, k2, ← c67931] at e_1098_19
  have hr := e_1098_19
  simp only [e_1098_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9435)) := by
  have c67932 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67933 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67934 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67935 := (publicBatchWrapper89_copies2122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67936 := (publicBatchWrapper89_copies2123 a h).1
  have c67937 := (publicBatchWrapper89_copies2123 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_0
  simp only [← c67932, ← c67933, ← c67934] at e_1099_0
  have e_1099_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_1
  simp only [← c67935, ← c67936, k2, ← c67937] at e_1099_1
  have hr := e_1099_1
  simp only [e_1099_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9436)) := by
  have c67938 := (publicBatchWrapper89_copies2123 a h).2.2.1
  have c67939 := (publicBatchWrapper89_copies2123 a h).2.2.2.1
  have c67940 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.1
  have c67941 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.1
  have c67942 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.1
  have c67943 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_2
  simp only [← c67938, ← c67939, ← c67940] at e_1099_2
  have e_1099_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_3
  simp only [← c67941, ← c67942, k2, ← c67943] at e_1099_3
  have hr := e_1099_3
  simp only [e_1099_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9437)) := by
  have c67944 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.1
  have c67945 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.1
  have c67946 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67947 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67948 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67949 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_4
  simp only [← c67944, ← c67945, ← c67946] at e_1099_4
  have e_1099_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_5
  simp only [← c67947, ← c67948, k2, ← c67949] at e_1099_5
  have hr := e_1099_5
  simp only [e_1099_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9438)) := by
  have c67950 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67951 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67952 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67953 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67954 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67955 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_6
  simp only [← c67950, ← c67951, ← c67952] at e_1099_6
  have e_1099_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_7
  simp only [← c67953, ← c67954, k2, ← c67955] at e_1099_7
  have hr := e_1099_7
  simp only [e_1099_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9439)) := by
  have c67956 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67957 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67958 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67959 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67960 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67961 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_8
  simp only [← c67956, ← c67957, ← c67958] at e_1099_8
  have e_1099_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_9
  simp only [← c67959, ← c67960, k2, ← c67961] at e_1099_9
  have hr := e_1099_9
  simp only [e_1099_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9440)) := by
  have c67962 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67963 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67964 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67965 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67966 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67967 := (publicBatchWrapper89_copies2123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_10
  simp only [← c67962, ← c67963, ← c67964] at e_1099_10
  have e_1099_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_11
  simp only [← c67965, ← c67966, k2, ← c67967] at e_1099_11
  have hr := e_1099_11
  simp only [e_1099_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9441)) := by
  have c67968 := (publicBatchWrapper89_copies2124 a h).1
  have c67969 := (publicBatchWrapper89_copies2124 a h).2.1
  have c67970 := (publicBatchWrapper89_copies2124 a h).2.2.1
  have c67971 := (publicBatchWrapper89_copies2124 a h).2.2.2.1
  have c67972 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.1
  have c67973 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_12
  simp only [← c67968, ← c67969, ← c67970] at e_1099_12
  have e_1099_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_13
  simp only [← c67971, ← c67972, k2, ← c67973] at e_1099_13
  have hr := e_1099_13
  simp only [e_1099_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9442)) := by
  have c67974 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.1
  have c67975 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.1
  have c67976 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.1
  have c67977 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.1
  have c67978 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67979 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_14
  simp only [← c67974, ← c67975, ← c67976] at e_1099_14
  have e_1099_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_15
  simp only [← c67977, ← c67978, k2, ← c67979] at e_1099_15
  have hr := e_1099_15
  simp only [e_1099_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9443)) := by
  have c67980 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67981 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67982 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67983 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67984 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67985 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_16
  simp only [← c67980, ← c67981, ← c67982] at e_1099_16
  have e_1099_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_17
  simp only [← c67983, ← c67984, k2, ← c67985] at e_1099_17
  have hr := e_1099_17
  simp only [e_1099_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1099 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9444)) := by
  have c67986 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67987 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67988 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67989 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67990 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67991 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1099_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_18
  simp only [← c67986, ← c67987, ← c67988] at e_1099_18
  have e_1099_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1099 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1099_19
  simp only [← c67989, ← c67990, k2, ← c67991] at e_1099_19
  have hr := e_1099_19
  simp only [e_1099_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9445)) := by
  have c67992 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67993 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67994 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67995 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67996 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67997 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_0
  simp only [← c67992, ← c67993, ← c67994] at e_1100_0
  have e_1100_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_1
  simp only [← c67995, ← c67996, k2, ← c67997] at e_1100_1
  have hr := e_1100_1
  simp only [e_1100_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9446)) := by
  have c67998 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67999 := (publicBatchWrapper89_copies2124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68000 := (publicBatchWrapper89_copies2125 a h).1
  have c68001 := (publicBatchWrapper89_copies2125 a h).2.1
  have c68002 := (publicBatchWrapper89_copies2125 a h).2.2.1
  have c68003 := (publicBatchWrapper89_copies2125 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_2
  simp only [← c67998, ← c67999, ← c68000] at e_1100_2
  have e_1100_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_3
  simp only [← c68001, ← c68002, k2, ← c68003] at e_1100_3
  have hr := e_1100_3
  simp only [e_1100_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9447)) := by
  have c68004 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.1
  have c68005 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.1
  have c68006 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.1
  have c68007 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.1
  have c68008 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.1
  have c68009 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_4
  simp only [← c68004, ← c68005, ← c68006] at e_1100_4
  have e_1100_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_5
  simp only [← c68007, ← c68008, k2, ← c68009] at e_1100_5
  have hr := e_1100_5
  simp only [e_1100_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9448)) := by
  have c68010 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68011 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68012 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68013 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68014 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68015 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_6
  simp only [← c68010, ← c68011, ← c68012] at e_1100_6
  have e_1100_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_7
  simp only [← c68013, ← c68014, k2, ← c68015] at e_1100_7
  have hr := e_1100_7
  simp only [e_1100_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9449)) := by
  have c68016 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68017 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68018 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68019 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68020 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68021 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_8
  simp only [← c68016, ← c68017, ← c68018] at e_1100_8
  have e_1100_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_9
  simp only [← c68019, ← c68020, k2, ← c68021] at e_1100_9
  have hr := e_1100_9
  simp only [e_1100_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9450)) := by
  have c68022 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68023 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68024 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68025 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68026 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68027 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_10
  simp only [← c68022, ← c68023, ← c68024] at e_1100_10
  have e_1100_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_11
  simp only [← c68025, ← c68026, k2, ← c68027] at e_1100_11
  have hr := e_1100_11
  simp only [e_1100_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9451)) := by
  have c68028 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68029 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68030 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68031 := (publicBatchWrapper89_copies2125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68032 := (publicBatchWrapper89_copies2126 a h).1
  have c68033 := (publicBatchWrapper89_copies2126 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_12
  simp only [← c68028, ← c68029, ← c68030] at e_1100_12
  have e_1100_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_13
  simp only [← c68031, ← c68032, k2, ← c68033] at e_1100_13
  have hr := e_1100_13
  simp only [e_1100_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9452)) := by
  have c68034 := (publicBatchWrapper89_copies2126 a h).2.2.1
  have c68035 := (publicBatchWrapper89_copies2126 a h).2.2.2.1
  have c68036 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.1
  have c68037 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.1
  have c68038 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.1
  have c68039 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_14
  simp only [← c68034, ← c68035, ← c68036] at e_1100_14
  have e_1100_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_15
  simp only [← c68037, ← c68038, k2, ← c68039] at e_1100_15
  have hr := e_1100_15
  simp only [e_1100_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9453)) := by
  have c68040 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.1
  have c68041 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.1
  have c68042 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68043 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68044 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68045 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_16
  simp only [← c68040, ← c68041, ← c68042] at e_1100_16
  have e_1100_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_17
  simp only [← c68043, ← c68044, k2, ← c68045] at e_1100_17
  have hr := e_1100_17
  simp only [e_1100_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1100 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9454)) := by
  have c68046 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68047 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68048 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68049 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68050 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68051 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1100_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_18
  simp only [← c68046, ← c68047, ← c68048] at e_1100_18
  have e_1100_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1100 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1100_19
  simp only [← c68049, ← c68050, k2, ← c68051] at e_1100_19
  have hr := e_1100_19
  simp only [e_1100_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9455)) := by
  have c68052 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68053 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68054 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68055 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68056 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68057 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_0
  simp only [← c68052, ← c68053, ← c68054] at e_1101_0
  have e_1101_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_1
  simp only [← c68055, ← c68056, k2, ← c68057] at e_1101_1
  have hr := e_1101_1
  simp only [e_1101_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9456)) := by
  have c68058 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68059 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68060 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68061 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68062 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68063 := (publicBatchWrapper89_copies2126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_2
  simp only [← c68058, ← c68059, ← c68060] at e_1101_2
  have e_1101_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_3
  simp only [← c68061, ← c68062, k2, ← c68063] at e_1101_3
  have hr := e_1101_3
  simp only [e_1101_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9457)) := by
  have c68064 := (publicBatchWrapper89_copies2127 a h).1
  have c68065 := (publicBatchWrapper89_copies2127 a h).2.1
  have c68066 := (publicBatchWrapper89_copies2127 a h).2.2.1
  have c68067 := (publicBatchWrapper89_copies2127 a h).2.2.2.1
  have c68068 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.1
  have c68069 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_4
  simp only [← c68064, ← c68065, ← c68066] at e_1101_4
  have e_1101_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_5
  simp only [← c68067, ← c68068, k2, ← c68069] at e_1101_5
  have hr := e_1101_5
  simp only [e_1101_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9458)) := by
  have c68070 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.1
  have c68071 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.1
  have c68072 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.1
  have c68073 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.1
  have c68074 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68075 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_6
  simp only [← c68070, ← c68071, ← c68072] at e_1101_6
  have e_1101_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_7
  simp only [← c68073, ← c68074, k2, ← c68075] at e_1101_7
  have hr := e_1101_7
  simp only [e_1101_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9459)) := by
  have c68076 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68077 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68078 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68079 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68080 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68081 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_8
  simp only [← c68076, ← c68077, ← c68078] at e_1101_8
  have e_1101_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_9
  simp only [← c68079, ← c68080, k2, ← c68081] at e_1101_9
  have hr := e_1101_9
  simp only [e_1101_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9460)) := by
  have c68082 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68083 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68084 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68085 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68086 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68087 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_10
  simp only [← c68082, ← c68083, ← c68084] at e_1101_10
  have e_1101_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_11
  simp only [← c68085, ← c68086, k2, ← c68087] at e_1101_11
  have hr := e_1101_11
  simp only [e_1101_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9461)) := by
  have c68088 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68089 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68090 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68091 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68092 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68093 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_12
  simp only [← c68088, ← c68089, ← c68090] at e_1101_12
  have e_1101_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_13
  simp only [← c68091, ← c68092, k2, ← c68093] at e_1101_13
  have hr := e_1101_13
  simp only [e_1101_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9462)) := by
  have c68094 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68095 := (publicBatchWrapper89_copies2127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68096 := (publicBatchWrapper89_copies2128 a h).1
  have c68097 := (publicBatchWrapper89_copies2128 a h).2.1
  have c68098 := (publicBatchWrapper89_copies2128 a h).2.2.1
  have c68099 := (publicBatchWrapper89_copies2128 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_14
  simp only [← c68094, ← c68095, ← c68096] at e_1101_14
  have e_1101_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_15
  simp only [← c68097, ← c68098, k2, ← c68099] at e_1101_15
  have hr := e_1101_15
  simp only [e_1101_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9463)) := by
  have c68100 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.1
  have c68101 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.1
  have c68102 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.1
  have c68103 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.1
  have c68104 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.1
  have c68105 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_16
  simp only [← c68100, ← c68101, ← c68102] at e_1101_16
  have e_1101_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_17
  simp only [← c68103, ← c68104, k2, ← c68105] at e_1101_17
  have hr := e_1101_17
  simp only [e_1101_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1101 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9464)) := by
  have c68106 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68107 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68108 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68109 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68110 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68111 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1101_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_18
  simp only [← c68106, ← c68107, ← c68108] at e_1101_18
  have e_1101_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1101 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1101_19
  simp only [← c68109, ← c68110, k2, ← c68111] at e_1101_19
  have hr := e_1101_19
  simp only [e_1101_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9465)) := by
  have c68112 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68113 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68114 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68115 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68116 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68117 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_0
  simp only [← c68112, ← c68113, ← c68114] at e_1102_0
  have e_1102_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_1
  simp only [← c68115, ← c68116, k2, ← c68117] at e_1102_1
  have hr := e_1102_1
  simp only [e_1102_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9466)) := by
  have c68118 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68119 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68120 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68121 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68122 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68123 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_2
  simp only [← c68118, ← c68119, ← c68120] at e_1102_2
  have e_1102_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_3
  simp only [← c68121, ← c68122, k2, ← c68123] at e_1102_3
  have hr := e_1102_3
  simp only [e_1102_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9467)) := by
  have c68124 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68125 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68126 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68127 := (publicBatchWrapper89_copies2128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68128 := (publicBatchWrapper89_copies2129 a h).1
  have c68129 := (publicBatchWrapper89_copies2129 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_4
  simp only [← c68124, ← c68125, ← c68126] at e_1102_4
  have e_1102_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_5
  simp only [← c68127, ← c68128, k2, ← c68129] at e_1102_5
  have hr := e_1102_5
  simp only [e_1102_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9468)) := by
  have c68130 := (publicBatchWrapper89_copies2129 a h).2.2.1
  have c68131 := (publicBatchWrapper89_copies2129 a h).2.2.2.1
  have c68132 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.1
  have c68133 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.1
  have c68134 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.1
  have c68135 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_6
  simp only [← c68130, ← c68131, ← c68132] at e_1102_6
  have e_1102_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_7
  simp only [← c68133, ← c68134, k2, ← c68135] at e_1102_7
  have hr := e_1102_7
  simp only [e_1102_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9469)) := by
  have c68136 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.1
  have c68137 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.1
  have c68138 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68139 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68140 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68141 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_8
  simp only [← c68136, ← c68137, ← c68138] at e_1102_8
  have e_1102_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_9
  simp only [← c68139, ← c68140, k2, ← c68141] at e_1102_9
  have hr := e_1102_9
  simp only [e_1102_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9470)) := by
  have c68142 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68143 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68144 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68145 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68146 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68147 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_10
  simp only [← c68142, ← c68143, ← c68144] at e_1102_10
  have e_1102_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_11
  simp only [← c68145, ← c68146, k2, ← c68147] at e_1102_11
  have hr := e_1102_11
  simp only [e_1102_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9471)) := by
  have c68148 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68149 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68150 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68151 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68152 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68153 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_12
  simp only [← c68148, ← c68149, ← c68150] at e_1102_12
  have e_1102_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_13
  simp only [← c68151, ← c68152, k2, ← c68153] at e_1102_13
  have hr := e_1102_13
  simp only [e_1102_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9472)) := by
  have c68154 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68155 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68156 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68157 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68158 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68159 := (publicBatchWrapper89_copies2129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_14
  simp only [← c68154, ← c68155, ← c68156] at e_1102_14
  have e_1102_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_15
  simp only [← c68157, ← c68158, k2, ← c68159] at e_1102_15
  have hr := e_1102_15
  simp only [e_1102_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9473)) := by
  have c68160 := (publicBatchWrapper89_copies2130 a h).1
  have c68161 := (publicBatchWrapper89_copies2130 a h).2.1
  have c68162 := (publicBatchWrapper89_copies2130 a h).2.2.1
  have c68163 := (publicBatchWrapper89_copies2130 a h).2.2.2.1
  have c68164 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.1
  have c68165 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_16
  simp only [← c68160, ← c68161, ← c68162] at e_1102_16
  have e_1102_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_17
  simp only [← c68163, ← c68164, k2, ← c68165] at e_1102_17
  have hr := e_1102_17
  simp only [e_1102_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1102 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9474)) := by
  have c68166 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.1
  have c68167 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.1
  have c68168 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.1
  have c68169 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.1
  have c68170 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68171 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1102_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_18
  simp only [← c68166, ← c68167, ← c68168] at e_1102_18
  have e_1102_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1102 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1102_19
  simp only [← c68169, ← c68170, k2, ← c68171] at e_1102_19
  have hr := e_1102_19
  simp only [e_1102_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9475)) := by
  have c68172 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68173 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68174 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68175 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68176 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68177 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_0
  simp only [← c68172, ← c68173, ← c68174] at e_1103_0
  have e_1103_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_1
  simp only [← c68175, ← c68176, k2, ← c68177] at e_1103_1
  have hr := e_1103_1
  simp only [e_1103_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9476)) := by
  have c68178 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68179 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68180 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68181 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68182 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68183 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_2
  simp only [← c68178, ← c68179, ← c68180] at e_1103_2
  have e_1103_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_3
  simp only [← c68181, ← c68182, k2, ← c68183] at e_1103_3
  have hr := e_1103_3
  simp only [e_1103_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9477)) := by
  have c68184 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68185 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68186 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68187 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68188 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68189 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_4
  simp only [← c68184, ← c68185, ← c68186] at e_1103_4
  have e_1103_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_5
  simp only [← c68187, ← c68188, k2, ← c68189] at e_1103_5
  have hr := e_1103_5
  simp only [e_1103_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9478)) := by
  have c68190 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68191 := (publicBatchWrapper89_copies2130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68192 := (publicBatchWrapper89_copies2131 a h).1
  have c68193 := (publicBatchWrapper89_copies2131 a h).2.1
  have c68194 := (publicBatchWrapper89_copies2131 a h).2.2.1
  have c68195 := (publicBatchWrapper89_copies2131 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_6
  simp only [← c68190, ← c68191, ← c68192] at e_1103_6
  have e_1103_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_7
  simp only [← c68193, ← c68194, k2, ← c68195] at e_1103_7
  have hr := e_1103_7
  simp only [e_1103_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9479)) := by
  have c68196 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.1
  have c68197 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.1
  have c68198 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.1
  have c68199 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.1
  have c68200 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.1
  have c68201 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_8
  simp only [← c68196, ← c68197, ← c68198] at e_1103_8
  have e_1103_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_9
  simp only [← c68199, ← c68200, k2, ← c68201] at e_1103_9
  have hr := e_1103_9
  simp only [e_1103_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9480)) := by
  have c68202 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68203 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68204 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68205 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68206 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68207 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_10
  simp only [← c68202, ← c68203, ← c68204] at e_1103_10
  have e_1103_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_11
  simp only [← c68205, ← c68206, k2, ← c68207] at e_1103_11
  have hr := e_1103_11
  simp only [e_1103_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9481)) := by
  have c68208 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68209 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68210 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68211 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68212 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68213 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_12
  simp only [← c68208, ← c68209, ← c68210] at e_1103_12
  have e_1103_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_13
  simp only [← c68211, ← c68212, k2, ← c68213] at e_1103_13
  have hr := e_1103_13
  simp only [e_1103_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9482)) := by
  have c68214 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68215 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68216 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68217 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68218 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68219 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_14
  simp only [← c68214, ← c68215, ← c68216] at e_1103_14
  have e_1103_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_15
  simp only [← c68217, ← c68218, k2, ← c68219] at e_1103_15
  have hr := e_1103_15
  simp only [e_1103_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9483)) := by
  have c68220 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68221 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68222 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68223 := (publicBatchWrapper89_copies2131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68224 := (publicBatchWrapper89_copies2132 a h).1
  have c68225 := (publicBatchWrapper89_copies2132 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_16
  simp only [← c68220, ← c68221, ← c68222] at e_1103_16
  have e_1103_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_17
  simp only [← c68223, ← c68224, k2, ← c68225] at e_1103_17
  have hr := e_1103_17
  simp only [e_1103_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1103 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9484)) := by
  have c68226 := (publicBatchWrapper89_copies2132 a h).2.2.1
  have c68227 := (publicBatchWrapper89_copies2132 a h).2.2.2.1
  have c68228 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.1
  have c68229 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.1
  have c68230 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.1
  have c68231 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1103_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_18
  simp only [← c68226, ← c68227, ← c68228] at e_1103_18
  have e_1103_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1103 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1103_19
  simp only [← c68229, ← c68230, k2, ← c68231] at e_1103_19
  have hr := e_1103_19
  simp only [e_1103_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9485)) := by
  have c68232 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.1
  have c68233 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.1
  have c68234 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68235 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68236 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68237 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_0
  simp only [← c68232, ← c68233, ← c68234] at e_1104_0
  have e_1104_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_1
  simp only [← c68235, ← c68236, k2, ← c68237] at e_1104_1
  have hr := e_1104_1
  simp only [e_1104_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9486)) := by
  have c68238 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68239 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68240 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68241 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68242 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68243 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_2
  simp only [← c68238, ← c68239, ← c68240] at e_1104_2
  have e_1104_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_3
  simp only [← c68241, ← c68242, k2, ← c68243] at e_1104_3
  have hr := e_1104_3
  simp only [e_1104_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9487)) := by
  have c68244 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68245 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68246 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68247 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68248 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68249 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_4
  simp only [← c68244, ← c68245, ← c68246] at e_1104_4
  have e_1104_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_5
  simp only [← c68247, ← c68248, k2, ← c68249] at e_1104_5
  have hr := e_1104_5
  simp only [e_1104_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9488)) := by
  have c68250 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68251 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68252 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68253 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68254 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68255 := (publicBatchWrapper89_copies2132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_6
  simp only [← c68250, ← c68251, ← c68252] at e_1104_6
  have e_1104_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_7
  simp only [← c68253, ← c68254, k2, ← c68255] at e_1104_7
  have hr := e_1104_7
  simp only [e_1104_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9489)) := by
  have c68256 := (publicBatchWrapper89_copies2133 a h).1
  have c68257 := (publicBatchWrapper89_copies2133 a h).2.1
  have c68258 := (publicBatchWrapper89_copies2133 a h).2.2.1
  have c68259 := (publicBatchWrapper89_copies2133 a h).2.2.2.1
  have c68260 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.1
  have c68261 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_8
  simp only [← c68256, ← c68257, ← c68258] at e_1104_8
  have e_1104_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_9
  simp only [← c68259, ← c68260, k2, ← c68261] at e_1104_9
  have hr := e_1104_9
  simp only [e_1104_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9490)) := by
  have c68262 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.1
  have c68263 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.1
  have c68264 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.1
  have c68265 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.1
  have c68266 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68267 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_10
  simp only [← c68262, ← c68263, ← c68264] at e_1104_10
  have e_1104_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_11
  simp only [← c68265, ← c68266, k2, ← c68267] at e_1104_11
  have hr := e_1104_11
  simp only [e_1104_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9491)) := by
  have c68268 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68269 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68270 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68271 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68272 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68273 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_12
  simp only [← c68268, ← c68269, ← c68270] at e_1104_12
  have e_1104_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_13
  simp only [← c68271, ← c68272, k2, ← c68273] at e_1104_13
  have hr := e_1104_13
  simp only [e_1104_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9492)) := by
  have c68274 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68275 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68276 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68277 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68278 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68279 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_14
  simp only [← c68274, ← c68275, ← c68276] at e_1104_14
  have e_1104_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_15
  simp only [← c68277, ← c68278, k2, ← c68279] at e_1104_15
  have hr := e_1104_15
  simp only [e_1104_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9493)) := by
  have c68280 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68281 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68282 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68283 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68284 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68285 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_16
  simp only [← c68280, ← c68281, ← c68282] at e_1104_16
  have e_1104_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_17
  simp only [← c68283, ← c68284, k2, ← c68285] at e_1104_17
  have hr := e_1104_17
  simp only [e_1104_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1104 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9494)) := by
  have c68286 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68287 := (publicBatchWrapper89_copies2133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68288 := (publicBatchWrapper89_copies2134 a h).1
  have c68289 := (publicBatchWrapper89_copies2134 a h).2.1
  have c68290 := (publicBatchWrapper89_copies2134 a h).2.2.1
  have c68291 := (publicBatchWrapper89_copies2134 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1104_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_18
  simp only [← c68286, ← c68287, ← c68288] at e_1104_18
  have e_1104_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1104 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1104_19
  simp only [← c68289, ← c68290, k2, ← c68291] at e_1104_19
  have hr := e_1104_19
  simp only [e_1104_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9495)) := by
  have c68292 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.1
  have c68293 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.1
  have c68294 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.1
  have c68295 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.1
  have c68296 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.1
  have c68297 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_0
  simp only [← c68292, ← c68293, ← c68294] at e_1105_0
  have e_1105_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_1
  simp only [← c68295, ← c68296, k2, ← c68297] at e_1105_1
  have hr := e_1105_1
  simp only [e_1105_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9496)) := by
  have c68298 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68299 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68300 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68301 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68302 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68303 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_2
  simp only [← c68298, ← c68299, ← c68300] at e_1105_2
  have e_1105_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_3
  simp only [← c68301, ← c68302, k2, ← c68303] at e_1105_3
  have hr := e_1105_3
  simp only [e_1105_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9497)) := by
  have c68304 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68305 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68306 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68307 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68308 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68309 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_4
  simp only [← c68304, ← c68305, ← c68306] at e_1105_4
  have e_1105_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_5
  simp only [← c68307, ← c68308, k2, ← c68309] at e_1105_5
  have hr := e_1105_5
  simp only [e_1105_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9498)) := by
  have c68310 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68311 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68312 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68313 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68314 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68315 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_6
  simp only [← c68310, ← c68311, ← c68312] at e_1105_6
  have e_1105_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_7
  simp only [← c68313, ← c68314, k2, ← c68315] at e_1105_7
  have hr := e_1105_7
  simp only [e_1105_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9499)) := by
  have c68316 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68317 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68318 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68319 := (publicBatchWrapper89_copies2134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68320 := (publicBatchWrapper89_copies2135 a h).1
  have c68321 := (publicBatchWrapper89_copies2135 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_8
  simp only [← c68316, ← c68317, ← c68318] at e_1105_8
  have e_1105_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_9
  simp only [← c68319, ← c68320, k2, ← c68321] at e_1105_9
  have hr := e_1105_9
  simp only [e_1105_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9500)) := by
  have c68322 := (publicBatchWrapper89_copies2135 a h).2.2.1
  have c68323 := (publicBatchWrapper89_copies2135 a h).2.2.2.1
  have c68324 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.1
  have c68325 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.1
  have c68326 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.1
  have c68327 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_10
  simp only [← c68322, ← c68323, ← c68324] at e_1105_10
  have e_1105_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_11
  simp only [← c68325, ← c68326, k2, ← c68327] at e_1105_11
  have hr := e_1105_11
  simp only [e_1105_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9501)) := by
  have c68328 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.1
  have c68329 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.1
  have c68330 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68331 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68332 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68333 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_12
  simp only [← c68328, ← c68329, ← c68330] at e_1105_12
  have e_1105_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_13
  simp only [← c68331, ← c68332, k2, ← c68333] at e_1105_13
  have hr := e_1105_13
  simp only [e_1105_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9502)) := by
  have c68334 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68335 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68336 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68337 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68338 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68339 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_14
  simp only [← c68334, ← c68335, ← c68336] at e_1105_14
  have e_1105_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_15
  simp only [← c68337, ← c68338, k2, ← c68339] at e_1105_15
  have hr := e_1105_15
  simp only [e_1105_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9503)) := by
  have c68340 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68341 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68342 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68343 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68344 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68345 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_16
  simp only [← c68340, ← c68341, ← c68342] at e_1105_16
  have e_1105_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_17
  simp only [← c68343, ← c68344, k2, ← c68345] at e_1105_17
  have hr := e_1105_17
  simp only [e_1105_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1105 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9504)) := by
  have c68346 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68347 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68348 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68349 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68350 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68351 := (publicBatchWrapper89_copies2135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1105_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_18
  simp only [← c68346, ← c68347, ← c68348] at e_1105_18
  have e_1105_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1105 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1105_19
  simp only [← c68349, ← c68350, k2, ← c68351] at e_1105_19
  have hr := e_1105_19
  simp only [e_1105_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9505)) := by
  have c68352 := (publicBatchWrapper89_copies2136 a h).1
  have c68353 := (publicBatchWrapper89_copies2136 a h).2.1
  have c68354 := (publicBatchWrapper89_copies2136 a h).2.2.1
  have c68355 := (publicBatchWrapper89_copies2136 a h).2.2.2.1
  have c68356 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.1
  have c68357 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_0
  simp only [← c68352, ← c68353, ← c68354] at e_1106_0
  have e_1106_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_1
  simp only [← c68355, ← c68356, k2, ← c68357] at e_1106_1
  have hr := e_1106_1
  simp only [e_1106_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9506)) := by
  have c68358 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.1
  have c68359 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.1
  have c68360 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.1
  have c68361 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.1
  have c68362 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68363 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_2
  simp only [← c68358, ← c68359, ← c68360] at e_1106_2
  have e_1106_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_3
  simp only [← c68361, ← c68362, k2, ← c68363] at e_1106_3
  have hr := e_1106_3
  simp only [e_1106_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9507)) := by
  have c68364 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68365 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68366 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68367 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68368 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68369 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_4
  simp only [← c68364, ← c68365, ← c68366] at e_1106_4
  have e_1106_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_5
  simp only [← c68367, ← c68368, k2, ← c68369] at e_1106_5
  have hr := e_1106_5
  simp only [e_1106_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9508)) := by
  have c68370 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68371 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68372 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68373 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68374 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68375 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_6
  simp only [← c68370, ← c68371, ← c68372] at e_1106_6
  have e_1106_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_7
  simp only [← c68373, ← c68374, k2, ← c68375] at e_1106_7
  have hr := e_1106_7
  simp only [e_1106_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9509)) := by
  have c68376 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68377 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68378 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68379 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68380 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68381 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_8
  simp only [← c68376, ← c68377, ← c68378] at e_1106_8
  have e_1106_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_9
  simp only [← c68379, ← c68380, k2, ← c68381] at e_1106_9
  have hr := e_1106_9
  simp only [e_1106_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9510)) := by
  have c68382 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68383 := (publicBatchWrapper89_copies2136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68384 := (publicBatchWrapper89_copies2137 a h).1
  have c68385 := (publicBatchWrapper89_copies2137 a h).2.1
  have c68386 := (publicBatchWrapper89_copies2137 a h).2.2.1
  have c68387 := (publicBatchWrapper89_copies2137 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_10
  simp only [← c68382, ← c68383, ← c68384] at e_1106_10
  have e_1106_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_11
  simp only [← c68385, ← c68386, k2, ← c68387] at e_1106_11
  have hr := e_1106_11
  simp only [e_1106_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9511)) := by
  have c68388 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.1
  have c68389 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.1
  have c68390 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.1
  have c68391 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.1
  have c68392 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.1
  have c68393 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_12
  simp only [← c68388, ← c68389, ← c68390] at e_1106_12
  have e_1106_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_13
  simp only [← c68391, ← c68392, k2, ← c68393] at e_1106_13
  have hr := e_1106_13
  simp only [e_1106_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9512)) := by
  have c68394 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68395 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68396 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68397 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68398 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68399 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_14
  simp only [← c68394, ← c68395, ← c68396] at e_1106_14
  have e_1106_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_15
  simp only [← c68397, ← c68398, k2, ← c68399] at e_1106_15
  have hr := e_1106_15
  simp only [e_1106_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9513)) := by
  have c68400 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68401 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68402 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68403 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68404 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68405 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_16
  simp only [← c68400, ← c68401, ← c68402] at e_1106_16
  have e_1106_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_17
  simp only [← c68403, ← c68404, k2, ← c68405] at e_1106_17
  have hr := e_1106_17
  simp only [e_1106_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1106 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9514)) := by
  have c68406 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68407 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68408 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68409 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68410 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68411 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1106_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_18
  simp only [← c68406, ← c68407, ← c68408] at e_1106_18
  have e_1106_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1106 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1106_19
  simp only [← c68409, ← c68410, k2, ← c68411] at e_1106_19
  have hr := e_1106_19
  simp only [e_1106_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9515)) := by
  have c68412 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68413 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68414 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68415 := (publicBatchWrapper89_copies2137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68416 := (publicBatchWrapper89_copies2138 a h).1
  have c68417 := (publicBatchWrapper89_copies2138 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_0
  simp only [← c68412, ← c68413, ← c68414] at e_1107_0
  have e_1107_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_1
  simp only [← c68415, ← c68416, k2, ← c68417] at e_1107_1
  have hr := e_1107_1
  simp only [e_1107_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9516)) := by
  have c68418 := (publicBatchWrapper89_copies2138 a h).2.2.1
  have c68419 := (publicBatchWrapper89_copies2138 a h).2.2.2.1
  have c68420 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.1
  have c68421 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.1
  have c68422 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.1
  have c68423 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_2
  simp only [← c68418, ← c68419, ← c68420] at e_1107_2
  have e_1107_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_3
  simp only [← c68421, ← c68422, k2, ← c68423] at e_1107_3
  have hr := e_1107_3
  simp only [e_1107_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9517)) := by
  have c68424 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.1
  have c68425 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.1
  have c68426 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68427 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68428 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68429 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_4
  simp only [← c68424, ← c68425, ← c68426] at e_1107_4
  have e_1107_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_5
  simp only [← c68427, ← c68428, k2, ← c68429] at e_1107_5
  have hr := e_1107_5
  simp only [e_1107_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9518)) := by
  have c68430 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68431 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68432 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68433 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68434 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68435 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_6
  simp only [← c68430, ← c68431, ← c68432] at e_1107_6
  have e_1107_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_7
  simp only [← c68433, ← c68434, k2, ← c68435] at e_1107_7
  have hr := e_1107_7
  simp only [e_1107_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9519)) := by
  have c68436 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68437 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68438 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68439 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68440 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68441 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_8
  simp only [← c68436, ← c68437, ← c68438] at e_1107_8
  have e_1107_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_9
  simp only [← c68439, ← c68440, k2, ← c68441] at e_1107_9
  have hr := e_1107_9
  simp only [e_1107_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9520)) := by
  have c68442 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68443 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68444 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68445 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68446 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68447 := (publicBatchWrapper89_copies2138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_10
  simp only [← c68442, ← c68443, ← c68444] at e_1107_10
  have e_1107_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_11
  simp only [← c68445, ← c68446, k2, ← c68447] at e_1107_11
  have hr := e_1107_11
  simp only [e_1107_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9521)) := by
  have c68448 := (publicBatchWrapper89_copies2139 a h).1
  have c68449 := (publicBatchWrapper89_copies2139 a h).2.1
  have c68450 := (publicBatchWrapper89_copies2139 a h).2.2.1
  have c68451 := (publicBatchWrapper89_copies2139 a h).2.2.2.1
  have c68452 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.1
  have c68453 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_12
  simp only [← c68448, ← c68449, ← c68450] at e_1107_12
  have e_1107_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_13
  simp only [← c68451, ← c68452, k2, ← c68453] at e_1107_13
  have hr := e_1107_13
  simp only [e_1107_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9522)) := by
  have c68454 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.1
  have c68455 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.1
  have c68456 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.1
  have c68457 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.1
  have c68458 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68459 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_14
  simp only [← c68454, ← c68455, ← c68456] at e_1107_14
  have e_1107_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_15
  simp only [← c68457, ← c68458, k2, ← c68459] at e_1107_15
  have hr := e_1107_15
  simp only [e_1107_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9523)) := by
  have c68460 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68461 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68462 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68463 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68464 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68465 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_16
  simp only [← c68460, ← c68461, ← c68462] at e_1107_16
  have e_1107_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_17
  simp only [← c68463, ← c68464, k2, ← c68465] at e_1107_17
  have hr := e_1107_17
  simp only [e_1107_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1107 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9524)) := by
  have c68466 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68467 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68468 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68469 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68470 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68471 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1107_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_18
  simp only [← c68466, ← c68467, ← c68468] at e_1107_18
  have e_1107_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1107 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1107_19
  simp only [← c68469, ← c68470, k2, ← c68471] at e_1107_19
  have hr := e_1107_19
  simp only [e_1107_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9525)) := by
  have c68472 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68473 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68474 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68475 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68476 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68477 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_0
  simp only [← c68472, ← c68473, ← c68474] at e_1108_0
  have e_1108_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_1
  simp only [← c68475, ← c68476, k2, ← c68477] at e_1108_1
  have hr := e_1108_1
  simp only [e_1108_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9526)) := by
  have c68478 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68479 := (publicBatchWrapper89_copies2139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68480 := (publicBatchWrapper89_copies2140 a h).1
  have c68481 := (publicBatchWrapper89_copies2140 a h).2.1
  have c68482 := (publicBatchWrapper89_copies2140 a h).2.2.1
  have c68483 := (publicBatchWrapper89_copies2140 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_2
  simp only [← c68478, ← c68479, ← c68480] at e_1108_2
  have e_1108_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_3
  simp only [← c68481, ← c68482, k2, ← c68483] at e_1108_3
  have hr := e_1108_3
  simp only [e_1108_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9527)) := by
  have c68484 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.1
  have c68485 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.1
  have c68486 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.1
  have c68487 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.1
  have c68488 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.1
  have c68489 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_4
  simp only [← c68484, ← c68485, ← c68486] at e_1108_4
  have e_1108_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_5
  simp only [← c68487, ← c68488, k2, ← c68489] at e_1108_5
  have hr := e_1108_5
  simp only [e_1108_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9528)) := by
  have c68490 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68491 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68492 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68493 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68494 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68495 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_6
  simp only [← c68490, ← c68491, ← c68492] at e_1108_6
  have e_1108_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_7
  simp only [← c68493, ← c68494, k2, ← c68495] at e_1108_7
  have hr := e_1108_7
  simp only [e_1108_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9529)) := by
  have c68496 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68497 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68498 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68499 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68500 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68501 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_8
  simp only [← c68496, ← c68497, ← c68498] at e_1108_8
  have e_1108_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_9
  simp only [← c68499, ← c68500, k2, ← c68501] at e_1108_9
  have hr := e_1108_9
  simp only [e_1108_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9530)) := by
  have c68502 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68503 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68504 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68505 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68506 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68507 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_10
  simp only [← c68502, ← c68503, ← c68504] at e_1108_10
  have e_1108_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_11
  simp only [← c68505, ← c68506, k2, ← c68507] at e_1108_11
  have hr := e_1108_11
  simp only [e_1108_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9531)) := by
  have c68508 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68509 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68510 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68511 := (publicBatchWrapper89_copies2140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68512 := (publicBatchWrapper89_copies2141 a h).1
  have c68513 := (publicBatchWrapper89_copies2141 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_12
  simp only [← c68508, ← c68509, ← c68510] at e_1108_12
  have e_1108_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_13
  simp only [← c68511, ← c68512, k2, ← c68513] at e_1108_13
  have hr := e_1108_13
  simp only [e_1108_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9532)) := by
  have c68514 := (publicBatchWrapper89_copies2141 a h).2.2.1
  have c68515 := (publicBatchWrapper89_copies2141 a h).2.2.2.1
  have c68516 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.1
  have c68517 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.1
  have c68518 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.1
  have c68519 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_14
  simp only [← c68514, ← c68515, ← c68516] at e_1108_14
  have e_1108_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_15
  simp only [← c68517, ← c68518, k2, ← c68519] at e_1108_15
  have hr := e_1108_15
  simp only [e_1108_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9533)) := by
  have c68520 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.1
  have c68521 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.1
  have c68522 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68523 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68524 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68525 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_16
  simp only [← c68520, ← c68521, ← c68522] at e_1108_16
  have e_1108_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_17
  simp only [← c68523, ← c68524, k2, ← c68525] at e_1108_17
  have hr := e_1108_17
  simp only [e_1108_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1108 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9534)) := by
  have c68526 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68527 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68528 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68529 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68530 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68531 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1108_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_18
  simp only [← c68526, ← c68527, ← c68528] at e_1108_18
  have e_1108_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1108 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1108_19
  simp only [← c68529, ← c68530, k2, ← c68531] at e_1108_19
  have hr := e_1108_19
  simp only [e_1108_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9535)) := by
  have c68532 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68533 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68534 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68535 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68536 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68537 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_0
  simp only [← c68532, ← c68533, ← c68534] at e_1109_0
  have e_1109_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_1
  simp only [← c68535, ← c68536, k2, ← c68537] at e_1109_1
  have hr := e_1109_1
  simp only [e_1109_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9536)) := by
  have c68538 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68539 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68540 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68541 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68542 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68543 := (publicBatchWrapper89_copies2141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_2
  simp only [← c68538, ← c68539, ← c68540] at e_1109_2
  have e_1109_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_3
  simp only [← c68541, ← c68542, k2, ← c68543] at e_1109_3
  have hr := e_1109_3
  simp only [e_1109_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9537)) := by
  have c68544 := (publicBatchWrapper89_copies2142 a h).1
  have c68545 := (publicBatchWrapper89_copies2142 a h).2.1
  have c68546 := (publicBatchWrapper89_copies2142 a h).2.2.1
  have c68547 := (publicBatchWrapper89_copies2142 a h).2.2.2.1
  have c68548 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.1
  have c68549 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_4
  simp only [← c68544, ← c68545, ← c68546] at e_1109_4
  have e_1109_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_5
  simp only [← c68547, ← c68548, k2, ← c68549] at e_1109_5
  have hr := e_1109_5
  simp only [e_1109_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9538)) := by
  have c68550 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.1
  have c68551 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.1
  have c68552 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.1
  have c68553 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.1
  have c68554 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68555 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_6
  simp only [← c68550, ← c68551, ← c68552] at e_1109_6
  have e_1109_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_7
  simp only [← c68553, ← c68554, k2, ← c68555] at e_1109_7
  have hr := e_1109_7
  simp only [e_1109_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9539)) := by
  have c68556 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68557 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68558 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68559 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68560 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68561 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_8
  simp only [← c68556, ← c68557, ← c68558] at e_1109_8
  have e_1109_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_9
  simp only [← c68559, ← c68560, k2, ← c68561] at e_1109_9
  have hr := e_1109_9
  simp only [e_1109_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9540)) := by
  have c68562 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68563 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68564 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68565 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68566 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68567 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_10
  simp only [← c68562, ← c68563, ← c68564] at e_1109_10
  have e_1109_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_11
  simp only [← c68565, ← c68566, k2, ← c68567] at e_1109_11
  have hr := e_1109_11
  simp only [e_1109_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9541)) := by
  have c68568 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68569 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68570 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68571 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68572 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68573 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_12
  simp only [← c68568, ← c68569, ← c68570] at e_1109_12
  have e_1109_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_13
  simp only [← c68571, ← c68572, k2, ← c68573] at e_1109_13
  have hr := e_1109_13
  simp only [e_1109_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9542)) := by
  have c68574 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68575 := (publicBatchWrapper89_copies2142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68576 := (publicBatchWrapper89_copies2143 a h).1
  have c68577 := (publicBatchWrapper89_copies2143 a h).2.1
  have c68578 := (publicBatchWrapper89_copies2143 a h).2.2.1
  have c68579 := (publicBatchWrapper89_copies2143 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_14
  simp only [← c68574, ← c68575, ← c68576] at e_1109_14
  have e_1109_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_15
  simp only [← c68577, ← c68578, k2, ← c68579] at e_1109_15
  have hr := e_1109_15
  simp only [e_1109_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9543)) := by
  have c68580 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.1
  have c68581 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.1
  have c68582 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.1
  have c68583 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.1
  have c68584 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.1
  have c68585 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_16
  simp only [← c68580, ← c68581, ← c68582] at e_1109_16
  have e_1109_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_17
  simp only [← c68583, ← c68584, k2, ← c68585] at e_1109_17
  have hr := e_1109_17
  simp only [e_1109_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1109 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9544)) := by
  have c68586 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68587 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68588 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68589 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68590 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68591 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1109_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_18
  simp only [← c68586, ← c68587, ← c68588] at e_1109_18
  have e_1109_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1109 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1109_19
  simp only [← c68589, ← c68590, k2, ← c68591] at e_1109_19
  have hr := e_1109_19
  simp only [e_1109_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9545)) := by
  have c68592 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68593 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68594 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68595 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68596 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68597 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_0
  simp only [← c68592, ← c68593, ← c68594] at e_1110_0
  have e_1110_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_1
  simp only [← c68595, ← c68596, k2, ← c68597] at e_1110_1
  have hr := e_1110_1
  simp only [e_1110_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9546)) := by
  have c68598 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68599 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68600 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68601 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68602 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68603 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_2
  simp only [← c68598, ← c68599, ← c68600] at e_1110_2
  have e_1110_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_3
  simp only [← c68601, ← c68602, k2, ← c68603] at e_1110_3
  have hr := e_1110_3
  simp only [e_1110_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9547)) := by
  have c68604 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68605 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68606 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68607 := (publicBatchWrapper89_copies2143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68608 := (publicBatchWrapper89_copies2144 a h).1
  have c68609 := (publicBatchWrapper89_copies2144 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_4
  simp only [← c68604, ← c68605, ← c68606] at e_1110_4
  have e_1110_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_5
  simp only [← c68607, ← c68608, k2, ← c68609] at e_1110_5
  have hr := e_1110_5
  simp only [e_1110_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9548)) := by
  have c68610 := (publicBatchWrapper89_copies2144 a h).2.2.1
  have c68611 := (publicBatchWrapper89_copies2144 a h).2.2.2.1
  have c68612 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.1
  have c68613 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.1
  have c68614 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.1
  have c68615 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_6
  simp only [← c68610, ← c68611, ← c68612] at e_1110_6
  have e_1110_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_7
  simp only [← c68613, ← c68614, k2, ← c68615] at e_1110_7
  have hr := e_1110_7
  simp only [e_1110_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9549)) := by
  have c68616 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.1
  have c68617 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.1
  have c68618 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68619 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68620 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68621 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_8
  simp only [← c68616, ← c68617, ← c68618] at e_1110_8
  have e_1110_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_9
  simp only [← c68619, ← c68620, k2, ← c68621] at e_1110_9
  have hr := e_1110_9
  simp only [e_1110_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9550)) := by
  have c68622 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68623 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68624 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68625 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68626 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68627 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_10
  simp only [← c68622, ← c68623, ← c68624] at e_1110_10
  have e_1110_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_11
  simp only [← c68625, ← c68626, k2, ← c68627] at e_1110_11
  have hr := e_1110_11
  simp only [e_1110_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9551)) := by
  have c68628 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68629 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68630 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68631 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68632 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68633 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_12
  simp only [← c68628, ← c68629, ← c68630] at e_1110_12
  have e_1110_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_13
  simp only [← c68631, ← c68632, k2, ← c68633] at e_1110_13
  have hr := e_1110_13
  simp only [e_1110_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9552)) := by
  have c68634 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68635 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68636 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68637 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68638 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68639 := (publicBatchWrapper89_copies2144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_14
  simp only [← c68634, ← c68635, ← c68636] at e_1110_14
  have e_1110_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_15
  simp only [← c68637, ← c68638, k2, ← c68639] at e_1110_15
  have hr := e_1110_15
  simp only [e_1110_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9553)) := by
  have c68640 := (publicBatchWrapper89_copies2145 a h).1
  have c68641 := (publicBatchWrapper89_copies2145 a h).2.1
  have c68642 := (publicBatchWrapper89_copies2145 a h).2.2.1
  have c68643 := (publicBatchWrapper89_copies2145 a h).2.2.2.1
  have c68644 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.1
  have c68645 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_16
  simp only [← c68640, ← c68641, ← c68642] at e_1110_16
  have e_1110_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_17
  simp only [← c68643, ← c68644, k2, ← c68645] at e_1110_17
  have hr := e_1110_17
  simp only [e_1110_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1110 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9554)) := by
  have c68646 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.1
  have c68647 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.1
  have c68648 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.1
  have c68649 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.1
  have c68650 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68651 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1110_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_18
  simp only [← c68646, ← c68647, ← c68648] at e_1110_18
  have e_1110_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1110 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1110_19
  simp only [← c68649, ← c68650, k2, ← c68651] at e_1110_19
  have hr := e_1110_19
  simp only [e_1110_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9555)) := by
  have c68652 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68653 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68654 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68655 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68656 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68657 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_0
  simp only [← c68652, ← c68653, ← c68654] at e_1111_0
  have e_1111_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_1
  simp only [← c68655, ← c68656, k2, ← c68657] at e_1111_1
  have hr := e_1111_1
  simp only [e_1111_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9556)) := by
  have c68658 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68659 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68660 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68661 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68662 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68663 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_2
  simp only [← c68658, ← c68659, ← c68660] at e_1111_2
  have e_1111_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_3
  simp only [← c68661, ← c68662, k2, ← c68663] at e_1111_3
  have hr := e_1111_3
  simp only [e_1111_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9557)) := by
  have c68664 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68665 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68666 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68667 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68668 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68669 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_4
  simp only [← c68664, ← c68665, ← c68666] at e_1111_4
  have e_1111_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_5
  simp only [← c68667, ← c68668, k2, ← c68669] at e_1111_5
  have hr := e_1111_5
  simp only [e_1111_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9558)) := by
  have c68670 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68671 := (publicBatchWrapper89_copies2145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68672 := (publicBatchWrapper89_copies2146 a h).1
  have c68673 := (publicBatchWrapper89_copies2146 a h).2.1
  have c68674 := (publicBatchWrapper89_copies2146 a h).2.2.1
  have c68675 := (publicBatchWrapper89_copies2146 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_6
  simp only [← c68670, ← c68671, ← c68672] at e_1111_6
  have e_1111_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_7
  simp only [← c68673, ← c68674, k2, ← c68675] at e_1111_7
  have hr := e_1111_7
  simp only [e_1111_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9559)) := by
  have c68676 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.1
  have c68677 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.1
  have c68678 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.1
  have c68679 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.1
  have c68680 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.1
  have c68681 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_8
  simp only [← c68676, ← c68677, ← c68678] at e_1111_8
  have e_1111_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_9
  simp only [← c68679, ← c68680, k2, ← c68681] at e_1111_9
  have hr := e_1111_9
  simp only [e_1111_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9560)) := by
  have c68682 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68683 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68684 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68685 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68686 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68687 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_10
  simp only [← c68682, ← c68683, ← c68684] at e_1111_10
  have e_1111_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_11
  simp only [← c68685, ← c68686, k2, ← c68687] at e_1111_11
  have hr := e_1111_11
  simp only [e_1111_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9561)) := by
  have c68688 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68689 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68690 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68691 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68692 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68693 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_12
  simp only [← c68688, ← c68689, ← c68690] at e_1111_12
  have e_1111_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_13
  simp only [← c68691, ← c68692, k2, ← c68693] at e_1111_13
  have hr := e_1111_13
  simp only [e_1111_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9562)) := by
  have c68694 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68695 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68696 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68697 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68698 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68699 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_14
  simp only [← c68694, ← c68695, ← c68696] at e_1111_14
  have e_1111_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_15
  simp only [← c68697, ← c68698, k2, ← c68699] at e_1111_15
  have hr := e_1111_15
  simp only [e_1111_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9563)) := by
  have c68700 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68701 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68702 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68703 := (publicBatchWrapper89_copies2146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68704 := (publicBatchWrapper89_copies2147 a h).1
  have c68705 := (publicBatchWrapper89_copies2147 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_16
  simp only [← c68700, ← c68701, ← c68702] at e_1111_16
  have e_1111_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_17
  simp only [← c68703, ← c68704, k2, ← c68705] at e_1111_17
  have hr := e_1111_17
  simp only [e_1111_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1111 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9564)) := by
  have c68706 := (publicBatchWrapper89_copies2147 a h).2.2.1
  have c68707 := (publicBatchWrapper89_copies2147 a h).2.2.2.1
  have c68708 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.1
  have c68709 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.1
  have c68710 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.1
  have c68711 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1111_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_18
  simp only [← c68706, ← c68707, ← c68708] at e_1111_18
  have e_1111_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1111 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1111_19
  simp only [← c68709, ← c68710, k2, ← c68711] at e_1111_19
  have hr := e_1111_19
  simp only [e_1111_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9565)) := by
  have c68712 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.1
  have c68713 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.1
  have c68714 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68715 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68716 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68717 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_0
  simp only [← c68712, ← c68713, ← c68714] at e_1112_0
  have e_1112_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_1
  simp only [← c68715, ← c68716, k2, ← c68717] at e_1112_1
  have hr := e_1112_1
  simp only [e_1112_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9566)) := by
  have c68718 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68719 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68720 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68721 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68722 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68723 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_2
  simp only [← c68718, ← c68719, ← c68720] at e_1112_2
  have e_1112_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_3
  simp only [← c68721, ← c68722, k2, ← c68723] at e_1112_3
  have hr := e_1112_3
  simp only [e_1112_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9567)) := by
  have c68724 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68725 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68726 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68727 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68728 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68729 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_4
  simp only [← c68724, ← c68725, ← c68726] at e_1112_4
  have e_1112_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_5
  simp only [← c68727, ← c68728, k2, ← c68729] at e_1112_5
  have hr := e_1112_5
  simp only [e_1112_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9568)) := by
  have c68730 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68731 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68732 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68733 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68734 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68735 := (publicBatchWrapper89_copies2147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_6
  simp only [← c68730, ← c68731, ← c68732] at e_1112_6
  have e_1112_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_7
  simp only [← c68733, ← c68734, k2, ← c68735] at e_1112_7
  have hr := e_1112_7
  simp only [e_1112_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9569)) := by
  have c68736 := (publicBatchWrapper89_copies2148 a h).1
  have c68737 := (publicBatchWrapper89_copies2148 a h).2.1
  have c68738 := (publicBatchWrapper89_copies2148 a h).2.2.1
  have c68739 := (publicBatchWrapper89_copies2148 a h).2.2.2.1
  have c68740 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.1
  have c68741 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_8
  simp only [← c68736, ← c68737, ← c68738] at e_1112_8
  have e_1112_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_9
  simp only [← c68739, ← c68740, k2, ← c68741] at e_1112_9
  have hr := e_1112_9
  simp only [e_1112_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9610)) := by
  have c68742 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.1
  have c68743 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.1
  have c68744 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.1
  have c68745 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.1
  have c68746 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68747 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_10
  simp only [← c68742, ← c68743, ← c68744] at e_1112_10
  have e_1112_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_11
  simp only [← c68745, ← c68746, k2, ← c68747] at e_1112_11
  have hr := e_1112_11
  simp only [e_1112_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9611)) := by
  have c68748 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68749 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68750 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68751 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68752 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68753 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_12
  simp only [← c68748, ← c68749, ← c68750] at e_1112_12
  have e_1112_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_13
  simp only [← c68751, ← c68752, k2, ← c68753] at e_1112_13
  have hr := e_1112_13
  simp only [e_1112_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9612)) := by
  have c68754 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68755 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68756 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68757 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68758 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68759 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_14
  simp only [← c68754, ← c68755, ← c68756] at e_1112_14
  have e_1112_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_15
  simp only [← c68757, ← c68758, k2, ← c68759] at e_1112_15
  have hr := e_1112_15
  simp only [e_1112_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9613)) := by
  have c68760 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68761 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68762 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68763 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68764 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68765 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_16
  simp only [← c68760, ← c68761, ← c68762] at e_1112_16
  have e_1112_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_17
  simp only [← c68763, ← c68764, k2, ← c68765] at e_1112_17
  have hr := e_1112_17
  simp only [e_1112_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1112 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9614)) := by
  have c68766 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68767 := (publicBatchWrapper89_copies2148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68768 := (publicBatchWrapper89_copies2149 a h).1
  have c68769 := (publicBatchWrapper89_copies2149 a h).2.1
  have c68770 := (publicBatchWrapper89_copies2149 a h).2.2.1
  have c68771 := (publicBatchWrapper89_copies2149 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1112_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_18
  simp only [← c68766, ← c68767, ← c68768] at e_1112_18
  have e_1112_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1112 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1112_19
  simp only [← c68769, ← c68770, k2, ← c68771] at e_1112_19
  have hr := e_1112_19
  simp only [e_1112_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9615)) := by
  have c68772 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.1
  have c68773 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.1
  have c68774 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.1
  have c68775 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.1
  have c68776 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.1
  have c68777 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_0
  simp only [← c68772, ← c68773, ← c68774] at e_1113_0
  have e_1113_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_1
  simp only [← c68775, ← c68776, k2, ← c68777] at e_1113_1
  have hr := e_1113_1
  simp only [e_1113_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9616)) := by
  have c68778 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68779 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68780 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68781 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68782 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68783 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_2
  simp only [← c68778, ← c68779, ← c68780] at e_1113_2
  have e_1113_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_3
  simp only [← c68781, ← c68782, k2, ← c68783] at e_1113_3
  have hr := e_1113_3
  simp only [e_1113_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9617)) := by
  have c68784 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68785 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68786 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68787 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68788 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68789 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_4
  simp only [← c68784, ← c68785, ← c68786] at e_1113_4
  have e_1113_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_5
  simp only [← c68787, ← c68788, k2, ← c68789] at e_1113_5
  have hr := e_1113_5
  simp only [e_1113_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9618)) := by
  have c68790 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68791 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68792 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68793 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68794 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68795 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_6
  simp only [← c68790, ← c68791, ← c68792] at e_1113_6
  have e_1113_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_7
  simp only [← c68793, ← c68794, k2, ← c68795] at e_1113_7
  have hr := e_1113_7
  simp only [e_1113_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9619)) := by
  have c68796 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68797 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68798 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68799 := (publicBatchWrapper89_copies2149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68800 := (publicBatchWrapper89_copies2150 a h).1
  have c68801 := (publicBatchWrapper89_copies2150 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_8
  simp only [← c68796, ← c68797, ← c68798] at e_1113_8
  have e_1113_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_9
  simp only [← c68799, ← c68800, k2, ← c68801] at e_1113_9
  have hr := e_1113_9
  simp only [e_1113_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9620)) := by
  have c68802 := (publicBatchWrapper89_copies2150 a h).2.2.1
  have c68803 := (publicBatchWrapper89_copies2150 a h).2.2.2.1
  have c68804 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.1
  have c68805 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.1
  have c68806 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.1
  have c68807 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_10
  simp only [← c68802, ← c68803, ← c68804] at e_1113_10
  have e_1113_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_11
  simp only [← c68805, ← c68806, k2, ← c68807] at e_1113_11
  have hr := e_1113_11
  simp only [e_1113_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9621)) := by
  have c68808 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.1
  have c68809 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.1
  have c68810 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68811 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68812 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68813 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_12
  simp only [← c68808, ← c68809, ← c68810] at e_1113_12
  have e_1113_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_13
  simp only [← c68811, ← c68812, k2, ← c68813] at e_1113_13
  have hr := e_1113_13
  simp only [e_1113_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9622)) := by
  have c68814 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68815 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68816 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68817 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68818 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68819 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_14
  simp only [← c68814, ← c68815, ← c68816] at e_1113_14
  have e_1113_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_15
  simp only [← c68817, ← c68818, k2, ← c68819] at e_1113_15
  have hr := e_1113_15
  simp only [e_1113_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9623)) := by
  have c68820 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68821 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68822 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68823 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68824 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68825 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_16
  simp only [← c68820, ← c68821, ← c68822] at e_1113_16
  have e_1113_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_17
  simp only [← c68823, ← c68824, k2, ← c68825] at e_1113_17
  have hr := e_1113_17
  simp only [e_1113_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1113 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9624)) := by
  have c68826 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68827 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68828 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68829 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68830 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68831 := (publicBatchWrapper89_copies2150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1113_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_18
  simp only [← c68826, ← c68827, ← c68828] at e_1113_18
  have e_1113_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1113 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1113_19
  simp only [← c68829, ← c68830, k2, ← c68831] at e_1113_19
  have hr := e_1113_19
  simp only [e_1113_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9625)) := by
  have c68832 := (publicBatchWrapper89_copies2151 a h).1
  have c68833 := (publicBatchWrapper89_copies2151 a h).2.1
  have c68834 := (publicBatchWrapper89_copies2151 a h).2.2.1
  have c68835 := (publicBatchWrapper89_copies2151 a h).2.2.2.1
  have c68836 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.1
  have c68837 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_0
  simp only [← c68832, ← c68833, ← c68834] at e_1114_0
  have e_1114_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_1
  simp only [← c68835, ← c68836, k2, ← c68837] at e_1114_1
  have hr := e_1114_1
  simp only [e_1114_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9626)) := by
  have c68838 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.1
  have c68839 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.1
  have c68840 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.1
  have c68841 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.1
  have c68842 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68843 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_2
  simp only [← c68838, ← c68839, ← c68840] at e_1114_2
  have e_1114_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_3
  simp only [← c68841, ← c68842, k2, ← c68843] at e_1114_3
  have hr := e_1114_3
  simp only [e_1114_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9627)) := by
  have c68844 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68845 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68846 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68847 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68848 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68849 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_4
  simp only [← c68844, ← c68845, ← c68846] at e_1114_4
  have e_1114_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_5
  simp only [← c68847, ← c68848, k2, ← c68849] at e_1114_5
  have hr := e_1114_5
  simp only [e_1114_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9628)) := by
  have c68850 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68851 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68852 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68853 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68854 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68855 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_6
  simp only [← c68850, ← c68851, ← c68852] at e_1114_6
  have e_1114_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_7
  simp only [← c68853, ← c68854, k2, ← c68855] at e_1114_7
  have hr := e_1114_7
  simp only [e_1114_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9629)) := by
  have c68856 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68857 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68858 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68859 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68860 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68861 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_8
  simp only [← c68856, ← c68857, ← c68858] at e_1114_8
  have e_1114_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_9
  simp only [← c68859, ← c68860, k2, ← c68861] at e_1114_9
  have hr := e_1114_9
  simp only [e_1114_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9630)) := by
  have c68862 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68863 := (publicBatchWrapper89_copies2151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68864 := (publicBatchWrapper89_copies2152 a h).1
  have c68865 := (publicBatchWrapper89_copies2152 a h).2.1
  have c68866 := (publicBatchWrapper89_copies2152 a h).2.2.1
  have c68867 := (publicBatchWrapper89_copies2152 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_10
  simp only [← c68862, ← c68863, ← c68864] at e_1114_10
  have e_1114_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_11
  simp only [← c68865, ← c68866, k2, ← c68867] at e_1114_11
  have hr := e_1114_11
  simp only [e_1114_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9631)) := by
  have c68868 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.1
  have c68869 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.1
  have c68870 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.1
  have c68871 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.1
  have c68872 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.1
  have c68873 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_12
  simp only [← c68868, ← c68869, ← c68870] at e_1114_12
  have e_1114_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_13
  simp only [← c68871, ← c68872, k2, ← c68873] at e_1114_13
  have hr := e_1114_13
  simp only [e_1114_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9632)) := by
  have c68874 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68875 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68876 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68877 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68878 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68879 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_14
  simp only [← c68874, ← c68875, ← c68876] at e_1114_14
  have e_1114_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_15
  simp only [← c68877, ← c68878, k2, ← c68879] at e_1114_15
  have hr := e_1114_15
  simp only [e_1114_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9633)) := by
  have c68880 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68881 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68882 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68883 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68884 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68885 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_16
  simp only [← c68880, ← c68881, ← c68882] at e_1114_16
  have e_1114_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_17
  simp only [← c68883, ← c68884, k2, ← c68885] at e_1114_17
  have hr := e_1114_17
  simp only [e_1114_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1114 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9634)) := by
  have c68886 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68887 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68888 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68889 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68890 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68891 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1114_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_18
  simp only [← c68886, ← c68887, ← c68888] at e_1114_18
  have e_1114_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1114 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1114_19
  simp only [← c68889, ← c68890, k2, ← c68891] at e_1114_19
  have hr := e_1114_19
  simp only [e_1114_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9635)) := by
  have c68892 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68893 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68894 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68895 := (publicBatchWrapper89_copies2152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68896 := (publicBatchWrapper89_copies2153 a h).1
  have c68897 := (publicBatchWrapper89_copies2153 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_0
  simp only [← c68892, ← c68893, ← c68894] at e_1115_0
  have e_1115_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_1
  simp only [← c68895, ← c68896, k2, ← c68897] at e_1115_1
  have hr := e_1115_1
  simp only [e_1115_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9636)) := by
  have c68898 := (publicBatchWrapper89_copies2153 a h).2.2.1
  have c68899 := (publicBatchWrapper89_copies2153 a h).2.2.2.1
  have c68900 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.1
  have c68901 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.1
  have c68902 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.1
  have c68903 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_2
  simp only [← c68898, ← c68899, ← c68900] at e_1115_2
  have e_1115_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_3
  simp only [← c68901, ← c68902, k2, ← c68903] at e_1115_3
  have hr := e_1115_3
  simp only [e_1115_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9637)) := by
  have c68904 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.1
  have c68905 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.1
  have c68906 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68907 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68908 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68909 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_4
  simp only [← c68904, ← c68905, ← c68906] at e_1115_4
  have e_1115_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_5
  simp only [← c68907, ← c68908, k2, ← c68909] at e_1115_5
  have hr := e_1115_5
  simp only [e_1115_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9638)) := by
  have c68910 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68911 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68912 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68913 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68914 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68915 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_6
  simp only [← c68910, ← c68911, ← c68912] at e_1115_6
  have e_1115_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_7
  simp only [← c68913, ← c68914, k2, ← c68915] at e_1115_7
  have hr := e_1115_7
  simp only [e_1115_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9639)) := by
  have c68916 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68917 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68918 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68919 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68920 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68921 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_8
  simp only [← c68916, ← c68917, ← c68918] at e_1115_8
  have e_1115_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_9
  simp only [← c68919, ← c68920, k2, ← c68921] at e_1115_9
  have hr := e_1115_9
  simp only [e_1115_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9640)) := by
  have c68922 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68923 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68924 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68925 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68926 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68927 := (publicBatchWrapper89_copies2153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_10
  simp only [← c68922, ← c68923, ← c68924] at e_1115_10
  have e_1115_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_11
  simp only [← c68925, ← c68926, k2, ← c68927] at e_1115_11
  have hr := e_1115_11
  simp only [e_1115_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9641)) := by
  have c68928 := (publicBatchWrapper89_copies2154 a h).1
  have c68929 := (publicBatchWrapper89_copies2154 a h).2.1
  have c68930 := (publicBatchWrapper89_copies2154 a h).2.2.1
  have c68931 := (publicBatchWrapper89_copies2154 a h).2.2.2.1
  have c68932 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.1
  have c68933 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_12
  simp only [← c68928, ← c68929, ← c68930] at e_1115_12
  have e_1115_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_13
  simp only [← c68931, ← c68932, k2, ← c68933] at e_1115_13
  have hr := e_1115_13
  simp only [e_1115_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9642)) := by
  have c68934 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.1
  have c68935 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.1
  have c68936 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.1
  have c68937 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.1
  have c68938 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68939 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_14
  simp only [← c68934, ← c68935, ← c68936] at e_1115_14
  have e_1115_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_15
  simp only [← c68937, ← c68938, k2, ← c68939] at e_1115_15
  have hr := e_1115_15
  simp only [e_1115_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9643)) := by
  have c68940 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68941 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68942 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68943 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68944 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68945 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_16
  simp only [← c68940, ← c68941, ← c68942] at e_1115_16
  have e_1115_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_17
  simp only [← c68943, ← c68944, k2, ← c68945] at e_1115_17
  have hr := e_1115_17
  simp only [e_1115_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1115 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9644)) := by
  have c68946 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68947 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68948 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68949 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68950 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68951 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1115_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_18
  simp only [← c68946, ← c68947, ← c68948] at e_1115_18
  have e_1115_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1115 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1115_19
  simp only [← c68949, ← c68950, k2, ← c68951] at e_1115_19
  have hr := e_1115_19
  simp only [e_1115_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9645)) := by
  have c68952 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68953 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68954 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68955 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68956 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68957 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_0
  simp only [← c68952, ← c68953, ← c68954] at e_1116_0
  have e_1116_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_1
  simp only [← c68955, ← c68956, k2, ← c68957] at e_1116_1
  have hr := e_1116_1
  simp only [e_1116_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9646)) := by
  have c68958 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68959 := (publicBatchWrapper89_copies2154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68960 := (publicBatchWrapper89_copies2155 a h).1
  have c68961 := (publicBatchWrapper89_copies2155 a h).2.1
  have c68962 := (publicBatchWrapper89_copies2155 a h).2.2.1
  have c68963 := (publicBatchWrapper89_copies2155 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_2
  simp only [← c68958, ← c68959, ← c68960] at e_1116_2
  have e_1116_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_3
  simp only [← c68961, ← c68962, k2, ← c68963] at e_1116_3
  have hr := e_1116_3
  simp only [e_1116_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9647)) := by
  have c68964 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.1
  have c68965 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.1
  have c68966 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.1
  have c68967 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.1
  have c68968 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.1
  have c68969 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_4
  simp only [← c68964, ← c68965, ← c68966] at e_1116_4
  have e_1116_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_5
  simp only [← c68967, ← c68968, k2, ← c68969] at e_1116_5
  have hr := e_1116_5
  simp only [e_1116_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9648)) := by
  have c68970 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c68971 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c68972 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68973 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68974 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68975 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_6
  simp only [← c68970, ← c68971, ← c68972] at e_1116_6
  have e_1116_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_7
  simp only [← c68973, ← c68974, k2, ← c68975] at e_1116_7
  have hr := e_1116_7
  simp only [e_1116_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9649)) := by
  have c68976 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68977 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68978 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68979 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68980 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68981 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_8
  simp only [← c68976, ← c68977, ← c68978] at e_1116_8
  have e_1116_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_9
  simp only [← c68979, ← c68980, k2, ← c68981] at e_1116_9
  have hr := e_1116_9
  simp only [e_1116_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9650)) := by
  have c68982 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68983 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68984 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68985 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68986 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68987 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_10
  simp only [← c68982, ← c68983, ← c68984] at e_1116_10
  have e_1116_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_11
  simp only [← c68985, ← c68986, k2, ← c68987] at e_1116_11
  have hr := e_1116_11
  simp only [e_1116_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9651)) := by
  have c68988 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68989 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68990 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c68991 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c68992 := (publicBatchWrapper89_copies2156 a h).1
  have c68993 := (publicBatchWrapper89_copies2156 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_12
  simp only [← c68988, ← c68989, ← c68990] at e_1116_12
  have e_1116_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_13
  simp only [← c68991, ← c68992, k2, ← c68993] at e_1116_13
  have hr := e_1116_13
  simp only [e_1116_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9652)) := by
  have c68994 := (publicBatchWrapper89_copies2156 a h).2.2.1
  have c68995 := (publicBatchWrapper89_copies2156 a h).2.2.2.1
  have c68996 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.1
  have c68997 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.1
  have c68998 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.1
  have c68999 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_14
  simp only [← c68994, ← c68995, ← c68996] at e_1116_14
  have e_1116_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_15
  simp only [← c68997, ← c68998, k2, ← c68999] at e_1116_15
  have hr := e_1116_15
  simp only [e_1116_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9653)) := by
  have c69000 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.1
  have c69001 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.1
  have c69002 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69003 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69004 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69005 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_16
  simp only [← c69000, ← c69001, ← c69002] at e_1116_16
  have e_1116_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_17
  simp only [← c69003, ← c69004, k2, ← c69005] at e_1116_17
  have hr := e_1116_17
  simp only [e_1116_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9654)) := by
  have c69006 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69007 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69008 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69009 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69010 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69011 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_18
  simp only [← c69006, ← c69007, ← c69008] at e_1116_18
  have e_1116_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1116 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_19
  simp only [← c69009, ← c69010, k2, ← c69011] at e_1116_19
  have hr := e_1116_19
  simp only [e_1116_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9655)) := by
  have c69012 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69013 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69014 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69015 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69016 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69017 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_0
  simp only [← c69012, ← c69013, ← c69014] at e_1117_0
  have e_1117_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_1
  simp only [← c69015, ← c69016, k2, ← c69017] at e_1117_1
  have hr := e_1117_1
  simp only [e_1117_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9656)) := by
  have c69018 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69019 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69020 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69021 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69022 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69023 := (publicBatchWrapper89_copies2156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_2
  simp only [← c69018, ← c69019, ← c69020] at e_1117_2
  have e_1117_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_3
  simp only [← c69021, ← c69022, k2, ← c69023] at e_1117_3
  have hr := e_1117_3
  simp only [e_1117_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9657)) := by
  have c69024 := (publicBatchWrapper89_copies2157 a h).1
  have c69025 := (publicBatchWrapper89_copies2157 a h).2.1
  have c69026 := (publicBatchWrapper89_copies2157 a h).2.2.1
  have c69027 := (publicBatchWrapper89_copies2157 a h).2.2.2.1
  have c69028 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.1
  have c69029 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_4
  simp only [← c69024, ← c69025, ← c69026] at e_1117_4
  have e_1117_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_5
  simp only [← c69027, ← c69028, k2, ← c69029] at e_1117_5
  have hr := e_1117_5
  simp only [e_1117_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9658)) := by
  have c69030 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.1
  have c69031 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.1
  have c69032 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.1
  have c69033 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.1
  have c69034 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69035 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_6
  simp only [← c69030, ← c69031, ← c69032] at e_1117_6
  have e_1117_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_7
  simp only [← c69033, ← c69034, k2, ← c69035] at e_1117_7
  have hr := e_1117_7
  simp only [e_1117_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9659)) := by
  have c69036 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69037 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69038 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69039 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69040 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69041 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_8
  simp only [← c69036, ← c69037, ← c69038] at e_1117_8
  have e_1117_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_9
  simp only [← c69039, ← c69040, k2, ← c69041] at e_1117_9
  have hr := e_1117_9
  simp only [e_1117_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9660)) := by
  have c69042 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69043 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69044 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69045 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69046 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69047 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_10
  simp only [← c69042, ← c69043, ← c69044] at e_1117_10
  have e_1117_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_11
  simp only [← c69045, ← c69046, k2, ← c69047] at e_1117_11
  have hr := e_1117_11
  simp only [e_1117_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9661)) := by
  have c69048 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69049 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69050 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69051 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69052 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69053 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_12
  simp only [← c69048, ← c69049, ← c69050] at e_1117_12
  have e_1117_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_13
  simp only [← c69051, ← c69052, k2, ← c69053] at e_1117_13
  have hr := e_1117_13
  simp only [e_1117_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9662)) := by
  have c69054 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69055 := (publicBatchWrapper89_copies2157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69056 := (publicBatchWrapper89_copies2158 a h).1
  have c69057 := (publicBatchWrapper89_copies2158 a h).2.1
  have c69058 := (publicBatchWrapper89_copies2158 a h).2.2.1
  have c69059 := (publicBatchWrapper89_copies2158 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_14
  simp only [← c69054, ← c69055, ← c69056] at e_1117_14
  have e_1117_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_15
  simp only [← c69057, ← c69058, k2, ← c69059] at e_1117_15
  have hr := e_1117_15
  simp only [e_1117_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9663)) := by
  have c69060 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.1
  have c69061 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.1
  have c69062 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.1
  have c69063 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.1
  have c69064 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.1
  have c69065 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_16
  simp only [← c69060, ← c69061, ← c69062] at e_1117_16
  have e_1117_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_17
  simp only [← c69063, ← c69064, k2, ← c69065] at e_1117_17
  have hr := e_1117_17
  simp only [e_1117_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1117 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9664)) := by
  have c69066 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69067 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69068 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69069 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69070 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69071 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1117_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_18
  simp only [← c69066, ← c69067, ← c69068] at e_1117_18
  have e_1117_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1117 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_19
  simp only [← c69069, ← c69070, k2, ← c69071] at e_1117_19
  have hr := e_1117_19
  simp only [e_1117_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9665)) := by
  have c69072 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69073 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69074 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69075 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69076 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69077 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_0
  simp only [← c69072, ← c69073, ← c69074] at e_1118_0
  have e_1118_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_1
  simp only [← c69075, ← c69076, k2, ← c69077] at e_1118_1
  have hr := e_1118_1
  simp only [e_1118_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9666)) := by
  have c69078 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69079 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69080 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69081 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69082 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69083 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_2
  simp only [← c69078, ← c69079, ← c69080] at e_1118_2
  have e_1118_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_3
  simp only [← c69081, ← c69082, k2, ← c69083] at e_1118_3
  have hr := e_1118_3
  simp only [e_1118_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9667)) := by
  have c69084 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69085 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69086 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69087 := (publicBatchWrapper89_copies2158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69088 := (publicBatchWrapper89_copies2159 a h).1
  have c69089 := (publicBatchWrapper89_copies2159 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_4
  simp only [← c69084, ← c69085, ← c69086] at e_1118_4
  have e_1118_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_5
  simp only [← c69087, ← c69088, k2, ← c69089] at e_1118_5
  have hr := e_1118_5
  simp only [e_1118_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9668)) := by
  have c69090 := (publicBatchWrapper89_copies2159 a h).2.2.1
  have c69091 := (publicBatchWrapper89_copies2159 a h).2.2.2.1
  have c69092 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.1
  have c69093 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.1
  have c69094 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.1
  have c69095 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_6
  simp only [← c69090, ← c69091, ← c69092] at e_1118_6
  have e_1118_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_7
  simp only [← c69093, ← c69094, k2, ← c69095] at e_1118_7
  have hr := e_1118_7
  simp only [e_1118_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9669)) := by
  have c69096 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.1
  have c69097 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.1
  have c69098 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69099 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69100 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69101 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_8
  simp only [← c69096, ← c69097, ← c69098] at e_1118_8
  have e_1118_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_9
  simp only [← c69099, ← c69100, k2, ← c69101] at e_1118_9
  have hr := e_1118_9
  simp only [e_1118_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9670)) := by
  have c69102 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69103 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69104 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69105 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69106 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69107 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_10
  simp only [← c69102, ← c69103, ← c69104] at e_1118_10
  have e_1118_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_11
  simp only [← c69105, ← c69106, k2, ← c69107] at e_1118_11
  have hr := e_1118_11
  simp only [e_1118_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9671)) := by
  have c69108 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69109 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69110 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69111 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69112 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69113 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_12
  simp only [← c69108, ← c69109, ← c69110] at e_1118_12
  have e_1118_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_13
  simp only [← c69111, ← c69112, k2, ← c69113] at e_1118_13
  have hr := e_1118_13
  simp only [e_1118_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9672)) := by
  have c69114 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69115 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69116 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69117 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69118 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69119 := (publicBatchWrapper89_copies2159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_14
  simp only [← c69114, ← c69115, ← c69116] at e_1118_14
  have e_1118_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_15
  simp only [← c69117, ← c69118, k2, ← c69119] at e_1118_15
  have hr := e_1118_15
  simp only [e_1118_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9673)) := by
  have c69120 := (publicBatchWrapper89_copies2160 a h).1
  have c69121 := (publicBatchWrapper89_copies2160 a h).2.1
  have c69122 := (publicBatchWrapper89_copies2160 a h).2.2.1
  have c69123 := (publicBatchWrapper89_copies2160 a h).2.2.2.1
  have c69124 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.1
  have c69125 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_16
  simp only [← c69120, ← c69121, ← c69122] at e_1118_16
  have e_1118_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_17
  simp only [← c69123, ← c69124, k2, ← c69125] at e_1118_17
  have hr := e_1118_17
  simp only [e_1118_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1118 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9674)) := by
  have c69126 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.1
  have c69127 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.1
  have c69128 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.1
  have c69129 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.1
  have c69130 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69131 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1118_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_18
  simp only [← c69126, ← c69127, ← c69128] at e_1118_18
  have e_1118_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1118 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_19
  simp only [← c69129, ← c69130, k2, ← c69131] at e_1118_19
  have hr := e_1118_19
  simp only [e_1118_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9675)) := by
  have c69132 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69133 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69134 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69135 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69136 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69137 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_0
  simp only [← c69132, ← c69133, ← c69134] at e_1119_0
  have e_1119_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_1
  simp only [← c69135, ← c69136, k2, ← c69137] at e_1119_1
  have hr := e_1119_1
  simp only [e_1119_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9676)) := by
  have c69138 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69139 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69140 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69141 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69142 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69143 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_2
  simp only [← c69138, ← c69139, ← c69140] at e_1119_2
  have e_1119_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_3
  simp only [← c69141, ← c69142, k2, ← c69143] at e_1119_3
  have hr := e_1119_3
  simp only [e_1119_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9677)) := by
  have c69144 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69145 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69146 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69147 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69148 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69149 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_4
  simp only [← c69144, ← c69145, ← c69146] at e_1119_4
  have e_1119_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_5
  simp only [← c69147, ← c69148, k2, ← c69149] at e_1119_5
  have hr := e_1119_5
  simp only [e_1119_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9678)) := by
  have c69150 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69151 := (publicBatchWrapper89_copies2160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69152 := (publicBatchWrapper89_copies2161 a h).1
  have c69153 := (publicBatchWrapper89_copies2161 a h).2.1
  have c69154 := (publicBatchWrapper89_copies2161 a h).2.2.1
  have c69155 := (publicBatchWrapper89_copies2161 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_6
  simp only [← c69150, ← c69151, ← c69152] at e_1119_6
  have e_1119_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_7
  simp only [← c69153, ← c69154, k2, ← c69155] at e_1119_7
  have hr := e_1119_7
  simp only [e_1119_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9679)) := by
  have c69156 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.1
  have c69157 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.1
  have c69158 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.1
  have c69159 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.1
  have c69160 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.1
  have c69161 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_8
  simp only [← c69156, ← c69157, ← c69158] at e_1119_8
  have e_1119_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_9
  simp only [← c69159, ← c69160, k2, ← c69161] at e_1119_9
  have hr := e_1119_9
  simp only [e_1119_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9680)) := by
  have c69162 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69163 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69164 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69165 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69166 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69167 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_10
  simp only [← c69162, ← c69163, ← c69164] at e_1119_10
  have e_1119_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_11
  simp only [← c69165, ← c69166, k2, ← c69167] at e_1119_11
  have hr := e_1119_11
  simp only [e_1119_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9681)) := by
  have c69168 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69169 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69170 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69171 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69172 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69173 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_12
  simp only [← c69168, ← c69169, ← c69170] at e_1119_12
  have e_1119_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_13
  simp only [← c69171, ← c69172, k2, ← c69173] at e_1119_13
  have hr := e_1119_13
  simp only [e_1119_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9682)) := by
  have c69174 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69175 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69176 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69177 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69178 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69179 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_14
  simp only [← c69174, ← c69175, ← c69176] at e_1119_14
  have e_1119_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_15
  simp only [← c69177, ← c69178, k2, ← c69179] at e_1119_15
  have hr := e_1119_15
  simp only [e_1119_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9683)) := by
  have c69180 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69181 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69182 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69183 := (publicBatchWrapper89_copies2161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69184 := (publicBatchWrapper89_copies2162 a h).1
  have c69185 := (publicBatchWrapper89_copies2162 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_16
  simp only [← c69180, ← c69181, ← c69182] at e_1119_16
  have e_1119_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_17
  simp only [← c69183, ← c69184, k2, ← c69185] at e_1119_17
  have hr := e_1119_17
  simp only [e_1119_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1119 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9684)) := by
  have c69186 := (publicBatchWrapper89_copies2162 a h).2.2.1
  have c69187 := (publicBatchWrapper89_copies2162 a h).2.2.2.1
  have c69188 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.1
  have c69189 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.1
  have c69190 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.1
  have c69191 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1119_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_18
  simp only [← c69186, ← c69187, ← c69188] at e_1119_18
  have e_1119_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1119 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_19
  simp only [← c69189, ← c69190, k2, ← c69191] at e_1119_19
  have hr := e_1119_19
  simp only [e_1119_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9685)) := by
  have c69192 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.1
  have c69193 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.1
  have c69194 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69195 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69196 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69197 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_0
  simp only [← c69192, ← c69193, ← c69194] at e_1120_0
  have e_1120_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_1
  simp only [← c69195, ← c69196, k2, ← c69197] at e_1120_1
  have hr := e_1120_1
  simp only [e_1120_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9686)) := by
  have c69198 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69199 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69200 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69201 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69202 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69203 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_2
  simp only [← c69198, ← c69199, ← c69200] at e_1120_2
  have e_1120_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_3
  simp only [← c69201, ← c69202, k2, ← c69203] at e_1120_3
  have hr := e_1120_3
  simp only [e_1120_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9687)) := by
  have c69204 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69205 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69206 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69207 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69208 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69209 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_4
  simp only [← c69204, ← c69205, ← c69206] at e_1120_4
  have e_1120_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_5
  simp only [← c69207, ← c69208, k2, ← c69209] at e_1120_5
  have hr := e_1120_5
  simp only [e_1120_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9688)) := by
  have c69210 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69211 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69212 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69213 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69214 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69215 := (publicBatchWrapper89_copies2162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_6
  simp only [← c69210, ← c69211, ← c69212] at e_1120_6
  have e_1120_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_7
  simp only [← c69213, ← c69214, k2, ← c69215] at e_1120_7
  have hr := e_1120_7
  simp only [e_1120_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9689)) := by
  have c69216 := (publicBatchWrapper89_copies2163 a h).1
  have c69217 := (publicBatchWrapper89_copies2163 a h).2.1
  have c69218 := (publicBatchWrapper89_copies2163 a h).2.2.1
  have c69219 := (publicBatchWrapper89_copies2163 a h).2.2.2.1
  have c69220 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.1
  have c69221 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_8
  simp only [← c69216, ← c69217, ← c69218] at e_1120_8
  have e_1120_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_9
  simp only [← c69219, ← c69220, k2, ← c69221] at e_1120_9
  have hr := e_1120_9
  simp only [e_1120_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9690)) := by
  have c69222 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.1
  have c69223 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.1
  have c69224 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.1
  have c69225 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.1
  have c69226 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69227 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_10
  simp only [← c69222, ← c69223, ← c69224] at e_1120_10
  have e_1120_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_11
  simp only [← c69225, ← c69226, k2, ← c69227] at e_1120_11
  have hr := e_1120_11
  simp only [e_1120_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9691)) := by
  have c69228 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69229 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69230 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69231 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69232 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69233 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_12
  simp only [← c69228, ← c69229, ← c69230] at e_1120_12
  have e_1120_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_13
  simp only [← c69231, ← c69232, k2, ← c69233] at e_1120_13
  have hr := e_1120_13
  simp only [e_1120_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9692)) := by
  have c69234 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69235 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69236 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69237 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69238 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69239 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_14
  simp only [← c69234, ← c69235, ← c69236] at e_1120_14
  have e_1120_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_15
  simp only [← c69237, ← c69238, k2, ← c69239] at e_1120_15
  have hr := e_1120_15
  simp only [e_1120_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9693)) := by
  have c69240 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69241 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69242 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69243 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69244 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69245 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_16
  simp only [← c69240, ← c69241, ← c69242] at e_1120_16
  have e_1120_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_17
  simp only [← c69243, ← c69244, k2, ← c69245] at e_1120_17
  have hr := e_1120_17
  simp only [e_1120_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1120 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9694)) := by
  have c69246 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69247 := (publicBatchWrapper89_copies2163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69248 := (publicBatchWrapper89_copies2164 a h).1
  have c69249 := (publicBatchWrapper89_copies2164 a h).2.1
  have c69250 := (publicBatchWrapper89_copies2164 a h).2.2.1
  have c69251 := (publicBatchWrapper89_copies2164 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1120_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_18
  simp only [← c69246, ← c69247, ← c69248] at e_1120_18
  have e_1120_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1120 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_19
  simp only [← c69249, ← c69250, k2, ← c69251] at e_1120_19
  have hr := e_1120_19
  simp only [e_1120_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9695)) := by
  have c69252 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.1
  have c69253 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.1
  have c69254 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.1
  have c69255 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.1
  have c69256 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.1
  have c69257 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_0
  simp only [← c69252, ← c69253, ← c69254] at e_1121_0
  have e_1121_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_1
  simp only [← c69255, ← c69256, k2, ← c69257] at e_1121_1
  have hr := e_1121_1
  simp only [e_1121_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9696)) := by
  have c69258 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69259 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69260 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69261 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69262 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69263 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_2
  simp only [← c69258, ← c69259, ← c69260] at e_1121_2
  have e_1121_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_3
  simp only [← c69261, ← c69262, k2, ← c69263] at e_1121_3
  have hr := e_1121_3
  simp only [e_1121_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9697)) := by
  have c69264 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69265 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69266 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69267 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69268 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69269 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_4
  simp only [← c69264, ← c69265, ← c69266] at e_1121_4
  have e_1121_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_5
  simp only [← c69267, ← c69268, k2, ← c69269] at e_1121_5
  have hr := e_1121_5
  simp only [e_1121_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9698)) := by
  have c69270 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69271 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69272 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69273 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69274 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69275 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_6
  simp only [← c69270, ← c69271, ← c69272] at e_1121_6
  have e_1121_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_7
  simp only [← c69273, ← c69274, k2, ← c69275] at e_1121_7
  have hr := e_1121_7
  simp only [e_1121_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9699)) := by
  have c69276 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69277 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69278 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69279 := (publicBatchWrapper89_copies2164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69280 := (publicBatchWrapper89_copies2165 a h).1
  have c69281 := (publicBatchWrapper89_copies2165 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_8
  simp only [← c69276, ← c69277, ← c69278] at e_1121_8
  have e_1121_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_9
  simp only [← c69279, ← c69280, k2, ← c69281] at e_1121_9
  have hr := e_1121_9
  simp only [e_1121_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9700)) := by
  have c69282 := (publicBatchWrapper89_copies2165 a h).2.2.1
  have c69283 := (publicBatchWrapper89_copies2165 a h).2.2.2.1
  have c69284 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.1
  have c69285 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.1
  have c69286 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.1
  have c69287 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_10
  simp only [← c69282, ← c69283, ← c69284] at e_1121_10
  have e_1121_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_11
  simp only [← c69285, ← c69286, k2, ← c69287] at e_1121_11
  have hr := e_1121_11
  simp only [e_1121_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9701)) := by
  have c69288 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.1
  have c69289 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.1
  have c69290 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69291 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69292 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69293 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_12
  simp only [← c69288, ← c69289, ← c69290] at e_1121_12
  have e_1121_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_13
  simp only [← c69291, ← c69292, k2, ← c69293] at e_1121_13
  have hr := e_1121_13
  simp only [e_1121_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9702)) := by
  have c69294 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69295 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69296 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69297 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69298 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69299 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_14
  simp only [← c69294, ← c69295, ← c69296] at e_1121_14
  have e_1121_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_15
  simp only [← c69297, ← c69298, k2, ← c69299] at e_1121_15
  have hr := e_1121_15
  simp only [e_1121_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9703)) := by
  have c69300 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69301 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69302 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69303 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69304 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69305 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_16
  simp only [← c69300, ← c69301, ← c69302] at e_1121_16
  have e_1121_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_17
  simp only [← c69303, ← c69304, k2, ← c69305] at e_1121_17
  have hr := e_1121_17
  simp only [e_1121_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1121 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9704)) := by
  have c69306 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69307 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69308 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69309 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69310 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69311 := (publicBatchWrapper89_copies2165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1121_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_18
  simp only [← c69306, ← c69307, ← c69308] at e_1121_18
  have e_1121_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1121 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_19
  simp only [← c69309, ← c69310, k2, ← c69311] at e_1121_19
  have hr := e_1121_19
  simp only [e_1121_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9705)) := by
  have c69312 := (publicBatchWrapper89_copies2166 a h).1
  have c69313 := (publicBatchWrapper89_copies2166 a h).2.1
  have c69314 := (publicBatchWrapper89_copies2166 a h).2.2.1
  have c69315 := (publicBatchWrapper89_copies2166 a h).2.2.2.1
  have c69316 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.1
  have c69317 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_0
  simp only [← c69312, ← c69313, ← c69314] at e_1122_0
  have e_1122_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_1
  simp only [← c69315, ← c69316, k2, ← c69317] at e_1122_1
  have hr := e_1122_1
  simp only [e_1122_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9706)) := by
  have c69318 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.1
  have c69319 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.1
  have c69320 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.1
  have c69321 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.1
  have c69322 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69323 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_2
  simp only [← c69318, ← c69319, ← c69320] at e_1122_2
  have e_1122_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_3
  simp only [← c69321, ← c69322, k2, ← c69323] at e_1122_3
  have hr := e_1122_3
  simp only [e_1122_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9707)) := by
  have c69324 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69325 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69326 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69327 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69328 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69329 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_4
  simp only [← c69324, ← c69325, ← c69326] at e_1122_4
  have e_1122_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_5
  simp only [← c69327, ← c69328, k2, ← c69329] at e_1122_5
  have hr := e_1122_5
  simp only [e_1122_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9708)) := by
  have c69330 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69331 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69332 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69333 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69334 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69335 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_6
  simp only [← c69330, ← c69331, ← c69332] at e_1122_6
  have e_1122_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_7
  simp only [← c69333, ← c69334, k2, ← c69335] at e_1122_7
  have hr := e_1122_7
  simp only [e_1122_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9709)) := by
  have c69336 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69337 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69338 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69339 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69340 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69341 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_8
  simp only [← c69336, ← c69337, ← c69338] at e_1122_8
  have e_1122_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_9
  simp only [← c69339, ← c69340, k2, ← c69341] at e_1122_9
  have hr := e_1122_9
  simp only [e_1122_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9710)) := by
  have c69342 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69343 := (publicBatchWrapper89_copies2166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69344 := (publicBatchWrapper89_copies2167 a h).1
  have c69345 := (publicBatchWrapper89_copies2167 a h).2.1
  have c69346 := (publicBatchWrapper89_copies2167 a h).2.2.1
  have c69347 := (publicBatchWrapper89_copies2167 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_10
  simp only [← c69342, ← c69343, ← c69344] at e_1122_10
  have e_1122_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_11
  simp only [← c69345, ← c69346, k2, ← c69347] at e_1122_11
  have hr := e_1122_11
  simp only [e_1122_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9711)) := by
  have c69348 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.1
  have c69349 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.1
  have c69350 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.1
  have c69351 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.1
  have c69352 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.1
  have c69353 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_12
  simp only [← c69348, ← c69349, ← c69350] at e_1122_12
  have e_1122_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_13
  simp only [← c69351, ← c69352, k2, ← c69353] at e_1122_13
  have hr := e_1122_13
  simp only [e_1122_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9712)) := by
  have c69354 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69355 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69356 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69357 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69358 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69359 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_14
  simp only [← c69354, ← c69355, ← c69356] at e_1122_14
  have e_1122_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_15
  simp only [← c69357, ← c69358, k2, ← c69359] at e_1122_15
  have hr := e_1122_15
  simp only [e_1122_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9713)) := by
  have c69360 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69361 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69362 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69363 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69364 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69365 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_16
  simp only [← c69360, ← c69361, ← c69362] at e_1122_16
  have e_1122_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_17
  simp only [← c69363, ← c69364, k2, ← c69365] at e_1122_17
  have hr := e_1122_17
  simp only [e_1122_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1122 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9714)) := by
  have c69366 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69367 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69368 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69369 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69370 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69371 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1122_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_18
  simp only [← c69366, ← c69367, ← c69368] at e_1122_18
  have e_1122_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1122 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_19
  simp only [← c69369, ← c69370, k2, ← c69371] at e_1122_19
  have hr := e_1122_19
  simp only [e_1122_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9715)) := by
  have c69372 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69373 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69374 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69375 := (publicBatchWrapper89_copies2167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69376 := (publicBatchWrapper89_copies2168 a h).1
  have c69377 := (publicBatchWrapper89_copies2168 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_0
  simp only [← c69372, ← c69373, ← c69374] at e_1123_0
  have e_1123_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_1
  simp only [← c69375, ← c69376, k2, ← c69377] at e_1123_1
  have hr := e_1123_1
  simp only [e_1123_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9716)) := by
  have c69378 := (publicBatchWrapper89_copies2168 a h).2.2.1
  have c69379 := (publicBatchWrapper89_copies2168 a h).2.2.2.1
  have c69380 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.1
  have c69381 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.1
  have c69382 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.1
  have c69383 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_2
  simp only [← c69378, ← c69379, ← c69380] at e_1123_2
  have e_1123_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_3
  simp only [← c69381, ← c69382, k2, ← c69383] at e_1123_3
  have hr := e_1123_3
  simp only [e_1123_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9717)) := by
  have c69384 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.1
  have c69385 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.1
  have c69386 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69387 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69388 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69389 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_4
  simp only [← c69384, ← c69385, ← c69386] at e_1123_4
  have e_1123_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_5
  simp only [← c69387, ← c69388, k2, ← c69389] at e_1123_5
  have hr := e_1123_5
  simp only [e_1123_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9718)) := by
  have c69390 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69391 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69392 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69393 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69394 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69395 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_6
  simp only [← c69390, ← c69391, ← c69392] at e_1123_6
  have e_1123_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_7
  simp only [← c69393, ← c69394, k2, ← c69395] at e_1123_7
  have hr := e_1123_7
  simp only [e_1123_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9719)) := by
  have c69396 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69397 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69398 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69399 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69400 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69401 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_8
  simp only [← c69396, ← c69397, ← c69398] at e_1123_8
  have e_1123_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_9
  simp only [← c69399, ← c69400, k2, ← c69401] at e_1123_9
  have hr := e_1123_9
  simp only [e_1123_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9720)) := by
  have c69402 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69403 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69404 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69405 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69406 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69407 := (publicBatchWrapper89_copies2168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_10
  simp only [← c69402, ← c69403, ← c69404] at e_1123_10
  have e_1123_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_11
  simp only [← c69405, ← c69406, k2, ← c69407] at e_1123_11
  have hr := e_1123_11
  simp only [e_1123_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9721)) := by
  have c69408 := (publicBatchWrapper89_copies2169 a h).1
  have c69409 := (publicBatchWrapper89_copies2169 a h).2.1
  have c69410 := (publicBatchWrapper89_copies2169 a h).2.2.1
  have c69411 := (publicBatchWrapper89_copies2169 a h).2.2.2.1
  have c69412 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.1
  have c69413 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_12
  simp only [← c69408, ← c69409, ← c69410] at e_1123_12
  have e_1123_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_13
  simp only [← c69411, ← c69412, k2, ← c69413] at e_1123_13
  have hr := e_1123_13
  simp only [e_1123_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9722)) := by
  have c69414 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.1
  have c69415 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.1
  have c69416 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.1
  have c69417 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.1
  have c69418 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69419 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_14
  simp only [← c69414, ← c69415, ← c69416] at e_1123_14
  have e_1123_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_15
  simp only [← c69417, ← c69418, k2, ← c69419] at e_1123_15
  have hr := e_1123_15
  simp only [e_1123_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9723)) := by
  have c69420 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69421 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69422 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69423 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69424 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69425 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_16
  simp only [← c69420, ← c69421, ← c69422] at e_1123_16
  have e_1123_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_17
  simp only [← c69423, ← c69424, k2, ← c69425] at e_1123_17
  have hr := e_1123_17
  simp only [e_1123_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1123 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9724)) := by
  have c69426 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69427 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69428 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69429 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69430 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69431 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1123_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_18
  simp only [← c69426, ← c69427, ← c69428] at e_1123_18
  have e_1123_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1123 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_19
  simp only [← c69429, ← c69430, k2, ← c69431] at e_1123_19
  have hr := e_1123_19
  simp only [e_1123_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9725)) := by
  have c69432 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69433 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69434 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69435 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69436 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69437 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_0
  simp only [← c69432, ← c69433, ← c69434] at e_1124_0
  have e_1124_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_1
  simp only [← c69435, ← c69436, k2, ← c69437] at e_1124_1
  have hr := e_1124_1
  simp only [e_1124_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9726)) := by
  have c69438 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69439 := (publicBatchWrapper89_copies2169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69440 := (publicBatchWrapper89_copies2170 a h).1
  have c69441 := (publicBatchWrapper89_copies2170 a h).2.1
  have c69442 := (publicBatchWrapper89_copies2170 a h).2.2.1
  have c69443 := (publicBatchWrapper89_copies2170 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_2
  simp only [← c69438, ← c69439, ← c69440] at e_1124_2
  have e_1124_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_3
  simp only [← c69441, ← c69442, k2, ← c69443] at e_1124_3
  have hr := e_1124_3
  simp only [e_1124_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9727)) := by
  have c69444 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.1
  have c69445 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.1
  have c69446 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.1
  have c69447 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.1
  have c69448 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.1
  have c69449 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_4
  simp only [← c69444, ← c69445, ← c69446] at e_1124_4
  have e_1124_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_5
  simp only [← c69447, ← c69448, k2, ← c69449] at e_1124_5
  have hr := e_1124_5
  simp only [e_1124_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9728)) := by
  have c69450 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69451 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69452 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69453 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69454 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69455 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_6
  simp only [← c69450, ← c69451, ← c69452] at e_1124_6
  have e_1124_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_7
  simp only [← c69453, ← c69454, k2, ← c69455] at e_1124_7
  have hr := e_1124_7
  simp only [e_1124_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9729)) := by
  have c69456 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69457 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69458 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69459 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69460 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69461 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_8
  simp only [← c69456, ← c69457, ← c69458] at e_1124_8
  have e_1124_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_9
  simp only [← c69459, ← c69460, k2, ← c69461] at e_1124_9
  have hr := e_1124_9
  simp only [e_1124_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9730)) := by
  have c69462 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69463 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69464 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69465 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69466 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69467 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_10
  simp only [← c69462, ← c69463, ← c69464] at e_1124_10
  have e_1124_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_11
  simp only [← c69465, ← c69466, k2, ← c69467] at e_1124_11
  have hr := e_1124_11
  simp only [e_1124_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9731)) := by
  have c69468 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69469 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69470 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69471 := (publicBatchWrapper89_copies2170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69472 := (publicBatchWrapper89_copies2171 a h).1
  have c69473 := (publicBatchWrapper89_copies2171 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_12
  simp only [← c69468, ← c69469, ← c69470] at e_1124_12
  have e_1124_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_13
  simp only [← c69471, ← c69472, k2, ← c69473] at e_1124_13
  have hr := e_1124_13
  simp only [e_1124_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9732)) := by
  have c69474 := (publicBatchWrapper89_copies2171 a h).2.2.1
  have c69475 := (publicBatchWrapper89_copies2171 a h).2.2.2.1
  have c69476 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.1
  have c69477 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.1
  have c69478 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.1
  have c69479 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_14
  simp only [← c69474, ← c69475, ← c69476] at e_1124_14
  have e_1124_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_15
  simp only [← c69477, ← c69478, k2, ← c69479] at e_1124_15
  have hr := e_1124_15
  simp only [e_1124_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9733)) := by
  have c69480 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.1
  have c69481 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.1
  have c69482 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69483 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69484 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69485 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_16
  simp only [← c69480, ← c69481, ← c69482] at e_1124_16
  have e_1124_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_17
  simp only [← c69483, ← c69484, k2, ← c69485] at e_1124_17
  have hr := e_1124_17
  simp only [e_1124_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1124 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9734)) := by
  have c69486 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69487 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69488 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69489 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69490 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69491 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1124_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_18
  simp only [← c69486, ← c69487, ← c69488] at e_1124_18
  have e_1124_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1124 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_19
  simp only [← c69489, ← c69490, k2, ← c69491] at e_1124_19
  have hr := e_1124_19
  simp only [e_1124_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9735)) := by
  have c69492 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69493 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69494 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69495 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69496 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69497 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_0
  simp only [← c69492, ← c69493, ← c69494] at e_1125_0
  have e_1125_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_1
  simp only [← c69495, ← c69496, k2, ← c69497] at e_1125_1
  have hr := e_1125_1
  simp only [e_1125_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9736)) := by
  have c69498 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69499 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69500 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69501 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69502 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69503 := (publicBatchWrapper89_copies2171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_2
  simp only [← c69498, ← c69499, ← c69500] at e_1125_2
  have e_1125_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_3
  simp only [← c69501, ← c69502, k2, ← c69503] at e_1125_3
  have hr := e_1125_3
  simp only [e_1125_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9737)) := by
  have c69504 := (publicBatchWrapper89_copies2172 a h).1
  have c69505 := (publicBatchWrapper89_copies2172 a h).2.1
  have c69506 := (publicBatchWrapper89_copies2172 a h).2.2.1
  have c69507 := (publicBatchWrapper89_copies2172 a h).2.2.2.1
  have c69508 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.1
  have c69509 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_4
  simp only [← c69504, ← c69505, ← c69506] at e_1125_4
  have e_1125_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_5
  simp only [← c69507, ← c69508, k2, ← c69509] at e_1125_5
  have hr := e_1125_5
  simp only [e_1125_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9738)) := by
  have c69510 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.1
  have c69511 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.1
  have c69512 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.1
  have c69513 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.1
  have c69514 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69515 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_6
  simp only [← c69510, ← c69511, ← c69512] at e_1125_6
  have e_1125_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_7
  simp only [← c69513, ← c69514, k2, ← c69515] at e_1125_7
  have hr := e_1125_7
  simp only [e_1125_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9739)) := by
  have c69516 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69517 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69518 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69519 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69520 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69521 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_8
  simp only [← c69516, ← c69517, ← c69518] at e_1125_8
  have e_1125_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_9
  simp only [← c69519, ← c69520, k2, ← c69521] at e_1125_9
  have hr := e_1125_9
  simp only [e_1125_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9740)) := by
  have c69522 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69523 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69524 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69525 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69526 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69527 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_10
  simp only [← c69522, ← c69523, ← c69524] at e_1125_10
  have e_1125_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_11
  simp only [← c69525, ← c69526, k2, ← c69527] at e_1125_11
  have hr := e_1125_11
  simp only [e_1125_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9741)) := by
  have c69528 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69529 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69530 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69531 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69532 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69533 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_12
  simp only [← c69528, ← c69529, ← c69530] at e_1125_12
  have e_1125_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_13
  simp only [← c69531, ← c69532, k2, ← c69533] at e_1125_13
  have hr := e_1125_13
  simp only [e_1125_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9742)) := by
  have c69534 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69535 := (publicBatchWrapper89_copies2172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69536 := (publicBatchWrapper89_copies2173 a h).1
  have c69537 := (publicBatchWrapper89_copies2173 a h).2.1
  have c69538 := (publicBatchWrapper89_copies2173 a h).2.2.1
  have c69539 := (publicBatchWrapper89_copies2173 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_14
  simp only [← c69534, ← c69535, ← c69536] at e_1125_14
  have e_1125_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_15
  simp only [← c69537, ← c69538, k2, ← c69539] at e_1125_15
  have hr := e_1125_15
  simp only [e_1125_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9743)) := by
  have c69540 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.1
  have c69541 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.1
  have c69542 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.1
  have c69543 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.1
  have c69544 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.1
  have c69545 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_16
  simp only [← c69540, ← c69541, ← c69542] at e_1125_16
  have e_1125_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_17
  simp only [← c69543, ← c69544, k2, ← c69545] at e_1125_17
  have hr := e_1125_17
  simp only [e_1125_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1125 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9744)) := by
  have c69546 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69547 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69548 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69549 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69550 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69551 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1125_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_18
  simp only [← c69546, ← c69547, ← c69548] at e_1125_18
  have e_1125_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1125 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_19
  simp only [← c69549, ← c69550, k2, ← c69551] at e_1125_19
  have hr := e_1125_19
  simp only [e_1125_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9745)) := by
  have c69552 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69553 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69554 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69555 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69556 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69557 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_0
  simp only [← c69552, ← c69553, ← c69554] at e_1126_0
  have e_1126_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_1
  simp only [← c69555, ← c69556, k2, ← c69557] at e_1126_1
  have hr := e_1126_1
  simp only [e_1126_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9746)) := by
  have c69558 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69559 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69560 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69561 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69562 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69563 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_2
  simp only [← c69558, ← c69559, ← c69560] at e_1126_2
  have e_1126_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_3
  simp only [← c69561, ← c69562, k2, ← c69563] at e_1126_3
  have hr := e_1126_3
  simp only [e_1126_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9747)) := by
  have c69564 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69565 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69566 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69567 := (publicBatchWrapper89_copies2173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69568 := (publicBatchWrapper89_copies2174 a h).1
  have c69569 := (publicBatchWrapper89_copies2174 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_4
  simp only [← c69564, ← c69565, ← c69566] at e_1126_4
  have e_1126_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_5
  simp only [← c69567, ← c69568, k2, ← c69569] at e_1126_5
  have hr := e_1126_5
  simp only [e_1126_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9748)) := by
  have c69570 := (publicBatchWrapper89_copies2174 a h).2.2.1
  have c69571 := (publicBatchWrapper89_copies2174 a h).2.2.2.1
  have c69572 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.1
  have c69573 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.1
  have c69574 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.1
  have c69575 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_6
  simp only [← c69570, ← c69571, ← c69572] at e_1126_6
  have e_1126_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_7
  simp only [← c69573, ← c69574, k2, ← c69575] at e_1126_7
  have hr := e_1126_7
  simp only [e_1126_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9749)) := by
  have c69576 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.1
  have c69577 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.1
  have c69578 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69579 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69580 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69581 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_8
  simp only [← c69576, ← c69577, ← c69578] at e_1126_8
  have e_1126_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_9
  simp only [← c69579, ← c69580, k2, ← c69581] at e_1126_9
  have hr := e_1126_9
  simp only [e_1126_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9750)) := by
  have c69582 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69583 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69584 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69585 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69586 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69587 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_10
  simp only [← c69582, ← c69583, ← c69584] at e_1126_10
  have e_1126_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_11
  simp only [← c69585, ← c69586, k2, ← c69587] at e_1126_11
  have hr := e_1126_11
  simp only [e_1126_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9751)) := by
  have c69588 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69589 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69590 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69591 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69592 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69593 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_12
  simp only [← c69588, ← c69589, ← c69590] at e_1126_12
  have e_1126_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_13
  simp only [← c69591, ← c69592, k2, ← c69593] at e_1126_13
  have hr := e_1126_13
  simp only [e_1126_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9752)) := by
  have c69594 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69595 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69596 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69597 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69598 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69599 := (publicBatchWrapper89_copies2174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_14
  simp only [← c69594, ← c69595, ← c69596] at e_1126_14
  have e_1126_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_15
  simp only [← c69597, ← c69598, k2, ← c69599] at e_1126_15
  have hr := e_1126_15
  simp only [e_1126_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9753)) := by
  have c69600 := (publicBatchWrapper89_copies2175 a h).1
  have c69601 := (publicBatchWrapper89_copies2175 a h).2.1
  have c69602 := (publicBatchWrapper89_copies2175 a h).2.2.1
  have c69603 := (publicBatchWrapper89_copies2175 a h).2.2.2.1
  have c69604 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.1
  have c69605 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_16
  simp only [← c69600, ← c69601, ← c69602] at e_1126_16
  have e_1126_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_17
  simp only [← c69603, ← c69604, k2, ← c69605] at e_1126_17
  have hr := e_1126_17
  simp only [e_1126_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1126 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9754)) := by
  have c69606 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.1
  have c69607 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.1
  have c69608 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.1
  have c69609 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.1
  have c69610 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69611 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1126_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_18
  simp only [← c69606, ← c69607, ← c69608] at e_1126_18
  have e_1126_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1126 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_19
  simp only [← c69609, ← c69610, k2, ← c69611] at e_1126_19
  have hr := e_1126_19
  simp only [e_1126_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9755)) := by
  have c69612 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69613 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69614 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69615 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69616 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69617 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_0
  simp only [← c69612, ← c69613, ← c69614] at e_1127_0
  have e_1127_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_1
  simp only [← c69615, ← c69616, k2, ← c69617] at e_1127_1
  have hr := e_1127_1
  simp only [e_1127_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9756)) := by
  have c69618 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69619 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69620 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69621 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69622 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69623 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_2
  simp only [← c69618, ← c69619, ← c69620] at e_1127_2
  have e_1127_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_3
  simp only [← c69621, ← c69622, k2, ← c69623] at e_1127_3
  have hr := e_1127_3
  simp only [e_1127_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9757)) := by
  have c69624 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69625 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69626 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69627 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69628 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69629 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_4
  simp only [← c69624, ← c69625, ← c69626] at e_1127_4
  have e_1127_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_5
  simp only [← c69627, ← c69628, k2, ← c69629] at e_1127_5
  have hr := e_1127_5
  simp only [e_1127_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9758)) := by
  have c69630 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69631 := (publicBatchWrapper89_copies2175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69632 := (publicBatchWrapper89_copies2176 a h).1
  have c69633 := (publicBatchWrapper89_copies2176 a h).2.1
  have c69634 := (publicBatchWrapper89_copies2176 a h).2.2.1
  have c69635 := (publicBatchWrapper89_copies2176 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_6
  simp only [← c69630, ← c69631, ← c69632] at e_1127_6
  have e_1127_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_7
  simp only [← c69633, ← c69634, k2, ← c69635] at e_1127_7
  have hr := e_1127_7
  simp only [e_1127_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9759)) := by
  have c69636 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.1
  have c69637 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.1
  have c69638 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.1
  have c69639 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.1
  have c69640 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.1
  have c69641 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_8
  simp only [← c69636, ← c69637, ← c69638] at e_1127_8
  have e_1127_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_9
  simp only [← c69639, ← c69640, k2, ← c69641] at e_1127_9
  have hr := e_1127_9
  simp only [e_1127_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9760)) := by
  have c69642 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69643 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69644 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69645 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69646 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69647 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_10
  simp only [← c69642, ← c69643, ← c69644] at e_1127_10
  have e_1127_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_11
  simp only [← c69645, ← c69646, k2, ← c69647] at e_1127_11
  have hr := e_1127_11
  simp only [e_1127_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9761)) := by
  have c69648 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69649 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69650 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69651 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69652 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69653 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_12
  simp only [← c69648, ← c69649, ← c69650] at e_1127_12
  have e_1127_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_13
  simp only [← c69651, ← c69652, k2, ← c69653] at e_1127_13
  have hr := e_1127_13
  simp only [e_1127_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9762)) := by
  have c69654 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69655 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69656 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69657 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69658 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69659 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_14
  simp only [← c69654, ← c69655, ← c69656] at e_1127_14
  have e_1127_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_15
  simp only [← c69657, ← c69658, k2, ← c69659] at e_1127_15
  have hr := e_1127_15
  simp only [e_1127_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9763)) := by
  have c69660 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69661 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69662 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69663 := (publicBatchWrapper89_copies2176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69664 := (publicBatchWrapper89_copies2177 a h).1
  have c69665 := (publicBatchWrapper89_copies2177 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_16
  simp only [← c69660, ← c69661, ← c69662] at e_1127_16
  have e_1127_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_17
  simp only [← c69663, ← c69664, k2, ← c69665] at e_1127_17
  have hr := e_1127_17
  simp only [e_1127_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1127 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9764)) := by
  have c69666 := (publicBatchWrapper89_copies2177 a h).2.2.1
  have c69667 := (publicBatchWrapper89_copies2177 a h).2.2.2.1
  have c69668 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.1
  have c69669 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.1
  have c69670 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.1
  have c69671 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1127_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_18
  simp only [← c69666, ← c69667, ← c69668] at e_1127_18
  have e_1127_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1127 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_19
  simp only [← c69669, ← c69670, k2, ← c69671] at e_1127_19
  have hr := e_1127_19
  simp only [e_1127_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9765)) := by
  have c69672 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.1
  have c69673 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.1
  have c69674 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69675 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69676 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69677 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_0
  simp only [← c69672, ← c69673, ← c69674] at e_1128_0
  have e_1128_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_1
  simp only [← c69675, ← c69676, k2, ← c69677] at e_1128_1
  have hr := e_1128_1
  simp only [e_1128_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9766)) := by
  have c69678 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69679 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69680 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69681 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69682 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69683 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_2
  simp only [← c69678, ← c69679, ← c69680] at e_1128_2
  have e_1128_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_3
  simp only [← c69681, ← c69682, k2, ← c69683] at e_1128_3
  have hr := e_1128_3
  simp only [e_1128_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9767)) := by
  have c69684 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69685 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69686 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69687 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69688 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69689 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_4
  simp only [← c69684, ← c69685, ← c69686] at e_1128_4
  have e_1128_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_5
  simp only [← c69687, ← c69688, k2, ← c69689] at e_1128_5
  have hr := e_1128_5
  simp only [e_1128_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9768)) := by
  have c69690 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69691 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69692 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69693 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69694 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69695 := (publicBatchWrapper89_copies2177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_6
  simp only [← c69690, ← c69691, ← c69692] at e_1128_6
  have e_1128_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_7
  simp only [← c69693, ← c69694, k2, ← c69695] at e_1128_7
  have hr := e_1128_7
  simp only [e_1128_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9769)) := by
  have c69696 := (publicBatchWrapper89_copies2178 a h).1
  have c69697 := (publicBatchWrapper89_copies2178 a h).2.1
  have c69698 := (publicBatchWrapper89_copies2178 a h).2.2.1
  have c69699 := (publicBatchWrapper89_copies2178 a h).2.2.2.1
  have c69700 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.1
  have c69701 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_8
  simp only [← c69696, ← c69697, ← c69698] at e_1128_8
  have e_1128_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_9
  simp only [← c69699, ← c69700, k2, ← c69701] at e_1128_9
  have hr := e_1128_9
  simp only [e_1128_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9810)) := by
  have c69702 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.1
  have c69703 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.1
  have c69704 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.1
  have c69705 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.1
  have c69706 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69707 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_10
  simp only [← c69702, ← c69703, ← c69704] at e_1128_10
  have e_1128_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_11
  simp only [← c69705, ← c69706, k2, ← c69707] at e_1128_11
  have hr := e_1128_11
  simp only [e_1128_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9811)) := by
  have c69708 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69709 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69710 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69711 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69712 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69713 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_12
  simp only [← c69708, ← c69709, ← c69710] at e_1128_12
  have e_1128_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_13
  simp only [← c69711, ← c69712, k2, ← c69713] at e_1128_13
  have hr := e_1128_13
  simp only [e_1128_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9812)) := by
  have c69714 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69715 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69716 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69717 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69718 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69719 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_14
  simp only [← c69714, ← c69715, ← c69716] at e_1128_14
  have e_1128_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_15
  simp only [← c69717, ← c69718, k2, ← c69719] at e_1128_15
  have hr := e_1128_15
  simp only [e_1128_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9813)) := by
  have c69720 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69721 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69722 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69723 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69724 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69725 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_16
  simp only [← c69720, ← c69721, ← c69722] at e_1128_16
  have e_1128_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_17
  simp only [← c69723, ← c69724, k2, ← c69725] at e_1128_17
  have hr := e_1128_17
  simp only [e_1128_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1128 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9814)) := by
  have c69726 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69727 := (publicBatchWrapper89_copies2178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69728 := (publicBatchWrapper89_copies2179 a h).1
  have c69729 := (publicBatchWrapper89_copies2179 a h).2.1
  have c69730 := (publicBatchWrapper89_copies2179 a h).2.2.1
  have c69731 := (publicBatchWrapper89_copies2179 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1128_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_18
  simp only [← c69726, ← c69727, ← c69728] at e_1128_18
  have e_1128_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1128 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_19
  simp only [← c69729, ← c69730, k2, ← c69731] at e_1128_19
  have hr := e_1128_19
  simp only [e_1128_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9815)) := by
  have c69732 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.1
  have c69733 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.1
  have c69734 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.1
  have c69735 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.1
  have c69736 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.1
  have c69737 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_0
  simp only [← c69732, ← c69733, ← c69734] at e_1129_0
  have e_1129_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_1
  simp only [← c69735, ← c69736, k2, ← c69737] at e_1129_1
  have hr := e_1129_1
  simp only [e_1129_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9816)) := by
  have c69738 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69739 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69740 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69741 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69742 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69743 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_2
  simp only [← c69738, ← c69739, ← c69740] at e_1129_2
  have e_1129_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_3
  simp only [← c69741, ← c69742, k2, ← c69743] at e_1129_3
  have hr := e_1129_3
  simp only [e_1129_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9817)) := by
  have c69744 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69745 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69746 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69747 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69748 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69749 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_4
  simp only [← c69744, ← c69745, ← c69746] at e_1129_4
  have e_1129_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_5
  simp only [← c69747, ← c69748, k2, ← c69749] at e_1129_5
  have hr := e_1129_5
  simp only [e_1129_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9818)) := by
  have c69750 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69751 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69752 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69753 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69754 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69755 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_6
  simp only [← c69750, ← c69751, ← c69752] at e_1129_6
  have e_1129_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_7
  simp only [← c69753, ← c69754, k2, ← c69755] at e_1129_7
  have hr := e_1129_7
  simp only [e_1129_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9819)) := by
  have c69756 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69757 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69758 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69759 := (publicBatchWrapper89_copies2179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69760 := (publicBatchWrapper89_copies2180 a h).1
  have c69761 := (publicBatchWrapper89_copies2180 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_8
  simp only [← c69756, ← c69757, ← c69758] at e_1129_8
  have e_1129_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_9
  simp only [← c69759, ← c69760, k2, ← c69761] at e_1129_9
  have hr := e_1129_9
  simp only [e_1129_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9820)) := by
  have c69762 := (publicBatchWrapper89_copies2180 a h).2.2.1
  have c69763 := (publicBatchWrapper89_copies2180 a h).2.2.2.1
  have c69764 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.1
  have c69765 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.1
  have c69766 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.1
  have c69767 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_10
  simp only [← c69762, ← c69763, ← c69764] at e_1129_10
  have e_1129_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_11
  simp only [← c69765, ← c69766, k2, ← c69767] at e_1129_11
  have hr := e_1129_11
  simp only [e_1129_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9821)) := by
  have c69768 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.1
  have c69769 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.1
  have c69770 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69771 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69772 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69773 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_12
  simp only [← c69768, ← c69769, ← c69770] at e_1129_12
  have e_1129_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_13
  simp only [← c69771, ← c69772, k2, ← c69773] at e_1129_13
  have hr := e_1129_13
  simp only [e_1129_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9822)) := by
  have c69774 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69775 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69776 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69777 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69778 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69779 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_14
  simp only [← c69774, ← c69775, ← c69776] at e_1129_14
  have e_1129_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_15
  simp only [← c69777, ← c69778, k2, ← c69779] at e_1129_15
  have hr := e_1129_15
  simp only [e_1129_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9823)) := by
  have c69780 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69781 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69782 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69783 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69784 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69785 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_16
  simp only [← c69780, ← c69781, ← c69782] at e_1129_16
  have e_1129_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_17
  simp only [← c69783, ← c69784, k2, ← c69785] at e_1129_17
  have hr := e_1129_17
  simp only [e_1129_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1129 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9824)) := by
  have c69786 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69787 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69788 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69789 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69790 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69791 := (publicBatchWrapper89_copies2180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1129_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_18
  simp only [← c69786, ← c69787, ← c69788] at e_1129_18
  have e_1129_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1129 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_19
  simp only [← c69789, ← c69790, k2, ← c69791] at e_1129_19
  have hr := e_1129_19
  simp only [e_1129_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9825)) := by
  have c69792 := (publicBatchWrapper89_copies2181 a h).1
  have c69793 := (publicBatchWrapper89_copies2181 a h).2.1
  have c69794 := (publicBatchWrapper89_copies2181 a h).2.2.1
  have c69795 := (publicBatchWrapper89_copies2181 a h).2.2.2.1
  have c69796 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.1
  have c69797 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_0
  simp only [← c69792, ← c69793, ← c69794] at e_1130_0
  have e_1130_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_1
  simp only [← c69795, ← c69796, k2, ← c69797] at e_1130_1
  have hr := e_1130_1
  simp only [e_1130_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9826)) := by
  have c69798 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.1
  have c69799 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.1
  have c69800 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.1
  have c69801 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.1
  have c69802 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69803 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_2
  simp only [← c69798, ← c69799, ← c69800] at e_1130_2
  have e_1130_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_3
  simp only [← c69801, ← c69802, k2, ← c69803] at e_1130_3
  have hr := e_1130_3
  simp only [e_1130_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9827)) := by
  have c69804 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69805 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69806 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69807 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69808 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69809 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_4
  simp only [← c69804, ← c69805, ← c69806] at e_1130_4
  have e_1130_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_5
  simp only [← c69807, ← c69808, k2, ← c69809] at e_1130_5
  have hr := e_1130_5
  simp only [e_1130_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9828)) := by
  have c69810 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69811 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69812 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69813 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69814 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69815 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_6
  simp only [← c69810, ← c69811, ← c69812] at e_1130_6
  have e_1130_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_7
  simp only [← c69813, ← c69814, k2, ← c69815] at e_1130_7
  have hr := e_1130_7
  simp only [e_1130_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9829)) := by
  have c69816 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69817 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69818 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69819 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69820 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69821 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_8
  simp only [← c69816, ← c69817, ← c69818] at e_1130_8
  have e_1130_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_9
  simp only [← c69819, ← c69820, k2, ← c69821] at e_1130_9
  have hr := e_1130_9
  simp only [e_1130_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9830)) := by
  have c69822 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69823 := (publicBatchWrapper89_copies2181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69824 := (publicBatchWrapper89_copies2182 a h).1
  have c69825 := (publicBatchWrapper89_copies2182 a h).2.1
  have c69826 := (publicBatchWrapper89_copies2182 a h).2.2.1
  have c69827 := (publicBatchWrapper89_copies2182 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_10
  simp only [← c69822, ← c69823, ← c69824] at e_1130_10
  have e_1130_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_11
  simp only [← c69825, ← c69826, k2, ← c69827] at e_1130_11
  have hr := e_1130_11
  simp only [e_1130_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9831)) := by
  have c69828 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.1
  have c69829 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.1
  have c69830 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.1
  have c69831 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.1
  have c69832 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.1
  have c69833 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_12
  simp only [← c69828, ← c69829, ← c69830] at e_1130_12
  have e_1130_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_13
  simp only [← c69831, ← c69832, k2, ← c69833] at e_1130_13
  have hr := e_1130_13
  simp only [e_1130_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9832)) := by
  have c69834 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69835 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69836 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69837 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69838 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69839 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_14
  simp only [← c69834, ← c69835, ← c69836] at e_1130_14
  have e_1130_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_15
  simp only [← c69837, ← c69838, k2, ← c69839] at e_1130_15
  have hr := e_1130_15
  simp only [e_1130_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9833)) := by
  have c69840 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69841 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69842 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69843 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69844 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69845 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_16
  simp only [← c69840, ← c69841, ← c69842] at e_1130_16
  have e_1130_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_17
  simp only [← c69843, ← c69844, k2, ← c69845] at e_1130_17
  have hr := e_1130_17
  simp only [e_1130_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1130 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9834)) := by
  have c69846 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69847 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69848 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69849 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69850 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69851 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1130_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_18
  simp only [← c69846, ← c69847, ← c69848] at e_1130_18
  have e_1130_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1130 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_19
  simp only [← c69849, ← c69850, k2, ← c69851] at e_1130_19
  have hr := e_1130_19
  simp only [e_1130_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9835)) := by
  have c69852 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69853 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69854 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69855 := (publicBatchWrapper89_copies2182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69856 := (publicBatchWrapper89_copies2183 a h).1
  have c69857 := (publicBatchWrapper89_copies2183 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_0
  simp only [← c69852, ← c69853, ← c69854] at e_1131_0
  have e_1131_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_1
  simp only [← c69855, ← c69856, k2, ← c69857] at e_1131_1
  have hr := e_1131_1
  simp only [e_1131_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9836)) := by
  have c69858 := (publicBatchWrapper89_copies2183 a h).2.2.1
  have c69859 := (publicBatchWrapper89_copies2183 a h).2.2.2.1
  have c69860 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.1
  have c69861 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.1
  have c69862 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.1
  have c69863 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_2
  simp only [← c69858, ← c69859, ← c69860] at e_1131_2
  have e_1131_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_3
  simp only [← c69861, ← c69862, k2, ← c69863] at e_1131_3
  have hr := e_1131_3
  simp only [e_1131_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9837)) := by
  have c69864 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.1
  have c69865 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.1
  have c69866 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69867 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69868 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69869 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_4
  simp only [← c69864, ← c69865, ← c69866] at e_1131_4
  have e_1131_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_5
  simp only [← c69867, ← c69868, k2, ← c69869] at e_1131_5
  have hr := e_1131_5
  simp only [e_1131_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9838)) := by
  have c69870 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69871 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69872 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69873 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69874 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69875 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_6
  simp only [← c69870, ← c69871, ← c69872] at e_1131_6
  have e_1131_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_7
  simp only [← c69873, ← c69874, k2, ← c69875] at e_1131_7
  have hr := e_1131_7
  simp only [e_1131_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9839)) := by
  have c69876 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69877 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69878 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69879 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69880 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69881 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_8
  simp only [← c69876, ← c69877, ← c69878] at e_1131_8
  have e_1131_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_9
  simp only [← c69879, ← c69880, k2, ← c69881] at e_1131_9
  have hr := e_1131_9
  simp only [e_1131_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9840)) := by
  have c69882 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69883 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69884 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69885 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69886 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69887 := (publicBatchWrapper89_copies2183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_10
  simp only [← c69882, ← c69883, ← c69884] at e_1131_10
  have e_1131_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_11
  simp only [← c69885, ← c69886, k2, ← c69887] at e_1131_11
  have hr := e_1131_11
  simp only [e_1131_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9841)) := by
  have c69888 := (publicBatchWrapper89_copies2184 a h).1
  have c69889 := (publicBatchWrapper89_copies2184 a h).2.1
  have c69890 := (publicBatchWrapper89_copies2184 a h).2.2.1
  have c69891 := (publicBatchWrapper89_copies2184 a h).2.2.2.1
  have c69892 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.1
  have c69893 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_12
  simp only [← c69888, ← c69889, ← c69890] at e_1131_12
  have e_1131_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_13
  simp only [← c69891, ← c69892, k2, ← c69893] at e_1131_13
  have hr := e_1131_13
  simp only [e_1131_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9842)) := by
  have c69894 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.1
  have c69895 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.1
  have c69896 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.1
  have c69897 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.1
  have c69898 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69899 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_14
  simp only [← c69894, ← c69895, ← c69896] at e_1131_14
  have e_1131_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_15
  simp only [← c69897, ← c69898, k2, ← c69899] at e_1131_15
  have hr := e_1131_15
  simp only [e_1131_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9843)) := by
  have c69900 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69901 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69902 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69903 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69904 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69905 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_16
  simp only [← c69900, ← c69901, ← c69902] at e_1131_16
  have e_1131_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_17
  simp only [← c69903, ← c69904, k2, ← c69905] at e_1131_17
  have hr := e_1131_17
  simp only [e_1131_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1131 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9844)) := by
  have c69906 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69907 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69908 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69909 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69910 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69911 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1131_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_18
  simp only [← c69906, ← c69907, ← c69908] at e_1131_18
  have e_1131_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1131 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_19
  simp only [← c69909, ← c69910, k2, ← c69911] at e_1131_19
  have hr := e_1131_19
  simp only [e_1131_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9845)) := by
  have c69912 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69913 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69914 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69915 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69916 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69917 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_0
  simp only [← c69912, ← c69913, ← c69914] at e_1132_0
  have e_1132_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_1
  simp only [← c69915, ← c69916, k2, ← c69917] at e_1132_1
  have hr := e_1132_1
  simp only [e_1132_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9846)) := by
  have c69918 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69919 := (publicBatchWrapper89_copies2184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69920 := (publicBatchWrapper89_copies2185 a h).1
  have c69921 := (publicBatchWrapper89_copies2185 a h).2.1
  have c69922 := (publicBatchWrapper89_copies2185 a h).2.2.1
  have c69923 := (publicBatchWrapper89_copies2185 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_2
  simp only [← c69918, ← c69919, ← c69920] at e_1132_2
  have e_1132_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_3
  simp only [← c69921, ← c69922, k2, ← c69923] at e_1132_3
  have hr := e_1132_3
  simp only [e_1132_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9847)) := by
  have c69924 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.1
  have c69925 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.1
  have c69926 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.1
  have c69927 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.1
  have c69928 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.1
  have c69929 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_4
  simp only [← c69924, ← c69925, ← c69926] at e_1132_4
  have e_1132_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_5
  simp only [← c69927, ← c69928, k2, ← c69929] at e_1132_5
  have hr := e_1132_5
  simp only [e_1132_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9848)) := by
  have c69930 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69931 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69932 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69933 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69934 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69935 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_6
  simp only [← c69930, ← c69931, ← c69932] at e_1132_6
  have e_1132_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_7
  simp only [← c69933, ← c69934, k2, ← c69935] at e_1132_7
  have hr := e_1132_7
  simp only [e_1132_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9849)) := by
  have c69936 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69937 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69938 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69939 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69940 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69941 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_8
  simp only [← c69936, ← c69937, ← c69938] at e_1132_8
  have e_1132_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_9
  simp only [← c69939, ← c69940, k2, ← c69941] at e_1132_9
  have hr := e_1132_9
  simp only [e_1132_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9850)) := by
  have c69942 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69943 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69944 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69945 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69946 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69947 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_10
  simp only [← c69942, ← c69943, ← c69944] at e_1132_10
  have e_1132_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_11
  simp only [← c69945, ← c69946, k2, ← c69947] at e_1132_11
  have hr := e_1132_11
  simp only [e_1132_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9851)) := by
  have c69948 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69949 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69950 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69951 := (publicBatchWrapper89_copies2185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c69952 := (publicBatchWrapper89_copies2186 a h).1
  have c69953 := (publicBatchWrapper89_copies2186 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_12
  simp only [← c69948, ← c69949, ← c69950] at e_1132_12
  have e_1132_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_13
  simp only [← c69951, ← c69952, k2, ← c69953] at e_1132_13
  have hr := e_1132_13
  simp only [e_1132_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9852)) := by
  have c69954 := (publicBatchWrapper89_copies2186 a h).2.2.1
  have c69955 := (publicBatchWrapper89_copies2186 a h).2.2.2.1
  have c69956 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.1
  have c69957 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.1
  have c69958 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.1
  have c69959 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_14
  simp only [← c69954, ← c69955, ← c69956] at e_1132_14
  have e_1132_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_15
  simp only [← c69957, ← c69958, k2, ← c69959] at e_1132_15
  have hr := e_1132_15
  simp only [e_1132_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9853)) := by
  have c69960 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.1
  have c69961 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.1
  have c69962 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69963 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c69964 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69965 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_16
  simp only [← c69960, ← c69961, ← c69962] at e_1132_16
  have e_1132_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_17
  simp only [← c69963, ← c69964, k2, ← c69965] at e_1132_17
  have hr := e_1132_17
  simp only [e_1132_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1132 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9854)) := by
  have c69966 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69967 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69968 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69969 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69970 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69971 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1132_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_18
  simp only [← c69966, ← c69967, ← c69968] at e_1132_18
  have e_1132_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1132 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_19
  simp only [← c69969, ← c69970, k2, ← c69971] at e_1132_19
  have hr := e_1132_19
  simp only [e_1132_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9855)) := by
  have c69972 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69973 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69974 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69975 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69976 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69977 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_0
  simp only [← c69972, ← c69973, ← c69974] at e_1133_0
  have e_1133_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_1
  simp only [← c69975, ← c69976, k2, ← c69977] at e_1133_1
  have hr := e_1133_1
  simp only [e_1133_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9856)) := by
  have c69978 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69979 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69980 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69981 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69982 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69983 := (publicBatchWrapper89_copies2186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_2
  simp only [← c69978, ← c69979, ← c69980] at e_1133_2
  have e_1133_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_3
  simp only [← c69981, ← c69982, k2, ← c69983] at e_1133_3
  have hr := e_1133_3
  simp only [e_1133_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9857)) := by
  have c69984 := (publicBatchWrapper89_copies2187 a h).1
  have c69985 := (publicBatchWrapper89_copies2187 a h).2.1
  have c69986 := (publicBatchWrapper89_copies2187 a h).2.2.1
  have c69987 := (publicBatchWrapper89_copies2187 a h).2.2.2.1
  have c69988 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.1
  have c69989 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_4
  simp only [← c69984, ← c69985, ← c69986] at e_1133_4
  have e_1133_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_5
  simp only [← c69987, ← c69988, k2, ← c69989] at e_1133_5
  have hr := e_1133_5
  simp only [e_1133_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9858)) := by
  have c69990 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.1
  have c69991 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.1
  have c69992 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.1
  have c69993 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.1
  have c69994 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c69995 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_6
  simp only [← c69990, ← c69991, ← c69992] at e_1133_6
  have e_1133_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_7
  simp only [← c69993, ← c69994, k2, ← c69995] at e_1133_7
  have hr := e_1133_7
  simp only [e_1133_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9859)) := by
  have c69996 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69997 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69998 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c69999 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70000 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70001 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_8
  simp only [← c69996, ← c69997, ← c69998] at e_1133_8
  have e_1133_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_9
  simp only [← c69999, ← c70000, k2, ← c70001] at e_1133_9
  have hr := e_1133_9
  simp only [e_1133_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9860)) := by
  have c70002 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70003 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70004 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70005 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70006 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70007 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_10
  simp only [← c70002, ← c70003, ← c70004] at e_1133_10
  have e_1133_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_11
  simp only [← c70005, ← c70006, k2, ← c70007] at e_1133_11
  have hr := e_1133_11
  simp only [e_1133_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9861)) := by
  have c70008 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70009 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70010 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70011 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70012 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70013 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_12
  simp only [← c70008, ← c70009, ← c70010] at e_1133_12
  have e_1133_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_13
  simp only [← c70011, ← c70012, k2, ← c70013] at e_1133_13
  have hr := e_1133_13
  simp only [e_1133_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9862)) := by
  have c70014 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70015 := (publicBatchWrapper89_copies2187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70016 := (publicBatchWrapper89_copies2188 a h).1
  have c70017 := (publicBatchWrapper89_copies2188 a h).2.1
  have c70018 := (publicBatchWrapper89_copies2188 a h).2.2.1
  have c70019 := (publicBatchWrapper89_copies2188 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_14
  simp only [← c70014, ← c70015, ← c70016] at e_1133_14
  have e_1133_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_15
  simp only [← c70017, ← c70018, k2, ← c70019] at e_1133_15
  have hr := e_1133_15
  simp only [e_1133_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9863)) := by
  have c70020 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.1
  have c70021 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.1
  have c70022 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.1
  have c70023 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.1
  have c70024 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.1
  have c70025 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_16
  simp only [← c70020, ← c70021, ← c70022] at e_1133_16
  have e_1133_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_17
  simp only [← c70023, ← c70024, k2, ← c70025] at e_1133_17
  have hr := e_1133_17
  simp only [e_1133_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1133 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9864)) := by
  have c70026 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70027 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70028 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70029 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70030 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70031 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1133_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_18
  simp only [← c70026, ← c70027, ← c70028] at e_1133_18
  have e_1133_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1133 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_19
  simp only [← c70029, ← c70030, k2, ← c70031] at e_1133_19
  have hr := e_1133_19
  simp only [e_1133_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9865)) := by
  have c70032 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70033 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70034 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70035 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70036 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70037 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_0
  simp only [← c70032, ← c70033, ← c70034] at e_1134_0
  have e_1134_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_1
  simp only [← c70035, ← c70036, k2, ← c70037] at e_1134_1
  have hr := e_1134_1
  simp only [e_1134_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9866)) := by
  have c70038 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70039 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70040 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70041 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70042 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70043 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_2
  simp only [← c70038, ← c70039, ← c70040] at e_1134_2
  have e_1134_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_3
  simp only [← c70041, ← c70042, k2, ← c70043] at e_1134_3
  have hr := e_1134_3
  simp only [e_1134_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9867)) := by
  have c70044 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70045 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70046 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70047 := (publicBatchWrapper89_copies2188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70048 := (publicBatchWrapper89_copies2189 a h).1
  have c70049 := (publicBatchWrapper89_copies2189 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_4
  simp only [← c70044, ← c70045, ← c70046] at e_1134_4
  have e_1134_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_5
  simp only [← c70047, ← c70048, k2, ← c70049] at e_1134_5
  have hr := e_1134_5
  simp only [e_1134_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9868)) := by
  have c70050 := (publicBatchWrapper89_copies2189 a h).2.2.1
  have c70051 := (publicBatchWrapper89_copies2189 a h).2.2.2.1
  have c70052 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.1
  have c70053 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.1
  have c70054 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.1
  have c70055 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_6
  simp only [← c70050, ← c70051, ← c70052] at e_1134_6
  have e_1134_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_7
  simp only [← c70053, ← c70054, k2, ← c70055] at e_1134_7
  have hr := e_1134_7
  simp only [e_1134_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9869)) := by
  have c70056 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.1
  have c70057 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.1
  have c70058 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70059 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70060 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70061 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_8
  simp only [← c70056, ← c70057, ← c70058] at e_1134_8
  have e_1134_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_9
  simp only [← c70059, ← c70060, k2, ← c70061] at e_1134_9
  have hr := e_1134_9
  simp only [e_1134_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9870)) := by
  have c70062 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70063 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70064 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70065 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70066 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70067 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_10
  simp only [← c70062, ← c70063, ← c70064] at e_1134_10
  have e_1134_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_11
  simp only [← c70065, ← c70066, k2, ← c70067] at e_1134_11
  have hr := e_1134_11
  simp only [e_1134_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9871)) := by
  have c70068 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70069 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70070 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70071 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70072 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70073 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_12
  simp only [← c70068, ← c70069, ← c70070] at e_1134_12
  have e_1134_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_13
  simp only [← c70071, ← c70072, k2, ← c70073] at e_1134_13
  have hr := e_1134_13
  simp only [e_1134_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9872)) := by
  have c70074 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70075 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70076 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70077 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70078 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70079 := (publicBatchWrapper89_copies2189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_14
  simp only [← c70074, ← c70075, ← c70076] at e_1134_14
  have e_1134_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_15
  simp only [← c70077, ← c70078, k2, ← c70079] at e_1134_15
  have hr := e_1134_15
  simp only [e_1134_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9873)) := by
  have c70080 := (publicBatchWrapper89_copies2190 a h).1
  have c70081 := (publicBatchWrapper89_copies2190 a h).2.1
  have c70082 := (publicBatchWrapper89_copies2190 a h).2.2.1
  have c70083 := (publicBatchWrapper89_copies2190 a h).2.2.2.1
  have c70084 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.1
  have c70085 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_16
  simp only [← c70080, ← c70081, ← c70082] at e_1134_16
  have e_1134_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_17
  simp only [← c70083, ← c70084, k2, ← c70085] at e_1134_17
  have hr := e_1134_17
  simp only [e_1134_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1134 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9874)) := by
  have c70086 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.1
  have c70087 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.1
  have c70088 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.1
  have c70089 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.1
  have c70090 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70091 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1134_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_18
  simp only [← c70086, ← c70087, ← c70088] at e_1134_18
  have e_1134_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1134 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_19
  simp only [← c70089, ← c70090, k2, ← c70091] at e_1134_19
  have hr := e_1134_19
  simp only [e_1134_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9875)) := by
  have c70092 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70093 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70094 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70095 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70096 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70097 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_0
  simp only [← c70092, ← c70093, ← c70094] at e_1135_0
  have e_1135_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_1
  simp only [← c70095, ← c70096, k2, ← c70097] at e_1135_1
  have hr := e_1135_1
  simp only [e_1135_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9876)) := by
  have c70098 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70099 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70100 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70101 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70102 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70103 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_2
  simp only [← c70098, ← c70099, ← c70100] at e_1135_2
  have e_1135_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_3
  simp only [← c70101, ← c70102, k2, ← c70103] at e_1135_3
  have hr := e_1135_3
  simp only [e_1135_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9877)) := by
  have c70104 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70105 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70106 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70107 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70108 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70109 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_4
  simp only [← c70104, ← c70105, ← c70106] at e_1135_4
  have e_1135_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_5
  simp only [← c70107, ← c70108, k2, ← c70109] at e_1135_5
  have hr := e_1135_5
  simp only [e_1135_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9878)) := by
  have c70110 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70111 := (publicBatchWrapper89_copies2190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70112 := (publicBatchWrapper89_copies2191 a h).1
  have c70113 := (publicBatchWrapper89_copies2191 a h).2.1
  have c70114 := (publicBatchWrapper89_copies2191 a h).2.2.1
  have c70115 := (publicBatchWrapper89_copies2191 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_6
  simp only [← c70110, ← c70111, ← c70112] at e_1135_6
  have e_1135_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_7
  simp only [← c70113, ← c70114, k2, ← c70115] at e_1135_7
  have hr := e_1135_7
  simp only [e_1135_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9879)) := by
  have c70116 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.1
  have c70117 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.1
  have c70118 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.1
  have c70119 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.1
  have c70120 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.1
  have c70121 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_8
  simp only [← c70116, ← c70117, ← c70118] at e_1135_8
  have e_1135_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_9
  simp only [← c70119, ← c70120, k2, ← c70121] at e_1135_9
  have hr := e_1135_9
  simp only [e_1135_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9880)) := by
  have c70122 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70123 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70124 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70125 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70126 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70127 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_10
  simp only [← c70122, ← c70123, ← c70124] at e_1135_10
  have e_1135_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_11
  simp only [← c70125, ← c70126, k2, ← c70127] at e_1135_11
  have hr := e_1135_11
  simp only [e_1135_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9881)) := by
  have c70128 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70129 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70130 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70131 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70132 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70133 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_12
  simp only [← c70128, ← c70129, ← c70130] at e_1135_12
  have e_1135_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_13
  simp only [← c70131, ← c70132, k2, ← c70133] at e_1135_13
  have hr := e_1135_13
  simp only [e_1135_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9882)) := by
  have c70134 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70135 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70136 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70137 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70138 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70139 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_14
  simp only [← c70134, ← c70135, ← c70136] at e_1135_14
  have e_1135_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_15
  simp only [← c70137, ← c70138, k2, ← c70139] at e_1135_15
  have hr := e_1135_15
  simp only [e_1135_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9883)) := by
  have c70140 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70141 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70142 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70143 := (publicBatchWrapper89_copies2191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70144 := (publicBatchWrapper89_copies2192 a h).1
  have c70145 := (publicBatchWrapper89_copies2192 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_16
  simp only [← c70140, ← c70141, ← c70142] at e_1135_16
  have e_1135_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_17
  simp only [← c70143, ← c70144, k2, ← c70145] at e_1135_17
  have hr := e_1135_17
  simp only [e_1135_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1135 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9884)) := by
  have c70146 := (publicBatchWrapper89_copies2192 a h).2.2.1
  have c70147 := (publicBatchWrapper89_copies2192 a h).2.2.2.1
  have c70148 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.1
  have c70149 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.1
  have c70150 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.1
  have c70151 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1135_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_18
  simp only [← c70146, ← c70147, ← c70148] at e_1135_18
  have e_1135_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1135 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_19
  simp only [← c70149, ← c70150, k2, ← c70151] at e_1135_19
  have hr := e_1135_19
  simp only [e_1135_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9885)) := by
  have c70152 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.1
  have c70153 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.1
  have c70154 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70155 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70156 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70157 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_0
  simp only [← c70152, ← c70153, ← c70154] at e_1136_0
  have e_1136_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_1
  simp only [← c70155, ← c70156, k2, ← c70157] at e_1136_1
  have hr := e_1136_1
  simp only [e_1136_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9886)) := by
  have c70158 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70159 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70160 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70161 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70162 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70163 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_2
  simp only [← c70158, ← c70159, ← c70160] at e_1136_2
  have e_1136_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_3
  simp only [← c70161, ← c70162, k2, ← c70163] at e_1136_3
  have hr := e_1136_3
  simp only [e_1136_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9887)) := by
  have c70164 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70165 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70166 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70167 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70168 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70169 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_4
  simp only [← c70164, ← c70165, ← c70166] at e_1136_4
  have e_1136_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_5
  simp only [← c70167, ← c70168, k2, ← c70169] at e_1136_5
  have hr := e_1136_5
  simp only [e_1136_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9888)) := by
  have c70170 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70171 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70172 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70173 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70174 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70175 := (publicBatchWrapper89_copies2192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_6
  simp only [← c70170, ← c70171, ← c70172] at e_1136_6
  have e_1136_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_7
  simp only [← c70173, ← c70174, k2, ← c70175] at e_1136_7
  have hr := e_1136_7
  simp only [e_1136_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9889)) := by
  have c70176 := (publicBatchWrapper89_copies2193 a h).1
  have c70177 := (publicBatchWrapper89_copies2193 a h).2.1
  have c70178 := (publicBatchWrapper89_copies2193 a h).2.2.1
  have c70179 := (publicBatchWrapper89_copies2193 a h).2.2.2.1
  have c70180 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.1
  have c70181 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_8
  simp only [← c70176, ← c70177, ← c70178] at e_1136_8
  have e_1136_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_9
  simp only [← c70179, ← c70180, k2, ← c70181] at e_1136_9
  have hr := e_1136_9
  simp only [e_1136_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9890)) := by
  have c70182 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.1
  have c70183 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.1
  have c70184 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.1
  have c70185 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.1
  have c70186 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70187 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_10
  simp only [← c70182, ← c70183, ← c70184] at e_1136_10
  have e_1136_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_11
  simp only [← c70185, ← c70186, k2, ← c70187] at e_1136_11
  have hr := e_1136_11
  simp only [e_1136_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9891)) := by
  have c70188 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70189 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70190 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70191 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70192 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70193 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_12
  simp only [← c70188, ← c70189, ← c70190] at e_1136_12
  have e_1136_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_13
  simp only [← c70191, ← c70192, k2, ← c70193] at e_1136_13
  have hr := e_1136_13
  simp only [e_1136_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9892)) := by
  have c70194 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70195 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70196 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70197 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70198 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70199 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_14
  simp only [← c70194, ← c70195, ← c70196] at e_1136_14
  have e_1136_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_15
  simp only [← c70197, ← c70198, k2, ← c70199] at e_1136_15
  have hr := e_1136_15
  simp only [e_1136_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9893)) := by
  have c70200 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70201 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70202 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70203 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70204 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70205 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_16
  simp only [← c70200, ← c70201, ← c70202] at e_1136_16
  have e_1136_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_17
  simp only [← c70203, ← c70204, k2, ← c70205] at e_1136_17
  have hr := e_1136_17
  simp only [e_1136_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1136 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9894)) := by
  have c70206 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70207 := (publicBatchWrapper89_copies2193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70208 := (publicBatchWrapper89_copies2194 a h).1
  have c70209 := (publicBatchWrapper89_copies2194 a h).2.1
  have c70210 := (publicBatchWrapper89_copies2194 a h).2.2.1
  have c70211 := (publicBatchWrapper89_copies2194 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1136_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_18
  simp only [← c70206, ← c70207, ← c70208] at e_1136_18
  have e_1136_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1136 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_19
  simp only [← c70209, ← c70210, k2, ← c70211] at e_1136_19
  have hr := e_1136_19
  simp only [e_1136_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9895)) := by
  have c70212 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.1
  have c70213 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.1
  have c70214 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.1
  have c70215 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.1
  have c70216 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.1
  have c70217 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_0
  simp only [← c70212, ← c70213, ← c70214] at e_1137_0
  have e_1137_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_1
  simp only [← c70215, ← c70216, k2, ← c70217] at e_1137_1
  have hr := e_1137_1
  simp only [e_1137_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9896)) := by
  have c70218 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70219 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70220 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70221 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70222 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70223 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_2
  simp only [← c70218, ← c70219, ← c70220] at e_1137_2
  have e_1137_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_3
  simp only [← c70221, ← c70222, k2, ← c70223] at e_1137_3
  have hr := e_1137_3
  simp only [e_1137_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9897)) := by
  have c70224 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70225 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70226 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70227 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70228 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70229 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_4
  simp only [← c70224, ← c70225, ← c70226] at e_1137_4
  have e_1137_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_5
  simp only [← c70227, ← c70228, k2, ← c70229] at e_1137_5
  have hr := e_1137_5
  simp only [e_1137_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9898)) := by
  have c70230 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70231 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70232 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70233 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70234 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70235 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_6
  simp only [← c70230, ← c70231, ← c70232] at e_1137_6
  have e_1137_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_7
  simp only [← c70233, ← c70234, k2, ← c70235] at e_1137_7
  have hr := e_1137_7
  simp only [e_1137_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9899)) := by
  have c70236 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70237 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70238 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70239 := (publicBatchWrapper89_copies2194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70240 := (publicBatchWrapper89_copies2195 a h).1
  have c70241 := (publicBatchWrapper89_copies2195 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_8
  simp only [← c70236, ← c70237, ← c70238] at e_1137_8
  have e_1137_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_9
  simp only [← c70239, ← c70240, k2, ← c70241] at e_1137_9
  have hr := e_1137_9
  simp only [e_1137_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9900)) := by
  have c70242 := (publicBatchWrapper89_copies2195 a h).2.2.1
  have c70243 := (publicBatchWrapper89_copies2195 a h).2.2.2.1
  have c70244 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.1
  have c70245 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.1
  have c70246 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.1
  have c70247 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_10
  simp only [← c70242, ← c70243, ← c70244] at e_1137_10
  have e_1137_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_11
  simp only [← c70245, ← c70246, k2, ← c70247] at e_1137_11
  have hr := e_1137_11
  simp only [e_1137_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9901)) := by
  have c70248 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.1
  have c70249 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.1
  have c70250 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70251 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70252 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70253 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_12
  simp only [← c70248, ← c70249, ← c70250] at e_1137_12
  have e_1137_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_13
  simp only [← c70251, ← c70252, k2, ← c70253] at e_1137_13
  have hr := e_1137_13
  simp only [e_1137_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9902)) := by
  have c70254 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70255 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70256 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70257 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70258 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70259 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_14
  simp only [← c70254, ← c70255, ← c70256] at e_1137_14
  have e_1137_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_15
  simp only [← c70257, ← c70258, k2, ← c70259] at e_1137_15
  have hr := e_1137_15
  simp only [e_1137_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9903)) := by
  have c70260 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70261 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70262 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70263 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70264 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70265 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_16
  simp only [← c70260, ← c70261, ← c70262] at e_1137_16
  have e_1137_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_17
  simp only [← c70263, ← c70264, k2, ← c70265] at e_1137_17
  have hr := e_1137_17
  simp only [e_1137_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1137 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9904)) := by
  have c70266 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70267 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70268 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70269 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70270 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70271 := (publicBatchWrapper89_copies2195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1137_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_18
  simp only [← c70266, ← c70267, ← c70268] at e_1137_18
  have e_1137_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1137 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_19
  simp only [← c70269, ← c70270, k2, ← c70271] at e_1137_19
  have hr := e_1137_19
  simp only [e_1137_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9905)) := by
  have c70272 := (publicBatchWrapper89_copies2196 a h).1
  have c70273 := (publicBatchWrapper89_copies2196 a h).2.1
  have c70274 := (publicBatchWrapper89_copies2196 a h).2.2.1
  have c70275 := (publicBatchWrapper89_copies2196 a h).2.2.2.1
  have c70276 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.1
  have c70277 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_0
  simp only [← c70272, ← c70273, ← c70274] at e_1138_0
  have e_1138_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_1
  simp only [← c70275, ← c70276, k2, ← c70277] at e_1138_1
  have hr := e_1138_1
  simp only [e_1138_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9906)) := by
  have c70278 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.1
  have c70279 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.1
  have c70280 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.1
  have c70281 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.1
  have c70282 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70283 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_2
  simp only [← c70278, ← c70279, ← c70280] at e_1138_2
  have e_1138_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_3
  simp only [← c70281, ← c70282, k2, ← c70283] at e_1138_3
  have hr := e_1138_3
  simp only [e_1138_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9907)) := by
  have c70284 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70285 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70286 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70287 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70288 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70289 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_4
  simp only [← c70284, ← c70285, ← c70286] at e_1138_4
  have e_1138_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_5
  simp only [← c70287, ← c70288, k2, ← c70289] at e_1138_5
  have hr := e_1138_5
  simp only [e_1138_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9908)) := by
  have c70290 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70291 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70292 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70293 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70294 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70295 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_6
  simp only [← c70290, ← c70291, ← c70292] at e_1138_6
  have e_1138_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_7
  simp only [← c70293, ← c70294, k2, ← c70295] at e_1138_7
  have hr := e_1138_7
  simp only [e_1138_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9909)) := by
  have c70296 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70297 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70298 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70299 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70300 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70301 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_8
  simp only [← c70296, ← c70297, ← c70298] at e_1138_8
  have e_1138_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_9
  simp only [← c70299, ← c70300, k2, ← c70301] at e_1138_9
  have hr := e_1138_9
  simp only [e_1138_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9910)) := by
  have c70302 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70303 := (publicBatchWrapper89_copies2196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70304 := (publicBatchWrapper89_copies2197 a h).1
  have c70305 := (publicBatchWrapper89_copies2197 a h).2.1
  have c70306 := (publicBatchWrapper89_copies2197 a h).2.2.1
  have c70307 := (publicBatchWrapper89_copies2197 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_10
  simp only [← c70302, ← c70303, ← c70304] at e_1138_10
  have e_1138_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_11
  simp only [← c70305, ← c70306, k2, ← c70307] at e_1138_11
  have hr := e_1138_11
  simp only [e_1138_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9911)) := by
  have c70308 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.1
  have c70309 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.1
  have c70310 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.1
  have c70311 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.1
  have c70312 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.1
  have c70313 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_12
  simp only [← c70308, ← c70309, ← c70310] at e_1138_12
  have e_1138_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_13
  simp only [← c70311, ← c70312, k2, ← c70313] at e_1138_13
  have hr := e_1138_13
  simp only [e_1138_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9912)) := by
  have c70314 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70315 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70316 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70317 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70318 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70319 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_14
  simp only [← c70314, ← c70315, ← c70316] at e_1138_14
  have e_1138_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_15
  simp only [← c70317, ← c70318, k2, ← c70319] at e_1138_15
  have hr := e_1138_15
  simp only [e_1138_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9913)) := by
  have c70320 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70321 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70322 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70323 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70324 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70325 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_16
  simp only [← c70320, ← c70321, ← c70322] at e_1138_16
  have e_1138_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_17
  simp only [← c70323, ← c70324, k2, ← c70325] at e_1138_17
  have hr := e_1138_17
  simp only [e_1138_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1138 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9914)) := by
  have c70326 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70327 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70328 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70329 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70330 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70331 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1138_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_18
  simp only [← c70326, ← c70327, ← c70328] at e_1138_18
  have e_1138_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1138 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_19
  simp only [← c70329, ← c70330, k2, ← c70331] at e_1138_19
  have hr := e_1138_19
  simp only [e_1138_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9915)) := by
  have c70332 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70333 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70334 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70335 := (publicBatchWrapper89_copies2197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70336 := (publicBatchWrapper89_copies2198 a h).1
  have c70337 := (publicBatchWrapper89_copies2198 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_0
  simp only [← c70332, ← c70333, ← c70334] at e_1139_0
  have e_1139_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_1
  simp only [← c70335, ← c70336, k2, ← c70337] at e_1139_1
  have hr := e_1139_1
  simp only [e_1139_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9916)) := by
  have c70338 := (publicBatchWrapper89_copies2198 a h).2.2.1
  have c70339 := (publicBatchWrapper89_copies2198 a h).2.2.2.1
  have c70340 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.1
  have c70341 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.1
  have c70342 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.1
  have c70343 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_2
  simp only [← c70338, ← c70339, ← c70340] at e_1139_2
  have e_1139_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_3
  simp only [← c70341, ← c70342, k2, ← c70343] at e_1139_3
  have hr := e_1139_3
  simp only [e_1139_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9917)) := by
  have c70344 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.1
  have c70345 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.1
  have c70346 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70347 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70348 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70349 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_4
  simp only [← c70344, ← c70345, ← c70346] at e_1139_4
  have e_1139_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_5
  simp only [← c70347, ← c70348, k2, ← c70349] at e_1139_5
  have hr := e_1139_5
  simp only [e_1139_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9918)) := by
  have c70350 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70351 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70352 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70353 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70354 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70355 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_6
  simp only [← c70350, ← c70351, ← c70352] at e_1139_6
  have e_1139_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_7
  simp only [← c70353, ← c70354, k2, ← c70355] at e_1139_7
  have hr := e_1139_7
  simp only [e_1139_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9919)) := by
  have c70356 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70357 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70358 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70359 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70360 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70361 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_8
  simp only [← c70356, ← c70357, ← c70358] at e_1139_8
  have e_1139_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_9
  simp only [← c70359, ← c70360, k2, ← c70361] at e_1139_9
  have hr := e_1139_9
  simp only [e_1139_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9920)) := by
  have c70362 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70363 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70364 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70365 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70366 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70367 := (publicBatchWrapper89_copies2198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_10
  simp only [← c70362, ← c70363, ← c70364] at e_1139_10
  have e_1139_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_11
  simp only [← c70365, ← c70366, k2, ← c70367] at e_1139_11
  have hr := e_1139_11
  simp only [e_1139_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9921)) := by
  have c70368 := (publicBatchWrapper89_copies2199 a h).1
  have c70369 := (publicBatchWrapper89_copies2199 a h).2.1
  have c70370 := (publicBatchWrapper89_copies2199 a h).2.2.1
  have c70371 := (publicBatchWrapper89_copies2199 a h).2.2.2.1
  have c70372 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.1
  have c70373 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_12
  simp only [← c70368, ← c70369, ← c70370] at e_1139_12
  have e_1139_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_13
  simp only [← c70371, ← c70372, k2, ← c70373] at e_1139_13
  have hr := e_1139_13
  simp only [e_1139_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9922)) := by
  have c70374 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.1
  have c70375 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.1
  have c70376 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.1
  have c70377 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.1
  have c70378 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70379 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_14
  simp only [← c70374, ← c70375, ← c70376] at e_1139_14
  have e_1139_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_15
  simp only [← c70377, ← c70378, k2, ← c70379] at e_1139_15
  have hr := e_1139_15
  simp only [e_1139_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9923)) := by
  have c70380 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70381 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70382 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70383 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70384 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70385 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_16
  simp only [← c70380, ← c70381, ← c70382] at e_1139_16
  have e_1139_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_17
  simp only [← c70383, ← c70384, k2, ← c70385] at e_1139_17
  have hr := e_1139_17
  simp only [e_1139_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1139 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9924)) := by
  have c70386 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70387 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70388 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70389 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70390 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70391 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1139_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_18
  simp only [← c70386, ← c70387, ← c70388] at e_1139_18
  have e_1139_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1139 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_19
  simp only [← c70389, ← c70390, k2, ← c70391] at e_1139_19
  have hr := e_1139_19
  simp only [e_1139_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9925)) := by
  have c70392 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70393 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70394 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70395 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70396 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70397 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_0
  simp only [← c70392, ← c70393, ← c70394] at e_1140_0
  have e_1140_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_1
  simp only [← c70395, ← c70396, k2, ← c70397] at e_1140_1
  have hr := e_1140_1
  simp only [e_1140_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9926)) := by
  have c70398 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70399 := (publicBatchWrapper89_copies2199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70400 := (publicBatchWrapper89_copies2200 a h).1
  have c70401 := (publicBatchWrapper89_copies2200 a h).2.1
  have c70402 := (publicBatchWrapper89_copies2200 a h).2.2.1
  have c70403 := (publicBatchWrapper89_copies2200 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_2
  simp only [← c70398, ← c70399, ← c70400] at e_1140_2
  have e_1140_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_3
  simp only [← c70401, ← c70402, k2, ← c70403] at e_1140_3
  have hr := e_1140_3
  simp only [e_1140_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9927)) := by
  have c70404 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.1
  have c70405 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.1
  have c70406 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.1
  have c70407 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.1
  have c70408 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.1
  have c70409 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_4
  simp only [← c70404, ← c70405, ← c70406] at e_1140_4
  have e_1140_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_5
  simp only [← c70407, ← c70408, k2, ← c70409] at e_1140_5
  have hr := e_1140_5
  simp only [e_1140_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9928)) := by
  have c70410 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70411 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70412 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70413 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70414 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70415 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_6
  simp only [← c70410, ← c70411, ← c70412] at e_1140_6
  have e_1140_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_7
  simp only [← c70413, ← c70414, k2, ← c70415] at e_1140_7
  have hr := e_1140_7
  simp only [e_1140_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9929)) := by
  have c70416 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70417 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70418 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70419 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70420 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70421 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_8
  simp only [← c70416, ← c70417, ← c70418] at e_1140_8
  have e_1140_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_9
  simp only [← c70419, ← c70420, k2, ← c70421] at e_1140_9
  have hr := e_1140_9
  simp only [e_1140_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9930)) := by
  have c70422 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70423 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70424 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70425 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70426 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70427 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_10
  simp only [← c70422, ← c70423, ← c70424] at e_1140_10
  have e_1140_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_11
  simp only [← c70425, ← c70426, k2, ← c70427] at e_1140_11
  have hr := e_1140_11
  simp only [e_1140_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9931)) := by
  have c70428 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70429 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70430 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70431 := (publicBatchWrapper89_copies2200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70432 := (publicBatchWrapper89_copies2201 a h).1
  have c70433 := (publicBatchWrapper89_copies2201 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_12
  simp only [← c70428, ← c70429, ← c70430] at e_1140_12
  have e_1140_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_13
  simp only [← c70431, ← c70432, k2, ← c70433] at e_1140_13
  have hr := e_1140_13
  simp only [e_1140_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9932)) := by
  have c70434 := (publicBatchWrapper89_copies2201 a h).2.2.1
  have c70435 := (publicBatchWrapper89_copies2201 a h).2.2.2.1
  have c70436 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.1
  have c70437 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.1
  have c70438 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.1
  have c70439 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_14
  simp only [← c70434, ← c70435, ← c70436] at e_1140_14
  have e_1140_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_15
  simp only [← c70437, ← c70438, k2, ← c70439] at e_1140_15
  have hr := e_1140_15
  simp only [e_1140_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9933)) := by
  have c70440 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.1
  have c70441 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.1
  have c70442 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70443 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70444 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70445 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_16
  simp only [← c70440, ← c70441, ← c70442] at e_1140_16
  have e_1140_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_17
  simp only [← c70443, ← c70444, k2, ← c70445] at e_1140_17
  have hr := e_1140_17
  simp only [e_1140_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1140 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9934)) := by
  have c70446 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70447 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70448 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70449 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70450 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70451 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1140_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_18
  simp only [← c70446, ← c70447, ← c70448] at e_1140_18
  have e_1140_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1140 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_19
  simp only [← c70449, ← c70450, k2, ← c70451] at e_1140_19
  have hr := e_1140_19
  simp only [e_1140_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9935)) := by
  have c70452 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70453 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70454 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70455 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70456 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70457 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_0
  simp only [← c70452, ← c70453, ← c70454] at e_1141_0
  have e_1141_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_1
  simp only [← c70455, ← c70456, k2, ← c70457] at e_1141_1
  have hr := e_1141_1
  simp only [e_1141_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9936)) := by
  have c70458 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70459 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70460 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70461 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70462 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70463 := (publicBatchWrapper89_copies2201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_2
  simp only [← c70458, ← c70459, ← c70460] at e_1141_2
  have e_1141_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_3
  simp only [← c70461, ← c70462, k2, ← c70463] at e_1141_3
  have hr := e_1141_3
  simp only [e_1141_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9937)) := by
  have c70464 := (publicBatchWrapper89_copies2202 a h).1
  have c70465 := (publicBatchWrapper89_copies2202 a h).2.1
  have c70466 := (publicBatchWrapper89_copies2202 a h).2.2.1
  have c70467 := (publicBatchWrapper89_copies2202 a h).2.2.2.1
  have c70468 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.1
  have c70469 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_4
  simp only [← c70464, ← c70465, ← c70466] at e_1141_4
  have e_1141_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_5
  simp only [← c70467, ← c70468, k2, ← c70469] at e_1141_5
  have hr := e_1141_5
  simp only [e_1141_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9938)) := by
  have c70470 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.1
  have c70471 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.1
  have c70472 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.1
  have c70473 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.1
  have c70474 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70475 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_6
  simp only [← c70470, ← c70471, ← c70472] at e_1141_6
  have e_1141_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_7
  simp only [← c70473, ← c70474, k2, ← c70475] at e_1141_7
  have hr := e_1141_7
  simp only [e_1141_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9939)) := by
  have c70476 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70477 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70478 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70479 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70480 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70481 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_8
  simp only [← c70476, ← c70477, ← c70478] at e_1141_8
  have e_1141_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_9
  simp only [← c70479, ← c70480, k2, ← c70481] at e_1141_9
  have hr := e_1141_9
  simp only [e_1141_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9940)) := by
  have c70482 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70483 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70484 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70485 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70486 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70487 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_10
  simp only [← c70482, ← c70483, ← c70484] at e_1141_10
  have e_1141_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_11
  simp only [← c70485, ← c70486, k2, ← c70487] at e_1141_11
  have hr := e_1141_11
  simp only [e_1141_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9941)) := by
  have c70488 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70489 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70490 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70491 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70492 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70493 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_12
  simp only [← c70488, ← c70489, ← c70490] at e_1141_12
  have e_1141_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_13
  simp only [← c70491, ← c70492, k2, ← c70493] at e_1141_13
  have hr := e_1141_13
  simp only [e_1141_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9942)) := by
  have c70494 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70495 := (publicBatchWrapper89_copies2202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70496 := (publicBatchWrapper89_copies2203 a h).1
  have c70497 := (publicBatchWrapper89_copies2203 a h).2.1
  have c70498 := (publicBatchWrapper89_copies2203 a h).2.2.1
  have c70499 := (publicBatchWrapper89_copies2203 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_14
  simp only [← c70494, ← c70495, ← c70496] at e_1141_14
  have e_1141_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_15
  simp only [← c70497, ← c70498, k2, ← c70499] at e_1141_15
  have hr := e_1141_15
  simp only [e_1141_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9943)) := by
  have c70500 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.1
  have c70501 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.1
  have c70502 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.1
  have c70503 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.1
  have c70504 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.1
  have c70505 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_16
  simp only [← c70500, ← c70501, ← c70502] at e_1141_16
  have e_1141_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_17
  simp only [← c70503, ← c70504, k2, ← c70505] at e_1141_17
  have hr := e_1141_17
  simp only [e_1141_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1141 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9944)) := by
  have c70506 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70507 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70508 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70509 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70510 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70511 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1141_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_18
  simp only [← c70506, ← c70507, ← c70508] at e_1141_18
  have e_1141_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1141 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_19
  simp only [← c70509, ← c70510, k2, ← c70511] at e_1141_19
  have hr := e_1141_19
  simp only [e_1141_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9945)) := by
  have c70512 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70513 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70514 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70515 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70516 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70517 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_0
  simp only [← c70512, ← c70513, ← c70514] at e_1142_0
  have e_1142_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_1
  simp only [← c70515, ← c70516, k2, ← c70517] at e_1142_1
  have hr := e_1142_1
  simp only [e_1142_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9946)) := by
  have c70518 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70519 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70520 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70521 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70522 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70523 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_2
  simp only [← c70518, ← c70519, ← c70520] at e_1142_2
  have e_1142_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_3
  simp only [← c70521, ← c70522, k2, ← c70523] at e_1142_3
  have hr := e_1142_3
  simp only [e_1142_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9947)) := by
  have c70524 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70525 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70526 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70527 := (publicBatchWrapper89_copies2203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70528 := (publicBatchWrapper89_copies2204 a h).1
  have c70529 := (publicBatchWrapper89_copies2204 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_4
  simp only [← c70524, ← c70525, ← c70526] at e_1142_4
  have e_1142_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_5
  simp only [← c70527, ← c70528, k2, ← c70529] at e_1142_5
  have hr := e_1142_5
  simp only [e_1142_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9948)) := by
  have c70530 := (publicBatchWrapper89_copies2204 a h).2.2.1
  have c70531 := (publicBatchWrapper89_copies2204 a h).2.2.2.1
  have c70532 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.1
  have c70533 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.1
  have c70534 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.1
  have c70535 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_6
  simp only [← c70530, ← c70531, ← c70532] at e_1142_6
  have e_1142_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_7
  simp only [← c70533, ← c70534, k2, ← c70535] at e_1142_7
  have hr := e_1142_7
  simp only [e_1142_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9949)) := by
  have c70536 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.1
  have c70537 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.1
  have c70538 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70539 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70540 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70541 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_8
  simp only [← c70536, ← c70537, ← c70538] at e_1142_8
  have e_1142_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_9
  simp only [← c70539, ← c70540, k2, ← c70541] at e_1142_9
  have hr := e_1142_9
  simp only [e_1142_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9950)) := by
  have c70542 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70543 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70544 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70545 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70546 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70547 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_10
  simp only [← c70542, ← c70543, ← c70544] at e_1142_10
  have e_1142_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_11
  simp only [← c70545, ← c70546, k2, ← c70547] at e_1142_11
  have hr := e_1142_11
  simp only [e_1142_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9951)) := by
  have c70548 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70549 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70550 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70551 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70552 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70553 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_12
  simp only [← c70548, ← c70549, ← c70550] at e_1142_12
  have e_1142_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_13
  simp only [← c70551, ← c70552, k2, ← c70553] at e_1142_13
  have hr := e_1142_13
  simp only [e_1142_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9952)) := by
  have c70554 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70555 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70556 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70557 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70558 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70559 := (publicBatchWrapper89_copies2204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_14
  simp only [← c70554, ← c70555, ← c70556] at e_1142_14
  have e_1142_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_15
  simp only [← c70557, ← c70558, k2, ← c70559] at e_1142_15
  have hr := e_1142_15
  simp only [e_1142_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9953)) := by
  have c70560 := (publicBatchWrapper89_copies2205 a h).1
  have c70561 := (publicBatchWrapper89_copies2205 a h).2.1
  have c70562 := (publicBatchWrapper89_copies2205 a h).2.2.1
  have c70563 := (publicBatchWrapper89_copies2205 a h).2.2.2.1
  have c70564 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.1
  have c70565 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_16
  simp only [← c70560, ← c70561, ← c70562] at e_1142_16
  have e_1142_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_17
  simp only [← c70563, ← c70564, k2, ← c70565] at e_1142_17
  have hr := e_1142_17
  simp only [e_1142_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1142 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9954)) := by
  have c70566 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.1
  have c70567 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.1
  have c70568 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.1
  have c70569 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.1
  have c70570 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70571 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1142_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_18
  simp only [← c70566, ← c70567, ← c70568] at e_1142_18
  have e_1142_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1142 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_19
  simp only [← c70569, ← c70570, k2, ← c70571] at e_1142_19
  have hr := e_1142_19
  simp only [e_1142_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9955)) := by
  have c70572 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70573 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70574 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70575 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70576 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70577 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_0
  simp only [← c70572, ← c70573, ← c70574] at e_1143_0
  have e_1143_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_1
  simp only [← c70575, ← c70576, k2, ← c70577] at e_1143_1
  have hr := e_1143_1
  simp only [e_1143_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9956)) := by
  have c70578 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70579 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70580 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70581 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70582 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70583 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_2
  simp only [← c70578, ← c70579, ← c70580] at e_1143_2
  have e_1143_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_3
  simp only [← c70581, ← c70582, k2, ← c70583] at e_1143_3
  have hr := e_1143_3
  simp only [e_1143_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9957)) := by
  have c70584 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70585 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70586 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70587 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70588 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70589 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_4
  simp only [← c70584, ← c70585, ← c70586] at e_1143_4
  have e_1143_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_5
  simp only [← c70587, ← c70588, k2, ← c70589] at e_1143_5
  have hr := e_1143_5
  simp only [e_1143_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9958)) := by
  have c70590 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70591 := (publicBatchWrapper89_copies2205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70592 := (publicBatchWrapper89_copies2206 a h).1
  have c70593 := (publicBatchWrapper89_copies2206 a h).2.1
  have c70594 := (publicBatchWrapper89_copies2206 a h).2.2.1
  have c70595 := (publicBatchWrapper89_copies2206 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_6
  simp only [← c70590, ← c70591, ← c70592] at e_1143_6
  have e_1143_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_7
  simp only [← c70593, ← c70594, k2, ← c70595] at e_1143_7
  have hr := e_1143_7
  simp only [e_1143_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9959)) := by
  have c70596 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.1
  have c70597 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.1
  have c70598 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.1
  have c70599 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.1
  have c70600 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.1
  have c70601 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_8
  simp only [← c70596, ← c70597, ← c70598] at e_1143_8
  have e_1143_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_9
  simp only [← c70599, ← c70600, k2, ← c70601] at e_1143_9
  have hr := e_1143_9
  simp only [e_1143_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9960)) := by
  have c70602 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70603 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70604 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70605 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70606 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70607 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_10
  simp only [← c70602, ← c70603, ← c70604] at e_1143_10
  have e_1143_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_11
  simp only [← c70605, ← c70606, k2, ← c70607] at e_1143_11
  have hr := e_1143_11
  simp only [e_1143_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9961)) := by
  have c70608 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70609 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70610 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70611 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70612 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70613 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_12
  simp only [← c70608, ← c70609, ← c70610] at e_1143_12
  have e_1143_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_13
  simp only [← c70611, ← c70612, k2, ← c70613] at e_1143_13
  have hr := e_1143_13
  simp only [e_1143_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9962)) := by
  have c70614 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70615 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70616 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70617 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70618 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70619 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_14
  simp only [← c70614, ← c70615, ← c70616] at e_1143_14
  have e_1143_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_15
  simp only [← c70617, ← c70618, k2, ← c70619] at e_1143_15
  have hr := e_1143_15
  simp only [e_1143_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9963)) := by
  have c70620 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70621 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70622 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70623 := (publicBatchWrapper89_copies2206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70624 := (publicBatchWrapper89_copies2207 a h).1
  have c70625 := (publicBatchWrapper89_copies2207 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_16
  simp only [← c70620, ← c70621, ← c70622] at e_1143_16
  have e_1143_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_17
  simp only [← c70623, ← c70624, k2, ← c70625] at e_1143_17
  have hr := e_1143_17
  simp only [e_1143_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1143 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9964)) := by
  have c70626 := (publicBatchWrapper89_copies2207 a h).2.2.1
  have c70627 := (publicBatchWrapper89_copies2207 a h).2.2.2.1
  have c70628 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.1
  have c70629 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.1
  have c70630 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.1
  have c70631 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1143_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_18
  simp only [← c70626, ← c70627, ← c70628] at e_1143_18
  have e_1143_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1143 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_19
  simp only [← c70629, ← c70630, k2, ← c70631] at e_1143_19
  have hr := e_1143_19
  simp only [e_1143_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9965)) := by
  have c70632 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.1
  have c70633 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.1
  have c70634 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70635 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70636 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70637 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_0
  simp only [← c70632, ← c70633, ← c70634] at e_1144_0
  have e_1144_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_1
  simp only [← c70635, ← c70636, k2, ← c70637] at e_1144_1
  have hr := e_1144_1
  simp only [e_1144_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9966)) := by
  have c70638 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70639 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70640 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70641 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70642 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70643 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_2
  simp only [← c70638, ← c70639, ← c70640] at e_1144_2
  have e_1144_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_3
  simp only [← c70641, ← c70642, k2, ← c70643] at e_1144_3
  have hr := e_1144_3
  simp only [e_1144_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9967)) := by
  have c70644 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70645 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70646 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70647 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70648 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70649 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_4
  simp only [← c70644, ← c70645, ← c70646] at e_1144_4
  have e_1144_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_5
  simp only [← c70647, ← c70648, k2, ← c70649] at e_1144_5
  have hr := e_1144_5
  simp only [e_1144_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9968)) := by
  have c70650 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70651 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70652 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70653 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70654 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70655 := (publicBatchWrapper89_copies2207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_6
  simp only [← c70650, ← c70651, ← c70652] at e_1144_6
  have e_1144_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_7
  simp only [← c70653, ← c70654, k2, ← c70655] at e_1144_7
  have hr := e_1144_7
  simp only [e_1144_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9969)) := by
  have c70656 := (publicBatchWrapper89_copies2208 a h).1
  have c70657 := (publicBatchWrapper89_copies2208 a h).2.1
  have c70658 := (publicBatchWrapper89_copies2208 a h).2.2.1
  have c70659 := (publicBatchWrapper89_copies2208 a h).2.2.2.1
  have c70660 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.1
  have c70661 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_8
  simp only [← c70656, ← c70657, ← c70658] at e_1144_8
  have e_1144_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_9
  simp only [← c70659, ← c70660, k2, ← c70661] at e_1144_9
  have hr := e_1144_9
  simp only [e_1144_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10010)) := by
  have c70662 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.1
  have c70663 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.1
  have c70664 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.1
  have c70665 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.1
  have c70666 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70667 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_10
  simp only [← c70662, ← c70663, ← c70664] at e_1144_10
  have e_1144_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_11
  simp only [← c70665, ← c70666, k2, ← c70667] at e_1144_11
  have hr := e_1144_11
  simp only [e_1144_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10011)) := by
  have c70668 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70669 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70670 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70671 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70672 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70673 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_12
  simp only [← c70668, ← c70669, ← c70670] at e_1144_12
  have e_1144_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_13
  simp only [← c70671, ← c70672, k2, ← c70673] at e_1144_13
  have hr := e_1144_13
  simp only [e_1144_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10012)) := by
  have c70674 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70675 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70676 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70677 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70678 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70679 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_14
  simp only [← c70674, ← c70675, ← c70676] at e_1144_14
  have e_1144_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_15
  simp only [← c70677, ← c70678, k2, ← c70679] at e_1144_15
  have hr := e_1144_15
  simp only [e_1144_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10013)) := by
  have c70680 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70681 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70682 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70683 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70684 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70685 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_16
  simp only [← c70680, ← c70681, ← c70682] at e_1144_16
  have e_1144_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_17
  simp only [← c70683, ← c70684, k2, ← c70685] at e_1144_17
  have hr := e_1144_17
  simp only [e_1144_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1144 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10014)) := by
  have c70686 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70687 := (publicBatchWrapper89_copies2208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70688 := (publicBatchWrapper89_copies2209 a h).1
  have c70689 := (publicBatchWrapper89_copies2209 a h).2.1
  have c70690 := (publicBatchWrapper89_copies2209 a h).2.2.1
  have c70691 := (publicBatchWrapper89_copies2209 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1144_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_18
  simp only [← c70686, ← c70687, ← c70688] at e_1144_18
  have e_1144_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1144 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_19
  simp only [← c70689, ← c70690, k2, ← c70691] at e_1144_19
  have hr := e_1144_19
  simp only [e_1144_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10015)) := by
  have c70692 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.1
  have c70693 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.1
  have c70694 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.1
  have c70695 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.1
  have c70696 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.1
  have c70697 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_0
  simp only [← c70692, ← c70693, ← c70694] at e_1145_0
  have e_1145_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_1
  simp only [← c70695, ← c70696, k2, ← c70697] at e_1145_1
  have hr := e_1145_1
  simp only [e_1145_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10016)) := by
  have c70698 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70699 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70700 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70701 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70702 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70703 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_2
  simp only [← c70698, ← c70699, ← c70700] at e_1145_2
  have e_1145_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_3
  simp only [← c70701, ← c70702, k2, ← c70703] at e_1145_3
  have hr := e_1145_3
  simp only [e_1145_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10017)) := by
  have c70704 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70705 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70706 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70707 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70708 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70709 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_4
  simp only [← c70704, ← c70705, ← c70706] at e_1145_4
  have e_1145_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_5
  simp only [← c70707, ← c70708, k2, ← c70709] at e_1145_5
  have hr := e_1145_5
  simp only [e_1145_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10018)) := by
  have c70710 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70711 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70712 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70713 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70714 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70715 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_6
  simp only [← c70710, ← c70711, ← c70712] at e_1145_6
  have e_1145_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_7
  simp only [← c70713, ← c70714, k2, ← c70715] at e_1145_7
  have hr := e_1145_7
  simp only [e_1145_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10019)) := by
  have c70716 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70717 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70718 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70719 := (publicBatchWrapper89_copies2209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70720 := (publicBatchWrapper89_copies2210 a h).1
  have c70721 := (publicBatchWrapper89_copies2210 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_8
  simp only [← c70716, ← c70717, ← c70718] at e_1145_8
  have e_1145_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_9
  simp only [← c70719, ← c70720, k2, ← c70721] at e_1145_9
  have hr := e_1145_9
  simp only [e_1145_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10020)) := by
  have c70722 := (publicBatchWrapper89_copies2210 a h).2.2.1
  have c70723 := (publicBatchWrapper89_copies2210 a h).2.2.2.1
  have c70724 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.1
  have c70725 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.1
  have c70726 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.1
  have c70727 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_10
  simp only [← c70722, ← c70723, ← c70724] at e_1145_10
  have e_1145_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_11
  simp only [← c70725, ← c70726, k2, ← c70727] at e_1145_11
  have hr := e_1145_11
  simp only [e_1145_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10021)) := by
  have c70728 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.1
  have c70729 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.1
  have c70730 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70731 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70732 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70733 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_12
  simp only [← c70728, ← c70729, ← c70730] at e_1145_12
  have e_1145_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_13
  simp only [← c70731, ← c70732, k2, ← c70733] at e_1145_13
  have hr := e_1145_13
  simp only [e_1145_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10022)) := by
  have c70734 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70735 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70736 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70737 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70738 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70739 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_14
  simp only [← c70734, ← c70735, ← c70736] at e_1145_14
  have e_1145_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_15
  simp only [← c70737, ← c70738, k2, ← c70739] at e_1145_15
  have hr := e_1145_15
  simp only [e_1145_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10023)) := by
  have c70740 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70741 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70742 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70743 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70744 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70745 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_16
  simp only [← c70740, ← c70741, ← c70742] at e_1145_16
  have e_1145_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_17
  simp only [← c70743, ← c70744, k2, ← c70745] at e_1145_17
  have hr := e_1145_17
  simp only [e_1145_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1145 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10024)) := by
  have c70746 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70747 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70748 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70749 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70750 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70751 := (publicBatchWrapper89_copies2210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1145_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_18
  simp only [← c70746, ← c70747, ← c70748] at e_1145_18
  have e_1145_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1145 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_19
  simp only [← c70749, ← c70750, k2, ← c70751] at e_1145_19
  have hr := e_1145_19
  simp only [e_1145_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10025)) := by
  have c70752 := (publicBatchWrapper89_copies2211 a h).1
  have c70753 := (publicBatchWrapper89_copies2211 a h).2.1
  have c70754 := (publicBatchWrapper89_copies2211 a h).2.2.1
  have c70755 := (publicBatchWrapper89_copies2211 a h).2.2.2.1
  have c70756 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.1
  have c70757 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_0
  simp only [← c70752, ← c70753, ← c70754] at e_1146_0
  have e_1146_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_1
  simp only [← c70755, ← c70756, k2, ← c70757] at e_1146_1
  have hr := e_1146_1
  simp only [e_1146_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10026)) := by
  have c70758 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.1
  have c70759 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.1
  have c70760 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.1
  have c70761 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.1
  have c70762 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70763 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_2
  simp only [← c70758, ← c70759, ← c70760] at e_1146_2
  have e_1146_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_3
  simp only [← c70761, ← c70762, k2, ← c70763] at e_1146_3
  have hr := e_1146_3
  simp only [e_1146_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10027)) := by
  have c70764 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70765 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70766 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70767 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70768 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70769 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_4
  simp only [← c70764, ← c70765, ← c70766] at e_1146_4
  have e_1146_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_5
  simp only [← c70767, ← c70768, k2, ← c70769] at e_1146_5
  have hr := e_1146_5
  simp only [e_1146_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10028)) := by
  have c70770 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70771 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70772 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70773 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70774 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70775 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_6
  simp only [← c70770, ← c70771, ← c70772] at e_1146_6
  have e_1146_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_7
  simp only [← c70773, ← c70774, k2, ← c70775] at e_1146_7
  have hr := e_1146_7
  simp only [e_1146_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10029)) := by
  have c70776 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70777 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70778 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70779 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70780 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70781 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_8
  simp only [← c70776, ← c70777, ← c70778] at e_1146_8
  have e_1146_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_9
  simp only [← c70779, ← c70780, k2, ← c70781] at e_1146_9
  have hr := e_1146_9
  simp only [e_1146_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10030)) := by
  have c70782 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70783 := (publicBatchWrapper89_copies2211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70784 := (publicBatchWrapper89_copies2212 a h).1
  have c70785 := (publicBatchWrapper89_copies2212 a h).2.1
  have c70786 := (publicBatchWrapper89_copies2212 a h).2.2.1
  have c70787 := (publicBatchWrapper89_copies2212 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_10
  simp only [← c70782, ← c70783, ← c70784] at e_1146_10
  have e_1146_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_11
  simp only [← c70785, ← c70786, k2, ← c70787] at e_1146_11
  have hr := e_1146_11
  simp only [e_1146_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10031)) := by
  have c70788 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.1
  have c70789 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.1
  have c70790 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.1
  have c70791 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.1
  have c70792 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.1
  have c70793 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_12
  simp only [← c70788, ← c70789, ← c70790] at e_1146_12
  have e_1146_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_13
  simp only [← c70791, ← c70792, k2, ← c70793] at e_1146_13
  have hr := e_1146_13
  simp only [e_1146_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10032)) := by
  have c70794 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70795 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70796 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70797 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70798 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70799 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_14
  simp only [← c70794, ← c70795, ← c70796] at e_1146_14
  have e_1146_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_15
  simp only [← c70797, ← c70798, k2, ← c70799] at e_1146_15
  have hr := e_1146_15
  simp only [e_1146_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10033)) := by
  have c70800 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70801 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70802 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70803 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70804 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70805 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_16
  simp only [← c70800, ← c70801, ← c70802] at e_1146_16
  have e_1146_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_17
  simp only [← c70803, ← c70804, k2, ← c70805] at e_1146_17
  have hr := e_1146_17
  simp only [e_1146_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1146 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10034)) := by
  have c70806 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70807 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70808 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70809 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70810 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70811 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1146_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_18
  simp only [← c70806, ← c70807, ← c70808] at e_1146_18
  have e_1146_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1146 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_19
  simp only [← c70809, ← c70810, k2, ← c70811] at e_1146_19
  have hr := e_1146_19
  simp only [e_1146_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10035)) := by
  have c70812 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70813 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70814 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70815 := (publicBatchWrapper89_copies2212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70816 := (publicBatchWrapper89_copies2213 a h).1
  have c70817 := (publicBatchWrapper89_copies2213 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_0
  simp only [← c70812, ← c70813, ← c70814] at e_1147_0
  have e_1147_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_1
  simp only [← c70815, ← c70816, k2, ← c70817] at e_1147_1
  have hr := e_1147_1
  simp only [e_1147_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10036)) := by
  have c70818 := (publicBatchWrapper89_copies2213 a h).2.2.1
  have c70819 := (publicBatchWrapper89_copies2213 a h).2.2.2.1
  have c70820 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.1
  have c70821 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.1
  have c70822 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.1
  have c70823 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_2
  simp only [← c70818, ← c70819, ← c70820] at e_1147_2
  have e_1147_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_3
  simp only [← c70821, ← c70822, k2, ← c70823] at e_1147_3
  have hr := e_1147_3
  simp only [e_1147_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10037)) := by
  have c70824 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.1
  have c70825 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.1
  have c70826 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70827 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70828 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70829 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_4
  simp only [← c70824, ← c70825, ← c70826] at e_1147_4
  have e_1147_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_5
  simp only [← c70827, ← c70828, k2, ← c70829] at e_1147_5
  have hr := e_1147_5
  simp only [e_1147_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10038)) := by
  have c70830 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70831 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70832 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70833 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70834 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70835 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_6
  simp only [← c70830, ← c70831, ← c70832] at e_1147_6
  have e_1147_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_7
  simp only [← c70833, ← c70834, k2, ← c70835] at e_1147_7
  have hr := e_1147_7
  simp only [e_1147_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10039)) := by
  have c70836 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70837 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70838 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70839 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70840 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70841 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_8
  simp only [← c70836, ← c70837, ← c70838] at e_1147_8
  have e_1147_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_9
  simp only [← c70839, ← c70840, k2, ← c70841] at e_1147_9
  have hr := e_1147_9
  simp only [e_1147_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10040)) := by
  have c70842 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70843 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70844 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70845 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70846 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70847 := (publicBatchWrapper89_copies2213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_10
  simp only [← c70842, ← c70843, ← c70844] at e_1147_10
  have e_1147_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_11
  simp only [← c70845, ← c70846, k2, ← c70847] at e_1147_11
  have hr := e_1147_11
  simp only [e_1147_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10041)) := by
  have c70848 := (publicBatchWrapper89_copies2214 a h).1
  have c70849 := (publicBatchWrapper89_copies2214 a h).2.1
  have c70850 := (publicBatchWrapper89_copies2214 a h).2.2.1
  have c70851 := (publicBatchWrapper89_copies2214 a h).2.2.2.1
  have c70852 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.1
  have c70853 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_12
  simp only [← c70848, ← c70849, ← c70850] at e_1147_12
  have e_1147_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_13
  simp only [← c70851, ← c70852, k2, ← c70853] at e_1147_13
  have hr := e_1147_13
  simp only [e_1147_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10042)) := by
  have c70854 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.1
  have c70855 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.1
  have c70856 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.1
  have c70857 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.1
  have c70858 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70859 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_14
  simp only [← c70854, ← c70855, ← c70856] at e_1147_14
  have e_1147_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_15
  simp only [← c70857, ← c70858, k2, ← c70859] at e_1147_15
  have hr := e_1147_15
  simp only [e_1147_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10043)) := by
  have c70860 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70861 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70862 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70863 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70864 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70865 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_16
  simp only [← c70860, ← c70861, ← c70862] at e_1147_16
  have e_1147_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_17
  simp only [← c70863, ← c70864, k2, ← c70865] at e_1147_17
  have hr := e_1147_17
  simp only [e_1147_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1147 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10044)) := by
  have c70866 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70867 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70868 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70869 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70870 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70871 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1147_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_18
  simp only [← c70866, ← c70867, ← c70868] at e_1147_18
  have e_1147_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1147 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_19
  simp only [← c70869, ← c70870, k2, ← c70871] at e_1147_19
  have hr := e_1147_19
  simp only [e_1147_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10045)) := by
  have c70872 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70873 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70874 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70875 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70876 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70877 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_0
  simp only [← c70872, ← c70873, ← c70874] at e_1148_0
  have e_1148_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_1
  simp only [← c70875, ← c70876, k2, ← c70877] at e_1148_1
  have hr := e_1148_1
  simp only [e_1148_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10046)) := by
  have c70878 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70879 := (publicBatchWrapper89_copies2214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70880 := (publicBatchWrapper89_copies2215 a h).1
  have c70881 := (publicBatchWrapper89_copies2215 a h).2.1
  have c70882 := (publicBatchWrapper89_copies2215 a h).2.2.1
  have c70883 := (publicBatchWrapper89_copies2215 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_2
  simp only [← c70878, ← c70879, ← c70880] at e_1148_2
  have e_1148_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_3
  simp only [← c70881, ← c70882, k2, ← c70883] at e_1148_3
  have hr := e_1148_3
  simp only [e_1148_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10047)) := by
  have c70884 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.1
  have c70885 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.1
  have c70886 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.1
  have c70887 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.1
  have c70888 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.1
  have c70889 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_4
  simp only [← c70884, ← c70885, ← c70886] at e_1148_4
  have e_1148_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_5
  simp only [← c70887, ← c70888, k2, ← c70889] at e_1148_5
  have hr := e_1148_5
  simp only [e_1148_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10048)) := by
  have c70890 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70891 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70892 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70893 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70894 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70895 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_6
  simp only [← c70890, ← c70891, ← c70892] at e_1148_6
  have e_1148_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_7
  simp only [← c70893, ← c70894, k2, ← c70895] at e_1148_7
  have hr := e_1148_7
  simp only [e_1148_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10049)) := by
  have c70896 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70897 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70898 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70899 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70900 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70901 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_8
  simp only [← c70896, ← c70897, ← c70898] at e_1148_8
  have e_1148_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_9
  simp only [← c70899, ← c70900, k2, ← c70901] at e_1148_9
  have hr := e_1148_9
  simp only [e_1148_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10050)) := by
  have c70902 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70903 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70904 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70905 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70906 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70907 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_10
  simp only [← c70902, ← c70903, ← c70904] at e_1148_10
  have e_1148_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_11
  simp only [← c70905, ← c70906, k2, ← c70907] at e_1148_11
  have hr := e_1148_11
  simp only [e_1148_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10051)) := by
  have c70908 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70909 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70910 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70911 := (publicBatchWrapper89_copies2215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70912 := (publicBatchWrapper89_copies2216 a h).1
  have c70913 := (publicBatchWrapper89_copies2216 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_12
  simp only [← c70908, ← c70909, ← c70910] at e_1148_12
  have e_1148_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_13
  simp only [← c70911, ← c70912, k2, ← c70913] at e_1148_13
  have hr := e_1148_13
  simp only [e_1148_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10052)) := by
  have c70914 := (publicBatchWrapper89_copies2216 a h).2.2.1
  have c70915 := (publicBatchWrapper89_copies2216 a h).2.2.2.1
  have c70916 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.1
  have c70917 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.1
  have c70918 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.1
  have c70919 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_14
  simp only [← c70914, ← c70915, ← c70916] at e_1148_14
  have e_1148_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_15
  simp only [← c70917, ← c70918, k2, ← c70919] at e_1148_15
  have hr := e_1148_15
  simp only [e_1148_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10053)) := by
  have c70920 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.1
  have c70921 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.1
  have c70922 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70923 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70924 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70925 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_16
  simp only [← c70920, ← c70921, ← c70922] at e_1148_16
  have e_1148_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_17
  simp only [← c70923, ← c70924, k2, ← c70925] at e_1148_17
  have hr := e_1148_17
  simp only [e_1148_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1148 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10054)) := by
  have c70926 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70927 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70928 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70929 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70930 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70931 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1148_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_18
  simp only [← c70926, ← c70927, ← c70928] at e_1148_18
  have e_1148_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1148 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_19
  simp only [← c70929, ← c70930, k2, ← c70931] at e_1148_19
  have hr := e_1148_19
  simp only [e_1148_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10055)) := by
  have c70932 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70933 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70934 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70935 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70936 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70937 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_0
  simp only [← c70932, ← c70933, ← c70934] at e_1149_0
  have e_1149_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_1
  simp only [← c70935, ← c70936, k2, ← c70937] at e_1149_1
  have hr := e_1149_1
  simp only [e_1149_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10056)) := by
  have c70938 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70939 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70940 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70941 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70942 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70943 := (publicBatchWrapper89_copies2216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_2
  simp only [← c70938, ← c70939, ← c70940] at e_1149_2
  have e_1149_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_3
  simp only [← c70941, ← c70942, k2, ← c70943] at e_1149_3
  have hr := e_1149_3
  simp only [e_1149_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10057)) := by
  have c70944 := (publicBatchWrapper89_copies2217 a h).1
  have c70945 := (publicBatchWrapper89_copies2217 a h).2.1
  have c70946 := (publicBatchWrapper89_copies2217 a h).2.2.1
  have c70947 := (publicBatchWrapper89_copies2217 a h).2.2.2.1
  have c70948 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.1
  have c70949 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_4
  simp only [← c70944, ← c70945, ← c70946] at e_1149_4
  have e_1149_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_5
  simp only [← c70947, ← c70948, k2, ← c70949] at e_1149_5
  have hr := e_1149_5
  simp only [e_1149_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10058)) := by
  have c70950 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.1
  have c70951 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.1
  have c70952 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.1
  have c70953 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.1
  have c70954 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70955 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_6
  simp only [← c70950, ← c70951, ← c70952] at e_1149_6
  have e_1149_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_7
  simp only [← c70953, ← c70954, k2, ← c70955] at e_1149_7
  have hr := e_1149_7
  simp only [e_1149_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10059)) := by
  have c70956 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70957 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70958 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70959 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70960 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70961 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_8
  simp only [← c70956, ← c70957, ← c70958] at e_1149_8
  have e_1149_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_9
  simp only [← c70959, ← c70960, k2, ← c70961] at e_1149_9
  have hr := e_1149_9
  simp only [e_1149_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10060)) := by
  have c70962 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70963 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70964 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70965 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70966 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70967 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_10
  simp only [← c70962, ← c70963, ← c70964] at e_1149_10
  have e_1149_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_11
  simp only [← c70965, ← c70966, k2, ← c70967] at e_1149_11
  have hr := e_1149_11
  simp only [e_1149_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10061)) := by
  have c70968 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70969 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70970 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70971 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70972 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70973 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_12
  simp only [← c70968, ← c70969, ← c70970] at e_1149_12
  have e_1149_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_13
  simp only [← c70971, ← c70972, k2, ← c70973] at e_1149_13
  have hr := e_1149_13
  simp only [e_1149_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10062)) := by
  have c70974 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70975 := (publicBatchWrapper89_copies2217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c70976 := (publicBatchWrapper89_copies2218 a h).1
  have c70977 := (publicBatchWrapper89_copies2218 a h).2.1
  have c70978 := (publicBatchWrapper89_copies2218 a h).2.2.1
  have c70979 := (publicBatchWrapper89_copies2218 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_14
  simp only [← c70974, ← c70975, ← c70976] at e_1149_14
  have e_1149_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_15
  simp only [← c70977, ← c70978, k2, ← c70979] at e_1149_15
  have hr := e_1149_15
  simp only [e_1149_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10063)) := by
  have c70980 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.1
  have c70981 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.1
  have c70982 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.1
  have c70983 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.1
  have c70984 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.1
  have c70985 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_16
  simp only [← c70980, ← c70981, ← c70982] at e_1149_16
  have e_1149_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_17
  simp only [← c70983, ← c70984, k2, ← c70985] at e_1149_17
  have hr := e_1149_17
  simp only [e_1149_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1149 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10064)) := by
  have c70986 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.1
  have c70987 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c70988 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70989 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70990 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70991 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1149_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_18
  simp only [← c70986, ← c70987, ← c70988] at e_1149_18
  have e_1149_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1149 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_19
  simp only [← c70989, ← c70990, k2, ← c70991] at e_1149_19
  have hr := e_1149_19
  simp only [e_1149_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10065)) := by
  have c70992 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70993 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70994 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70995 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70996 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70997 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_0
  simp only [← c70992, ← c70993, ← c70994] at e_1150_0
  have e_1150_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_1
  simp only [← c70995, ← c70996, k2, ← c70997] at e_1150_1
  have hr := e_1150_1
  simp only [e_1150_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10066)) := by
  have c70998 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c70999 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71000 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71001 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71002 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71003 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_2
  simp only [← c70998, ← c70999, ← c71000] at e_1150_2
  have e_1150_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_3
  simp only [← c71001, ← c71002, k2, ← c71003] at e_1150_3
  have hr := e_1150_3
  simp only [e_1150_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10067)) := by
  have c71004 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71005 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71006 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71007 := (publicBatchWrapper89_copies2218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71008 := (publicBatchWrapper89_copies2219 a h).1
  have c71009 := (publicBatchWrapper89_copies2219 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_4
  simp only [← c71004, ← c71005, ← c71006] at e_1150_4
  have e_1150_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_5
  simp only [← c71007, ← c71008, k2, ← c71009] at e_1150_5
  have hr := e_1150_5
  simp only [e_1150_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10068)) := by
  have c71010 := (publicBatchWrapper89_copies2219 a h).2.2.1
  have c71011 := (publicBatchWrapper89_copies2219 a h).2.2.2.1
  have c71012 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.1
  have c71013 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.1
  have c71014 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.1
  have c71015 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_6
  simp only [← c71010, ← c71011, ← c71012] at e_1150_6
  have e_1150_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_7
  simp only [← c71013, ← c71014, k2, ← c71015] at e_1150_7
  have hr := e_1150_7
  simp only [e_1150_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10069)) := by
  have c71016 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.1
  have c71017 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.1
  have c71018 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71019 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71020 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71021 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_8
  simp only [← c71016, ← c71017, ← c71018] at e_1150_8
  have e_1150_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_9
  simp only [← c71019, ← c71020, k2, ← c71021] at e_1150_9
  have hr := e_1150_9
  simp only [e_1150_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10070)) := by
  have c71022 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71023 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71024 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71025 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71026 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71027 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_10
  simp only [← c71022, ← c71023, ← c71024] at e_1150_10
  have e_1150_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_11
  simp only [← c71025, ← c71026, k2, ← c71027] at e_1150_11
  have hr := e_1150_11
  simp only [e_1150_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10071)) := by
  have c71028 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71029 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71030 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71031 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71032 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71033 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_12
  simp only [← c71028, ← c71029, ← c71030] at e_1150_12
  have e_1150_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_13
  simp only [← c71031, ← c71032, k2, ← c71033] at e_1150_13
  have hr := e_1150_13
  simp only [e_1150_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10072)) := by
  have c71034 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71035 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71036 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71037 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71038 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71039 := (publicBatchWrapper89_copies2219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_14
  simp only [← c71034, ← c71035, ← c71036] at e_1150_14
  have e_1150_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_15
  simp only [← c71037, ← c71038, k2, ← c71039] at e_1150_15
  have hr := e_1150_15
  simp only [e_1150_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10073)) := by
  have c71040 := (publicBatchWrapper89_copies2220 a h).1
  have c71041 := (publicBatchWrapper89_copies2220 a h).2.1
  have c71042 := (publicBatchWrapper89_copies2220 a h).2.2.1
  have c71043 := (publicBatchWrapper89_copies2220 a h).2.2.2.1
  have c71044 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.1
  have c71045 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_16
  simp only [← c71040, ← c71041, ← c71042] at e_1150_16
  have e_1150_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_17
  simp only [← c71043, ← c71044, k2, ← c71045] at e_1150_17
  have hr := e_1150_17
  simp only [e_1150_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1150 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10074)) := by
  have c71046 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.1
  have c71047 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.1
  have c71048 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.1
  have c71049 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.1
  have c71050 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71051 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1150_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_18
  simp only [← c71046, ← c71047, ← c71048] at e_1150_18
  have e_1150_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1150 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_19
  simp only [← c71049, ← c71050, k2, ← c71051] at e_1150_19
  have hr := e_1150_19
  simp only [e_1150_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10075)) := by
  have c71052 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71053 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71054 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71055 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71056 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71057 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_0
  simp only [← c71052, ← c71053, ← c71054] at e_1151_0
  have e_1151_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_1
  simp only [← c71055, ← c71056, k2, ← c71057] at e_1151_1
  have hr := e_1151_1
  simp only [e_1151_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10076)) := by
  have c71058 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71059 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71060 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71061 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71062 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71063 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_2
  simp only [← c71058, ← c71059, ← c71060] at e_1151_2
  have e_1151_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_3
  simp only [← c71061, ← c71062, k2, ← c71063] at e_1151_3
  have hr := e_1151_3
  simp only [e_1151_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10077)) := by
  have c71064 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71065 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71066 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71067 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71068 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71069 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_4
  simp only [← c71064, ← c71065, ← c71066] at e_1151_4
  have e_1151_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_5
  simp only [← c71067, ← c71068, k2, ← c71069] at e_1151_5
  have hr := e_1151_5
  simp only [e_1151_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10078)) := by
  have c71070 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71071 := (publicBatchWrapper89_copies2220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71072 := (publicBatchWrapper89_copies2221 a h).1
  have c71073 := (publicBatchWrapper89_copies2221 a h).2.1
  have c71074 := (publicBatchWrapper89_copies2221 a h).2.2.1
  have c71075 := (publicBatchWrapper89_copies2221 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_6
  simp only [← c71070, ← c71071, ← c71072] at e_1151_6
  have e_1151_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_7
  simp only [← c71073, ← c71074, k2, ← c71075] at e_1151_7
  have hr := e_1151_7
  simp only [e_1151_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10079)) := by
  have c71076 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.1
  have c71077 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.1
  have c71078 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.1
  have c71079 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.1
  have c71080 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.1
  have c71081 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_8
  simp only [← c71076, ← c71077, ← c71078] at e_1151_8
  have e_1151_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_9
  simp only [← c71079, ← c71080, k2, ← c71081] at e_1151_9
  have hr := e_1151_9
  simp only [e_1151_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10080)) := by
  have c71082 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71083 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71084 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71085 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71086 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71087 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_10
  simp only [← c71082, ← c71083, ← c71084] at e_1151_10
  have e_1151_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_11
  simp only [← c71085, ← c71086, k2, ← c71087] at e_1151_11
  have hr := e_1151_11
  simp only [e_1151_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10081)) := by
  have c71088 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71089 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71090 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71091 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71092 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71093 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_12
  simp only [← c71088, ← c71089, ← c71090] at e_1151_12
  have e_1151_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_13
  simp only [← c71091, ← c71092, k2, ← c71093] at e_1151_13
  have hr := e_1151_13
  simp only [e_1151_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10082)) := by
  have c71094 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71095 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71096 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71097 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71098 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71099 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_14
  simp only [← c71094, ← c71095, ← c71096] at e_1151_14
  have e_1151_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_15
  simp only [← c71097, ← c71098, k2, ← c71099] at e_1151_15
  have hr := e_1151_15
  simp only [e_1151_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10083)) := by
  have c71100 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71101 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71102 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71103 := (publicBatchWrapper89_copies2221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71104 := (publicBatchWrapper89_copies2222 a h).1
  have c71105 := (publicBatchWrapper89_copies2222 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_16
  simp only [← c71100, ← c71101, ← c71102] at e_1151_16
  have e_1151_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_17
  simp only [← c71103, ← c71104, k2, ← c71105] at e_1151_17
  have hr := e_1151_17
  simp only [e_1151_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1151 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10084)) := by
  have c71106 := (publicBatchWrapper89_copies2222 a h).2.2.1
  have c71107 := (publicBatchWrapper89_copies2222 a h).2.2.2.1
  have c71108 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.1
  have c71109 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.1
  have c71110 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.1
  have c71111 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1151_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_18
  simp only [← c71106, ← c71107, ← c71108] at e_1151_18
  have e_1151_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1151 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_19
  simp only [← c71109, ← c71110, k2, ← c71111] at e_1151_19
  have hr := e_1151_19
  simp only [e_1151_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10085)) := by
  have c71112 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.1
  have c71113 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.1
  have c71114 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71115 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71116 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71117 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_0
  simp only [← c71112, ← c71113, ← c71114] at e_1152_0
  have e_1152_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_1
  simp only [← c71115, ← c71116, k2, ← c71117] at e_1152_1
  have hr := e_1152_1
  simp only [e_1152_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10086)) := by
  have c71118 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71119 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71120 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71121 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71122 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71123 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_2
  simp only [← c71118, ← c71119, ← c71120] at e_1152_2
  have e_1152_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_3
  simp only [← c71121, ← c71122, k2, ← c71123] at e_1152_3
  have hr := e_1152_3
  simp only [e_1152_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10087)) := by
  have c71124 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71125 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71126 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71127 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71128 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71129 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_4
  simp only [← c71124, ← c71125, ← c71126] at e_1152_4
  have e_1152_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_5
  simp only [← c71127, ← c71128, k2, ← c71129] at e_1152_5
  have hr := e_1152_5
  simp only [e_1152_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10088)) := by
  have c71130 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71131 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71132 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71133 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71134 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71135 := (publicBatchWrapper89_copies2222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_6
  simp only [← c71130, ← c71131, ← c71132] at e_1152_6
  have e_1152_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_7
  simp only [← c71133, ← c71134, k2, ← c71135] at e_1152_7
  have hr := e_1152_7
  simp only [e_1152_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10089)) := by
  have c71136 := (publicBatchWrapper89_copies2223 a h).1
  have c71137 := (publicBatchWrapper89_copies2223 a h).2.1
  have c71138 := (publicBatchWrapper89_copies2223 a h).2.2.1
  have c71139 := (publicBatchWrapper89_copies2223 a h).2.2.2.1
  have c71140 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.1
  have c71141 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_8
  simp only [← c71136, ← c71137, ← c71138] at e_1152_8
  have e_1152_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_9
  simp only [← c71139, ← c71140, k2, ← c71141] at e_1152_9
  have hr := e_1152_9
  simp only [e_1152_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10090)) := by
  have c71142 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.1
  have c71143 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.1
  have c71144 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.1
  have c71145 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.1
  have c71146 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71147 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_10
  simp only [← c71142, ← c71143, ← c71144] at e_1152_10
  have e_1152_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_11
  simp only [← c71145, ← c71146, k2, ← c71147] at e_1152_11
  have hr := e_1152_11
  simp only [e_1152_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10091)) := by
  have c71148 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71149 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71150 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71151 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71152 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71153 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_12
  simp only [← c71148, ← c71149, ← c71150] at e_1152_12
  have e_1152_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_13
  simp only [← c71151, ← c71152, k2, ← c71153] at e_1152_13
  have hr := e_1152_13
  simp only [e_1152_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10092)) := by
  have c71154 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71155 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71156 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71157 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71158 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71159 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_14
  simp only [← c71154, ← c71155, ← c71156] at e_1152_14
  have e_1152_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_15
  simp only [← c71157, ← c71158, k2, ← c71159] at e_1152_15
  have hr := e_1152_15
  simp only [e_1152_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10093)) := by
  have c71160 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71161 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71162 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71163 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71164 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71165 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_16
  simp only [← c71160, ← c71161, ← c71162] at e_1152_16
  have e_1152_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_17
  simp only [← c71163, ← c71164, k2, ← c71165] at e_1152_17
  have hr := e_1152_17
  simp only [e_1152_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1152 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10094)) := by
  have c71166 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71167 := (publicBatchWrapper89_copies2223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71168 := (publicBatchWrapper89_copies2224 a h).1
  have c71169 := (publicBatchWrapper89_copies2224 a h).2.1
  have c71170 := (publicBatchWrapper89_copies2224 a h).2.2.1
  have c71171 := (publicBatchWrapper89_copies2224 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1152_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_18
  simp only [← c71166, ← c71167, ← c71168] at e_1152_18
  have e_1152_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1152 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_19
  simp only [← c71169, ← c71170, k2, ← c71171] at e_1152_19
  have hr := e_1152_19
  simp only [e_1152_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10095)) := by
  have c71172 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.1
  have c71173 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.1
  have c71174 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.1
  have c71175 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.1
  have c71176 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.1
  have c71177 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_0
  simp only [← c71172, ← c71173, ← c71174] at e_1153_0
  have e_1153_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_1
  simp only [← c71175, ← c71176, k2, ← c71177] at e_1153_1
  have hr := e_1153_1
  simp only [e_1153_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10096)) := by
  have c71178 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71179 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71180 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71181 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71182 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71183 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_2
  simp only [← c71178, ← c71179, ← c71180] at e_1153_2
  have e_1153_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_3
  simp only [← c71181, ← c71182, k2, ← c71183] at e_1153_3
  have hr := e_1153_3
  simp only [e_1153_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10097)) := by
  have c71184 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71185 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71186 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71187 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71188 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71189 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_4
  simp only [← c71184, ← c71185, ← c71186] at e_1153_4
  have e_1153_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_5
  simp only [← c71187, ← c71188, k2, ← c71189] at e_1153_5
  have hr := e_1153_5
  simp only [e_1153_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10098)) := by
  have c71190 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71191 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71192 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71193 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71194 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71195 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_6
  simp only [← c71190, ← c71191, ← c71192] at e_1153_6
  have e_1153_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_7
  simp only [← c71193, ← c71194, k2, ← c71195] at e_1153_7
  have hr := e_1153_7
  simp only [e_1153_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10099)) := by
  have c71196 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71197 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71198 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71199 := (publicBatchWrapper89_copies2224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71200 := (publicBatchWrapper89_copies2225 a h).1
  have c71201 := (publicBatchWrapper89_copies2225 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_8
  simp only [← c71196, ← c71197, ← c71198] at e_1153_8
  have e_1153_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_9
  simp only [← c71199, ← c71200, k2, ← c71201] at e_1153_9
  have hr := e_1153_9
  simp only [e_1153_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10100)) := by
  have c71202 := (publicBatchWrapper89_copies2225 a h).2.2.1
  have c71203 := (publicBatchWrapper89_copies2225 a h).2.2.2.1
  have c71204 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.1
  have c71205 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.1
  have c71206 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.1
  have c71207 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_10
  simp only [← c71202, ← c71203, ← c71204] at e_1153_10
  have e_1153_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_11
  simp only [← c71205, ← c71206, k2, ← c71207] at e_1153_11
  have hr := e_1153_11
  simp only [e_1153_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10101)) := by
  have c71208 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.1
  have c71209 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.1
  have c71210 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71211 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71212 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71213 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_12
  simp only [← c71208, ← c71209, ← c71210] at e_1153_12
  have e_1153_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_13
  simp only [← c71211, ← c71212, k2, ← c71213] at e_1153_13
  have hr := e_1153_13
  simp only [e_1153_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10102)) := by
  have c71214 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71215 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71216 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71217 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71218 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71219 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_14
  simp only [← c71214, ← c71215, ← c71216] at e_1153_14
  have e_1153_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_15
  simp only [← c71217, ← c71218, k2, ← c71219] at e_1153_15
  have hr := e_1153_15
  simp only [e_1153_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10103)) := by
  have c71220 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71221 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71222 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71223 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71224 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71225 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_16
  simp only [← c71220, ← c71221, ← c71222] at e_1153_16
  have e_1153_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_17
  simp only [← c71223, ← c71224, k2, ← c71225] at e_1153_17
  have hr := e_1153_17
  simp only [e_1153_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1153 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10104)) := by
  have c71226 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71227 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71228 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71229 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71230 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71231 := (publicBatchWrapper89_copies2225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1153_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_18
  simp only [← c71226, ← c71227, ← c71228] at e_1153_18
  have e_1153_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1153 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_19
  simp only [← c71229, ← c71230, k2, ← c71231] at e_1153_19
  have hr := e_1153_19
  simp only [e_1153_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10105)) := by
  have c71232 := (publicBatchWrapper89_copies2226 a h).1
  have c71233 := (publicBatchWrapper89_copies2226 a h).2.1
  have c71234 := (publicBatchWrapper89_copies2226 a h).2.2.1
  have c71235 := (publicBatchWrapper89_copies2226 a h).2.2.2.1
  have c71236 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.1
  have c71237 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_0
  simp only [← c71232, ← c71233, ← c71234] at e_1154_0
  have e_1154_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_1
  simp only [← c71235, ← c71236, k2, ← c71237] at e_1154_1
  have hr := e_1154_1
  simp only [e_1154_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10106)) := by
  have c71238 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.1
  have c71239 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.1
  have c71240 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.1
  have c71241 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.1
  have c71242 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71243 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_2
  simp only [← c71238, ← c71239, ← c71240] at e_1154_2
  have e_1154_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_3
  simp only [← c71241, ← c71242, k2, ← c71243] at e_1154_3
  have hr := e_1154_3
  simp only [e_1154_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10107)) := by
  have c71244 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71245 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71246 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71247 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71248 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71249 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_4
  simp only [← c71244, ← c71245, ← c71246] at e_1154_4
  have e_1154_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_5
  simp only [← c71247, ← c71248, k2, ← c71249] at e_1154_5
  have hr := e_1154_5
  simp only [e_1154_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10108)) := by
  have c71250 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71251 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71252 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71253 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71254 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71255 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_6
  simp only [← c71250, ← c71251, ← c71252] at e_1154_6
  have e_1154_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_7
  simp only [← c71253, ← c71254, k2, ← c71255] at e_1154_7
  have hr := e_1154_7
  simp only [e_1154_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10109)) := by
  have c71256 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71257 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71258 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71259 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71260 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71261 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_8
  simp only [← c71256, ← c71257, ← c71258] at e_1154_8
  have e_1154_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_9
  simp only [← c71259, ← c71260, k2, ← c71261] at e_1154_9
  have hr := e_1154_9
  simp only [e_1154_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10110)) := by
  have c71262 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71263 := (publicBatchWrapper89_copies2226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71264 := (publicBatchWrapper89_copies2227 a h).1
  have c71265 := (publicBatchWrapper89_copies2227 a h).2.1
  have c71266 := (publicBatchWrapper89_copies2227 a h).2.2.1
  have c71267 := (publicBatchWrapper89_copies2227 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_10
  simp only [← c71262, ← c71263, ← c71264] at e_1154_10
  have e_1154_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_11
  simp only [← c71265, ← c71266, k2, ← c71267] at e_1154_11
  have hr := e_1154_11
  simp only [e_1154_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10111)) := by
  have c71268 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.1
  have c71269 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.1
  have c71270 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.1
  have c71271 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.1
  have c71272 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.1
  have c71273 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_12
  simp only [← c71268, ← c71269, ← c71270] at e_1154_12
  have e_1154_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_13
  simp only [← c71271, ← c71272, k2, ← c71273] at e_1154_13
  have hr := e_1154_13
  simp only [e_1154_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10112)) := by
  have c71274 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71275 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71276 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71277 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71278 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71279 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_14
  simp only [← c71274, ← c71275, ← c71276] at e_1154_14
  have e_1154_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_15
  simp only [← c71277, ← c71278, k2, ← c71279] at e_1154_15
  have hr := e_1154_15
  simp only [e_1154_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10113)) := by
  have c71280 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71281 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71282 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71283 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71284 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71285 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_16
  simp only [← c71280, ← c71281, ← c71282] at e_1154_16
  have e_1154_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_17
  simp only [← c71283, ← c71284, k2, ← c71285] at e_1154_17
  have hr := e_1154_17
  simp only [e_1154_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1154 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10114)) := by
  have c71286 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71287 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71288 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71289 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71290 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71291 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1154_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_18
  simp only [← c71286, ← c71287, ← c71288] at e_1154_18
  have e_1154_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1154 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_19
  simp only [← c71289, ← c71290, k2, ← c71291] at e_1154_19
  have hr := e_1154_19
  simp only [e_1154_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10115)) := by
  have c71292 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71293 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71294 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71295 := (publicBatchWrapper89_copies2227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71296 := (publicBatchWrapper89_copies2228 a h).1
  have c71297 := (publicBatchWrapper89_copies2228 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_0
  simp only [← c71292, ← c71293, ← c71294] at e_1155_0
  have e_1155_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_1
  simp only [← c71295, ← c71296, k2, ← c71297] at e_1155_1
  have hr := e_1155_1
  simp only [e_1155_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10116)) := by
  have c71298 := (publicBatchWrapper89_copies2228 a h).2.2.1
  have c71299 := (publicBatchWrapper89_copies2228 a h).2.2.2.1
  have c71300 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.1
  have c71301 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.1
  have c71302 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.1
  have c71303 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_2
  simp only [← c71298, ← c71299, ← c71300] at e_1155_2
  have e_1155_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_3
  simp only [← c71301, ← c71302, k2, ← c71303] at e_1155_3
  have hr := e_1155_3
  simp only [e_1155_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10117)) := by
  have c71304 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.1
  have c71305 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.1
  have c71306 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71307 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71308 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71309 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_4
  simp only [← c71304, ← c71305, ← c71306] at e_1155_4
  have e_1155_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_5
  simp only [← c71307, ← c71308, k2, ← c71309] at e_1155_5
  have hr := e_1155_5
  simp only [e_1155_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10118)) := by
  have c71310 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71311 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71312 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71313 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71314 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71315 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_6
  simp only [← c71310, ← c71311, ← c71312] at e_1155_6
  have e_1155_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_7
  simp only [← c71313, ← c71314, k2, ← c71315] at e_1155_7
  have hr := e_1155_7
  simp only [e_1155_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10119)) := by
  have c71316 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71317 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71318 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71319 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71320 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71321 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_8
  simp only [← c71316, ← c71317, ← c71318] at e_1155_8
  have e_1155_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_9
  simp only [← c71319, ← c71320, k2, ← c71321] at e_1155_9
  have hr := e_1155_9
  simp only [e_1155_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10120)) := by
  have c71322 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71323 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71324 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71325 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71326 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71327 := (publicBatchWrapper89_copies2228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_10
  simp only [← c71322, ← c71323, ← c71324] at e_1155_10
  have e_1155_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_11
  simp only [← c71325, ← c71326, k2, ← c71327] at e_1155_11
  have hr := e_1155_11
  simp only [e_1155_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10121)) := by
  have c71328 := (publicBatchWrapper89_copies2229 a h).1
  have c71329 := (publicBatchWrapper89_copies2229 a h).2.1
  have c71330 := (publicBatchWrapper89_copies2229 a h).2.2.1
  have c71331 := (publicBatchWrapper89_copies2229 a h).2.2.2.1
  have c71332 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.1
  have c71333 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_12
  simp only [← c71328, ← c71329, ← c71330] at e_1155_12
  have e_1155_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_13
  simp only [← c71331, ← c71332, k2, ← c71333] at e_1155_13
  have hr := e_1155_13
  simp only [e_1155_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10122)) := by
  have c71334 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.1
  have c71335 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.1
  have c71336 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.1
  have c71337 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.1
  have c71338 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71339 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_14
  simp only [← c71334, ← c71335, ← c71336] at e_1155_14
  have e_1155_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_15
  simp only [← c71337, ← c71338, k2, ← c71339] at e_1155_15
  have hr := e_1155_15
  simp only [e_1155_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10123)) := by
  have c71340 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71341 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71342 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71343 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71344 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71345 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_16
  simp only [← c71340, ← c71341, ← c71342] at e_1155_16
  have e_1155_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_17
  simp only [← c71343, ← c71344, k2, ← c71345] at e_1155_17
  have hr := e_1155_17
  simp only [e_1155_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1155 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10124)) := by
  have c71346 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71347 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71348 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71349 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71350 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71351 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1155_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_18
  simp only [← c71346, ← c71347, ← c71348] at e_1155_18
  have e_1155_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1155 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_19
  simp only [← c71349, ← c71350, k2, ← c71351] at e_1155_19
  have hr := e_1155_19
  simp only [e_1155_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10125)) := by
  have c71352 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71353 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71354 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71355 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71356 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71357 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_0
  simp only [← c71352, ← c71353, ← c71354] at e_1156_0
  have e_1156_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_1
  simp only [← c71355, ← c71356, k2, ← c71357] at e_1156_1
  have hr := e_1156_1
  simp only [e_1156_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10126)) := by
  have c71358 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71359 := (publicBatchWrapper89_copies2229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71360 := (publicBatchWrapper89_copies2230 a h).1
  have c71361 := (publicBatchWrapper89_copies2230 a h).2.1
  have c71362 := (publicBatchWrapper89_copies2230 a h).2.2.1
  have c71363 := (publicBatchWrapper89_copies2230 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_2
  simp only [← c71358, ← c71359, ← c71360] at e_1156_2
  have e_1156_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_3
  simp only [← c71361, ← c71362, k2, ← c71363] at e_1156_3
  have hr := e_1156_3
  simp only [e_1156_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10127)) := by
  have c71364 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.1
  have c71365 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.1
  have c71366 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.1
  have c71367 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.1
  have c71368 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.1
  have c71369 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_4
  simp only [← c71364, ← c71365, ← c71366] at e_1156_4
  have e_1156_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_5
  simp only [← c71367, ← c71368, k2, ← c71369] at e_1156_5
  have hr := e_1156_5
  simp only [e_1156_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10128)) := by
  have c71370 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71371 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71372 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71373 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71374 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71375 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_6
  simp only [← c71370, ← c71371, ← c71372] at e_1156_6
  have e_1156_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_7
  simp only [← c71373, ← c71374, k2, ← c71375] at e_1156_7
  have hr := e_1156_7
  simp only [e_1156_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10129)) := by
  have c71376 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71377 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71378 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71379 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71380 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71381 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_8
  simp only [← c71376, ← c71377, ← c71378] at e_1156_8
  have e_1156_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_9
  simp only [← c71379, ← c71380, k2, ← c71381] at e_1156_9
  have hr := e_1156_9
  simp only [e_1156_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10130)) := by
  have c71382 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71383 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71384 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71385 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71386 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71387 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_10
  simp only [← c71382, ← c71383, ← c71384] at e_1156_10
  have e_1156_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_11
  simp only [← c71385, ← c71386, k2, ← c71387] at e_1156_11
  have hr := e_1156_11
  simp only [e_1156_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10131)) := by
  have c71388 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71389 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71390 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71391 := (publicBatchWrapper89_copies2230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71392 := (publicBatchWrapper89_copies2231 a h).1
  have c71393 := (publicBatchWrapper89_copies2231 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_12
  simp only [← c71388, ← c71389, ← c71390] at e_1156_12
  have e_1156_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_13
  simp only [← c71391, ← c71392, k2, ← c71393] at e_1156_13
  have hr := e_1156_13
  simp only [e_1156_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10132)) := by
  have c71394 := (publicBatchWrapper89_copies2231 a h).2.2.1
  have c71395 := (publicBatchWrapper89_copies2231 a h).2.2.2.1
  have c71396 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.1
  have c71397 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.1
  have c71398 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.1
  have c71399 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_14
  simp only [← c71394, ← c71395, ← c71396] at e_1156_14
  have e_1156_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_15
  simp only [← c71397, ← c71398, k2, ← c71399] at e_1156_15
  have hr := e_1156_15
  simp only [e_1156_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10133)) := by
  have c71400 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.1
  have c71401 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.1
  have c71402 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71403 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71404 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71405 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_16
  simp only [← c71400, ← c71401, ← c71402] at e_1156_16
  have e_1156_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_17
  simp only [← c71403, ← c71404, k2, ← c71405] at e_1156_17
  have hr := e_1156_17
  simp only [e_1156_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1156 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10134)) := by
  have c71406 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71407 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71408 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71409 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71410 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71411 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1156_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_18
  simp only [← c71406, ← c71407, ← c71408] at e_1156_18
  have e_1156_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1156 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_19
  simp only [← c71409, ← c71410, k2, ← c71411] at e_1156_19
  have hr := e_1156_19
  simp only [e_1156_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10135)) := by
  have c71412 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71413 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71414 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71415 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71416 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71417 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_0
  simp only [← c71412, ← c71413, ← c71414] at e_1157_0
  have e_1157_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_1
  simp only [← c71415, ← c71416, k2, ← c71417] at e_1157_1
  have hr := e_1157_1
  simp only [e_1157_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10136)) := by
  have c71418 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71419 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71420 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71421 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71422 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71423 := (publicBatchWrapper89_copies2231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_2
  simp only [← c71418, ← c71419, ← c71420] at e_1157_2
  have e_1157_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_3
  simp only [← c71421, ← c71422, k2, ← c71423] at e_1157_3
  have hr := e_1157_3
  simp only [e_1157_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10137)) := by
  have c71424 := (publicBatchWrapper89_copies2232 a h).1
  have c71425 := (publicBatchWrapper89_copies2232 a h).2.1
  have c71426 := (publicBatchWrapper89_copies2232 a h).2.2.1
  have c71427 := (publicBatchWrapper89_copies2232 a h).2.2.2.1
  have c71428 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.1
  have c71429 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_4
  simp only [← c71424, ← c71425, ← c71426] at e_1157_4
  have e_1157_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_5
  simp only [← c71427, ← c71428, k2, ← c71429] at e_1157_5
  have hr := e_1157_5
  simp only [e_1157_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10138)) := by
  have c71430 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.1
  have c71431 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.1
  have c71432 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.1
  have c71433 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.1
  have c71434 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71435 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_6
  simp only [← c71430, ← c71431, ← c71432] at e_1157_6
  have e_1157_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_7
  simp only [← c71433, ← c71434, k2, ← c71435] at e_1157_7
  have hr := e_1157_7
  simp only [e_1157_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10139)) := by
  have c71436 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71437 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71438 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71439 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71440 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71441 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_8
  simp only [← c71436, ← c71437, ← c71438] at e_1157_8
  have e_1157_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_9
  simp only [← c71439, ← c71440, k2, ← c71441] at e_1157_9
  have hr := e_1157_9
  simp only [e_1157_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10140)) := by
  have c71442 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71443 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71444 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71445 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71446 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71447 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_10
  simp only [← c71442, ← c71443, ← c71444] at e_1157_10
  have e_1157_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_11
  simp only [← c71445, ← c71446, k2, ← c71447] at e_1157_11
  have hr := e_1157_11
  simp only [e_1157_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10141)) := by
  have c71448 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71449 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71450 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71451 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71452 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71453 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_12
  simp only [← c71448, ← c71449, ← c71450] at e_1157_12
  have e_1157_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_13
  simp only [← c71451, ← c71452, k2, ← c71453] at e_1157_13
  have hr := e_1157_13
  simp only [e_1157_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10142)) := by
  have c71454 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71455 := (publicBatchWrapper89_copies2232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71456 := (publicBatchWrapper89_copies2233 a h).1
  have c71457 := (publicBatchWrapper89_copies2233 a h).2.1
  have c71458 := (publicBatchWrapper89_copies2233 a h).2.2.1
  have c71459 := (publicBatchWrapper89_copies2233 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_14
  simp only [← c71454, ← c71455, ← c71456] at e_1157_14
  have e_1157_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_15
  simp only [← c71457, ← c71458, k2, ← c71459] at e_1157_15
  have hr := e_1157_15
  simp only [e_1157_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10143)) := by
  have c71460 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.1
  have c71461 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.1
  have c71462 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.1
  have c71463 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.1
  have c71464 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.1
  have c71465 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_16
  simp only [← c71460, ← c71461, ← c71462] at e_1157_16
  have e_1157_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_17
  simp only [← c71463, ← c71464, k2, ← c71465] at e_1157_17
  have hr := e_1157_17
  simp only [e_1157_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1157 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10144)) := by
  have c71466 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71467 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71468 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71469 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71470 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71471 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1157_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_18
  simp only [← c71466, ← c71467, ← c71468] at e_1157_18
  have e_1157_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1157 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_19
  simp only [← c71469, ← c71470, k2, ← c71471] at e_1157_19
  have hr := e_1157_19
  simp only [e_1157_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10145)) := by
  have c71472 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71473 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71474 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71475 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71476 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71477 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_0
  simp only [← c71472, ← c71473, ← c71474] at e_1158_0
  have e_1158_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_1
  simp only [← c71475, ← c71476, k2, ← c71477] at e_1158_1
  have hr := e_1158_1
  simp only [e_1158_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10146)) := by
  have c71478 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71479 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71480 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71481 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71482 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71483 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_2
  simp only [← c71478, ← c71479, ← c71480] at e_1158_2
  have e_1158_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_3
  simp only [← c71481, ← c71482, k2, ← c71483] at e_1158_3
  have hr := e_1158_3
  simp only [e_1158_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10147)) := by
  have c71484 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71485 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71486 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71487 := (publicBatchWrapper89_copies2233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71488 := (publicBatchWrapper89_copies2234 a h).1
  have c71489 := (publicBatchWrapper89_copies2234 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_4
  simp only [← c71484, ← c71485, ← c71486] at e_1158_4
  have e_1158_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_5
  simp only [← c71487, ← c71488, k2, ← c71489] at e_1158_5
  have hr := e_1158_5
  simp only [e_1158_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10148)) := by
  have c71490 := (publicBatchWrapper89_copies2234 a h).2.2.1
  have c71491 := (publicBatchWrapper89_copies2234 a h).2.2.2.1
  have c71492 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.1
  have c71493 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.1
  have c71494 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.1
  have c71495 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_6
  simp only [← c71490, ← c71491, ← c71492] at e_1158_6
  have e_1158_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_7
  simp only [← c71493, ← c71494, k2, ← c71495] at e_1158_7
  have hr := e_1158_7
  simp only [e_1158_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10149)) := by
  have c71496 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.1
  have c71497 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.1
  have c71498 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71499 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71500 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71501 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_8
  simp only [← c71496, ← c71497, ← c71498] at e_1158_8
  have e_1158_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_9
  simp only [← c71499, ← c71500, k2, ← c71501] at e_1158_9
  have hr := e_1158_9
  simp only [e_1158_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10150)) := by
  have c71502 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71503 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71504 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71505 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71506 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71507 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_10
  simp only [← c71502, ← c71503, ← c71504] at e_1158_10
  have e_1158_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_11
  simp only [← c71505, ← c71506, k2, ← c71507] at e_1158_11
  have hr := e_1158_11
  simp only [e_1158_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10151)) := by
  have c71508 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71509 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71510 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71511 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71512 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71513 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_12
  simp only [← c71508, ← c71509, ← c71510] at e_1158_12
  have e_1158_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_13
  simp only [← c71511, ← c71512, k2, ← c71513] at e_1158_13
  have hr := e_1158_13
  simp only [e_1158_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10152)) := by
  have c71514 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71515 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71516 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71517 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71518 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71519 := (publicBatchWrapper89_copies2234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_14
  simp only [← c71514, ← c71515, ← c71516] at e_1158_14
  have e_1158_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_15
  simp only [← c71517, ← c71518, k2, ← c71519] at e_1158_15
  have hr := e_1158_15
  simp only [e_1158_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10153)) := by
  have c71520 := (publicBatchWrapper89_copies2235 a h).1
  have c71521 := (publicBatchWrapper89_copies2235 a h).2.1
  have c71522 := (publicBatchWrapper89_copies2235 a h).2.2.1
  have c71523 := (publicBatchWrapper89_copies2235 a h).2.2.2.1
  have c71524 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.1
  have c71525 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_16
  simp only [← c71520, ← c71521, ← c71522] at e_1158_16
  have e_1158_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_17
  simp only [← c71523, ← c71524, k2, ← c71525] at e_1158_17
  have hr := e_1158_17
  simp only [e_1158_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1158 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10154)) := by
  have c71526 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.1
  have c71527 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.1
  have c71528 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.1
  have c71529 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.1
  have c71530 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71531 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1158_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_18
  simp only [← c71526, ← c71527, ← c71528] at e_1158_18
  have e_1158_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1158 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_19
  simp only [← c71529, ← c71530, k2, ← c71531] at e_1158_19
  have hr := e_1158_19
  simp only [e_1158_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10155)) := by
  have c71532 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71533 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71534 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71535 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71536 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71537 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_0
  simp only [← c71532, ← c71533, ← c71534] at e_1159_0
  have e_1159_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_1
  simp only [← c71535, ← c71536, k2, ← c71537] at e_1159_1
  have hr := e_1159_1
  simp only [e_1159_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10156)) := by
  have c71538 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71539 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71540 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71541 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71542 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71543 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_2
  simp only [← c71538, ← c71539, ← c71540] at e_1159_2
  have e_1159_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_3
  simp only [← c71541, ← c71542, k2, ← c71543] at e_1159_3
  have hr := e_1159_3
  simp only [e_1159_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10157)) := by
  have c71544 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71545 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71546 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71547 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71548 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71549 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_4
  simp only [← c71544, ← c71545, ← c71546] at e_1159_4
  have e_1159_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_5
  simp only [← c71547, ← c71548, k2, ← c71549] at e_1159_5
  have hr := e_1159_5
  simp only [e_1159_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10158)) := by
  have c71550 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71551 := (publicBatchWrapper89_copies2235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71552 := (publicBatchWrapper89_copies2236 a h).1
  have c71553 := (publicBatchWrapper89_copies2236 a h).2.1
  have c71554 := (publicBatchWrapper89_copies2236 a h).2.2.1
  have c71555 := (publicBatchWrapper89_copies2236 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_6
  simp only [← c71550, ← c71551, ← c71552] at e_1159_6
  have e_1159_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_7
  simp only [← c71553, ← c71554, k2, ← c71555] at e_1159_7
  have hr := e_1159_7
  simp only [e_1159_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10159)) := by
  have c71556 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.1
  have c71557 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.1
  have c71558 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.1
  have c71559 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.1
  have c71560 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.1
  have c71561 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_8
  simp only [← c71556, ← c71557, ← c71558] at e_1159_8
  have e_1159_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_9
  simp only [← c71559, ← c71560, k2, ← c71561] at e_1159_9
  have hr := e_1159_9
  simp only [e_1159_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10160)) := by
  have c71562 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71563 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71564 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71565 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71566 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71567 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_10
  simp only [← c71562, ← c71563, ← c71564] at e_1159_10
  have e_1159_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_11
  simp only [← c71565, ← c71566, k2, ← c71567] at e_1159_11
  have hr := e_1159_11
  simp only [e_1159_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10161)) := by
  have c71568 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71569 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71570 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71571 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71572 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71573 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_12
  simp only [← c71568, ← c71569, ← c71570] at e_1159_12
  have e_1159_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_13
  simp only [← c71571, ← c71572, k2, ← c71573] at e_1159_13
  have hr := e_1159_13
  simp only [e_1159_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10162)) := by
  have c71574 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71575 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71576 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71577 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71578 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71579 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_14
  simp only [← c71574, ← c71575, ← c71576] at e_1159_14
  have e_1159_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_15
  simp only [← c71577, ← c71578, k2, ← c71579] at e_1159_15
  have hr := e_1159_15
  simp only [e_1159_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10163)) := by
  have c71580 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71581 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71582 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71583 := (publicBatchWrapper89_copies2236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71584 := (publicBatchWrapper89_copies2237 a h).1
  have c71585 := (publicBatchWrapper89_copies2237 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_16
  simp only [← c71580, ← c71581, ← c71582] at e_1159_16
  have e_1159_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_17
  simp only [← c71583, ← c71584, k2, ← c71585] at e_1159_17
  have hr := e_1159_17
  simp only [e_1159_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1159 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10164)) := by
  have c71586 := (publicBatchWrapper89_copies2237 a h).2.2.1
  have c71587 := (publicBatchWrapper89_copies2237 a h).2.2.2.1
  have c71588 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.1
  have c71589 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.1
  have c71590 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.1
  have c71591 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1159_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_18
  simp only [← c71586, ← c71587, ← c71588] at e_1159_18
  have e_1159_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1159 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_19
  simp only [← c71589, ← c71590, k2, ← c71591] at e_1159_19
  have hr := e_1159_19
  simp only [e_1159_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10165)) := by
  have c71592 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.1
  have c71593 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.1
  have c71594 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71595 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71596 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71597 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_0
  simp only [← c71592, ← c71593, ← c71594] at e_1160_0
  have e_1160_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_1
  simp only [← c71595, ← c71596, k2, ← c71597] at e_1160_1
  have hr := e_1160_1
  simp only [e_1160_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10166)) := by
  have c71598 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71599 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71600 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71601 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71602 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71603 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_2
  simp only [← c71598, ← c71599, ← c71600] at e_1160_2
  have e_1160_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_3
  simp only [← c71601, ← c71602, k2, ← c71603] at e_1160_3
  have hr := e_1160_3
  simp only [e_1160_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10167)) := by
  have c71604 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71605 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71606 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71607 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71608 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71609 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_4
  simp only [← c71604, ← c71605, ← c71606] at e_1160_4
  have e_1160_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_5
  simp only [← c71607, ← c71608, k2, ← c71609] at e_1160_5
  have hr := e_1160_5
  simp only [e_1160_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10168)) := by
  have c71610 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71611 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71612 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71613 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71614 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71615 := (publicBatchWrapper89_copies2237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_6
  simp only [← c71610, ← c71611, ← c71612] at e_1160_6
  have e_1160_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_7
  simp only [← c71613, ← c71614, k2, ← c71615] at e_1160_7
  have hr := e_1160_7
  simp only [e_1160_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10169)) := by
  have c71616 := (publicBatchWrapper89_copies2238 a h).1
  have c71617 := (publicBatchWrapper89_copies2238 a h).2.1
  have c71618 := (publicBatchWrapper89_copies2238 a h).2.2.1
  have c71619 := (publicBatchWrapper89_copies2238 a h).2.2.2.1
  have c71620 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.1
  have c71621 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_8
  simp only [← c71616, ← c71617, ← c71618] at e_1160_8
  have e_1160_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_9
  simp only [← c71619, ← c71620, k2, ← c71621] at e_1160_9
  have hr := e_1160_9
  simp only [e_1160_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10210)) := by
  have c71622 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.1
  have c71623 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.1
  have c71624 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.1
  have c71625 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.1
  have c71626 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71627 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_10
  simp only [← c71622, ← c71623, ← c71624] at e_1160_10
  have e_1160_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_11
  simp only [← c71625, ← c71626, k2, ← c71627] at e_1160_11
  have hr := e_1160_11
  simp only [e_1160_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10211)) := by
  have c71628 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71629 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71630 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71631 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71632 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71633 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_12
  simp only [← c71628, ← c71629, ← c71630] at e_1160_12
  have e_1160_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_13
  simp only [← c71631, ← c71632, k2, ← c71633] at e_1160_13
  have hr := e_1160_13
  simp only [e_1160_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10212)) := by
  have c71634 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71635 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71636 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71637 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71638 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71639 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_14
  simp only [← c71634, ← c71635, ← c71636] at e_1160_14
  have e_1160_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_15
  simp only [← c71637, ← c71638, k2, ← c71639] at e_1160_15
  have hr := e_1160_15
  simp only [e_1160_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10213)) := by
  have c71640 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71641 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71642 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71643 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71644 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71645 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_16
  simp only [← c71640, ← c71641, ← c71642] at e_1160_16
  have e_1160_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_17
  simp only [← c71643, ← c71644, k2, ← c71645] at e_1160_17
  have hr := e_1160_17
  simp only [e_1160_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1160 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10214)) := by
  have c71646 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71647 := (publicBatchWrapper89_copies2238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71648 := (publicBatchWrapper89_copies2239 a h).1
  have c71649 := (publicBatchWrapper89_copies2239 a h).2.1
  have c71650 := (publicBatchWrapper89_copies2239 a h).2.2.1
  have c71651 := (publicBatchWrapper89_copies2239 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1160_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_18
  simp only [← c71646, ← c71647, ← c71648] at e_1160_18
  have e_1160_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1160 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_19
  simp only [← c71649, ← c71650, k2, ← c71651] at e_1160_19
  have hr := e_1160_19
  simp only [e_1160_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10215)) := by
  have c71652 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.1
  have c71653 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.1
  have c71654 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.1
  have c71655 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.1
  have c71656 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.1
  have c71657 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_0
  simp only [← c71652, ← c71653, ← c71654] at e_1161_0
  have e_1161_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_1
  simp only [← c71655, ← c71656, k2, ← c71657] at e_1161_1
  have hr := e_1161_1
  simp only [e_1161_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10216)) := by
  have c71658 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71659 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71660 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71661 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71662 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71663 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_2
  simp only [← c71658, ← c71659, ← c71660] at e_1161_2
  have e_1161_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_3
  simp only [← c71661, ← c71662, k2, ← c71663] at e_1161_3
  have hr := e_1161_3
  simp only [e_1161_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10217)) := by
  have c71664 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71665 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71666 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71667 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71668 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71669 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_4
  simp only [← c71664, ← c71665, ← c71666] at e_1161_4
  have e_1161_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_5
  simp only [← c71667, ← c71668, k2, ← c71669] at e_1161_5
  have hr := e_1161_5
  simp only [e_1161_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10218)) := by
  have c71670 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71671 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71672 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71673 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71674 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71675 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_6
  simp only [← c71670, ← c71671, ← c71672] at e_1161_6
  have e_1161_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_7
  simp only [← c71673, ← c71674, k2, ← c71675] at e_1161_7
  have hr := e_1161_7
  simp only [e_1161_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10219)) := by
  have c71676 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71677 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71678 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71679 := (publicBatchWrapper89_copies2239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71680 := (publicBatchWrapper89_copies2240 a h).1
  have c71681 := (publicBatchWrapper89_copies2240 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_8
  simp only [← c71676, ← c71677, ← c71678] at e_1161_8
  have e_1161_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_9
  simp only [← c71679, ← c71680, k2, ← c71681] at e_1161_9
  have hr := e_1161_9
  simp only [e_1161_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10220)) := by
  have c71682 := (publicBatchWrapper89_copies2240 a h).2.2.1
  have c71683 := (publicBatchWrapper89_copies2240 a h).2.2.2.1
  have c71684 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.1
  have c71685 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.1
  have c71686 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.1
  have c71687 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_10
  simp only [← c71682, ← c71683, ← c71684] at e_1161_10
  have e_1161_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_11
  simp only [← c71685, ← c71686, k2, ← c71687] at e_1161_11
  have hr := e_1161_11
  simp only [e_1161_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10221)) := by
  have c71688 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.1
  have c71689 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.1
  have c71690 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71691 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71692 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71693 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_12
  simp only [← c71688, ← c71689, ← c71690] at e_1161_12
  have e_1161_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_13
  simp only [← c71691, ← c71692, k2, ← c71693] at e_1161_13
  have hr := e_1161_13
  simp only [e_1161_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10222)) := by
  have c71694 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71695 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71696 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71697 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71698 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71699 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_14
  simp only [← c71694, ← c71695, ← c71696] at e_1161_14
  have e_1161_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_15
  simp only [← c71697, ← c71698, k2, ← c71699] at e_1161_15
  have hr := e_1161_15
  simp only [e_1161_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10223)) := by
  have c71700 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71701 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71702 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71703 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71704 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71705 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_16
  simp only [← c71700, ← c71701, ← c71702] at e_1161_16
  have e_1161_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_17
  simp only [← c71703, ← c71704, k2, ← c71705] at e_1161_17
  have hr := e_1161_17
  simp only [e_1161_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1161 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10224)) := by
  have c71706 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71707 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71708 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71709 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71710 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71711 := (publicBatchWrapper89_copies2240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1161_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_18
  simp only [← c71706, ← c71707, ← c71708] at e_1161_18
  have e_1161_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1161 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_19
  simp only [← c71709, ← c71710, k2, ← c71711] at e_1161_19
  have hr := e_1161_19
  simp only [e_1161_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10225)) := by
  have c71712 := (publicBatchWrapper89_copies2241 a h).1
  have c71713 := (publicBatchWrapper89_copies2241 a h).2.1
  have c71714 := (publicBatchWrapper89_copies2241 a h).2.2.1
  have c71715 := (publicBatchWrapper89_copies2241 a h).2.2.2.1
  have c71716 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.1
  have c71717 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_0
  simp only [← c71712, ← c71713, ← c71714] at e_1162_0
  have e_1162_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_1
  simp only [← c71715, ← c71716, k2, ← c71717] at e_1162_1
  have hr := e_1162_1
  simp only [e_1162_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10226)) := by
  have c71718 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.1
  have c71719 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.1
  have c71720 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.1
  have c71721 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.1
  have c71722 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71723 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_2
  simp only [← c71718, ← c71719, ← c71720] at e_1162_2
  have e_1162_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_3
  simp only [← c71721, ← c71722, k2, ← c71723] at e_1162_3
  have hr := e_1162_3
  simp only [e_1162_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10227)) := by
  have c71724 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71725 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71726 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71727 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71728 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71729 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_4
  simp only [← c71724, ← c71725, ← c71726] at e_1162_4
  have e_1162_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_5
  simp only [← c71727, ← c71728, k2, ← c71729] at e_1162_5
  have hr := e_1162_5
  simp only [e_1162_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10228)) := by
  have c71730 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71731 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71732 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71733 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71734 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71735 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_6
  simp only [← c71730, ← c71731, ← c71732] at e_1162_6
  have e_1162_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_7
  simp only [← c71733, ← c71734, k2, ← c71735] at e_1162_7
  have hr := e_1162_7
  simp only [e_1162_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10229)) := by
  have c71736 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71737 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71738 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71739 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71740 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71741 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_8
  simp only [← c71736, ← c71737, ← c71738] at e_1162_8
  have e_1162_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_9
  simp only [← c71739, ← c71740, k2, ← c71741] at e_1162_9
  have hr := e_1162_9
  simp only [e_1162_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10230)) := by
  have c71742 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71743 := (publicBatchWrapper89_copies2241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71744 := (publicBatchWrapper89_copies2242 a h).1
  have c71745 := (publicBatchWrapper89_copies2242 a h).2.1
  have c71746 := (publicBatchWrapper89_copies2242 a h).2.2.1
  have c71747 := (publicBatchWrapper89_copies2242 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_10
  simp only [← c71742, ← c71743, ← c71744] at e_1162_10
  have e_1162_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_11
  simp only [← c71745, ← c71746, k2, ← c71747] at e_1162_11
  have hr := e_1162_11
  simp only [e_1162_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10231)) := by
  have c71748 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.1
  have c71749 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.1
  have c71750 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.1
  have c71751 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.1
  have c71752 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.1
  have c71753 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_12
  simp only [← c71748, ← c71749, ← c71750] at e_1162_12
  have e_1162_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_13
  simp only [← c71751, ← c71752, k2, ← c71753] at e_1162_13
  have hr := e_1162_13
  simp only [e_1162_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10232)) := by
  have c71754 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71755 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71756 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71757 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71758 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71759 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_14
  simp only [← c71754, ← c71755, ← c71756] at e_1162_14
  have e_1162_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_15
  simp only [← c71757, ← c71758, k2, ← c71759] at e_1162_15
  have hr := e_1162_15
  simp only [e_1162_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10233)) := by
  have c71760 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71761 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71762 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71763 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71764 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71765 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_16
  simp only [← c71760, ← c71761, ← c71762] at e_1162_16
  have e_1162_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_17
  simp only [← c71763, ← c71764, k2, ← c71765] at e_1162_17
  have hr := e_1162_17
  simp only [e_1162_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1162 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10234)) := by
  have c71766 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71767 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71768 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71769 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71770 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71771 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1162_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_18
  simp only [← c71766, ← c71767, ← c71768] at e_1162_18
  have e_1162_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1162 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_19
  simp only [← c71769, ← c71770, k2, ← c71771] at e_1162_19
  have hr := e_1162_19
  simp only [e_1162_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10235)) := by
  have c71772 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71773 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71774 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71775 := (publicBatchWrapper89_copies2242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71776 := (publicBatchWrapper89_copies2243 a h).1
  have c71777 := (publicBatchWrapper89_copies2243 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_0
  simp only [← c71772, ← c71773, ← c71774] at e_1163_0
  have e_1163_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_1
  simp only [← c71775, ← c71776, k2, ← c71777] at e_1163_1
  have hr := e_1163_1
  simp only [e_1163_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10236)) := by
  have c71778 := (publicBatchWrapper89_copies2243 a h).2.2.1
  have c71779 := (publicBatchWrapper89_copies2243 a h).2.2.2.1
  have c71780 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.1
  have c71781 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.1
  have c71782 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.1
  have c71783 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_2
  simp only [← c71778, ← c71779, ← c71780] at e_1163_2
  have e_1163_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_3
  simp only [← c71781, ← c71782, k2, ← c71783] at e_1163_3
  have hr := e_1163_3
  simp only [e_1163_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10237)) := by
  have c71784 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.1
  have c71785 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.1
  have c71786 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71787 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71788 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71789 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_4
  simp only [← c71784, ← c71785, ← c71786] at e_1163_4
  have e_1163_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_5
  simp only [← c71787, ← c71788, k2, ← c71789] at e_1163_5
  have hr := e_1163_5
  simp only [e_1163_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10238)) := by
  have c71790 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71791 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71792 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71793 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71794 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71795 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_6
  simp only [← c71790, ← c71791, ← c71792] at e_1163_6
  have e_1163_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_7
  simp only [← c71793, ← c71794, k2, ← c71795] at e_1163_7
  have hr := e_1163_7
  simp only [e_1163_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10239)) := by
  have c71796 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71797 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71798 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71799 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71800 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71801 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_8
  simp only [← c71796, ← c71797, ← c71798] at e_1163_8
  have e_1163_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_9
  simp only [← c71799, ← c71800, k2, ← c71801] at e_1163_9
  have hr := e_1163_9
  simp only [e_1163_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10240)) := by
  have c71802 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71803 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71804 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71805 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71806 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71807 := (publicBatchWrapper89_copies2243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_10
  simp only [← c71802, ← c71803, ← c71804] at e_1163_10
  have e_1163_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_11
  simp only [← c71805, ← c71806, k2, ← c71807] at e_1163_11
  have hr := e_1163_11
  simp only [e_1163_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10241)) := by
  have c71808 := (publicBatchWrapper89_copies2244 a h).1
  have c71809 := (publicBatchWrapper89_copies2244 a h).2.1
  have c71810 := (publicBatchWrapper89_copies2244 a h).2.2.1
  have c71811 := (publicBatchWrapper89_copies2244 a h).2.2.2.1
  have c71812 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.1
  have c71813 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_12
  simp only [← c71808, ← c71809, ← c71810] at e_1163_12
  have e_1163_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_13
  simp only [← c71811, ← c71812, k2, ← c71813] at e_1163_13
  have hr := e_1163_13
  simp only [e_1163_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10242)) := by
  have c71814 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.1
  have c71815 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.1
  have c71816 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.1
  have c71817 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.1
  have c71818 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71819 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_14
  simp only [← c71814, ← c71815, ← c71816] at e_1163_14
  have e_1163_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_15
  simp only [← c71817, ← c71818, k2, ← c71819] at e_1163_15
  have hr := e_1163_15
  simp only [e_1163_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10243)) := by
  have c71820 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71821 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71822 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71823 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71824 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71825 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_16
  simp only [← c71820, ← c71821, ← c71822] at e_1163_16
  have e_1163_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_17
  simp only [← c71823, ← c71824, k2, ← c71825] at e_1163_17
  have hr := e_1163_17
  simp only [e_1163_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1163 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10244)) := by
  have c71826 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71827 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71828 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71829 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71830 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71831 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1163_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_18
  simp only [← c71826, ← c71827, ← c71828] at e_1163_18
  have e_1163_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1163 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_19
  simp only [← c71829, ← c71830, k2, ← c71831] at e_1163_19
  have hr := e_1163_19
  simp only [e_1163_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10245)) := by
  have c71832 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71833 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71834 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71835 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71836 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71837 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_0
  simp only [← c71832, ← c71833, ← c71834] at e_1164_0
  have e_1164_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_1
  simp only [← c71835, ← c71836, k2, ← c71837] at e_1164_1
  have hr := e_1164_1
  simp only [e_1164_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10246)) := by
  have c71838 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71839 := (publicBatchWrapper89_copies2244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71840 := (publicBatchWrapper89_copies2245 a h).1
  have c71841 := (publicBatchWrapper89_copies2245 a h).2.1
  have c71842 := (publicBatchWrapper89_copies2245 a h).2.2.1
  have c71843 := (publicBatchWrapper89_copies2245 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_2
  simp only [← c71838, ← c71839, ← c71840] at e_1164_2
  have e_1164_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_3
  simp only [← c71841, ← c71842, k2, ← c71843] at e_1164_3
  have hr := e_1164_3
  simp only [e_1164_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10247)) := by
  have c71844 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.1
  have c71845 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.1
  have c71846 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.1
  have c71847 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.1
  have c71848 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.1
  have c71849 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_4
  simp only [← c71844, ← c71845, ← c71846] at e_1164_4
  have e_1164_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_5
  simp only [← c71847, ← c71848, k2, ← c71849] at e_1164_5
  have hr := e_1164_5
  simp only [e_1164_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10248)) := by
  have c71850 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71851 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71852 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71853 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71854 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71855 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_6
  simp only [← c71850, ← c71851, ← c71852] at e_1164_6
  have e_1164_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_7
  simp only [← c71853, ← c71854, k2, ← c71855] at e_1164_7
  have hr := e_1164_7
  simp only [e_1164_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10249)) := by
  have c71856 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71857 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71858 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71859 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71860 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71861 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_8
  simp only [← c71856, ← c71857, ← c71858] at e_1164_8
  have e_1164_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_9
  simp only [← c71859, ← c71860, k2, ← c71861] at e_1164_9
  have hr := e_1164_9
  simp only [e_1164_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10250)) := by
  have c71862 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71863 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71864 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71865 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71866 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71867 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_10
  simp only [← c71862, ← c71863, ← c71864] at e_1164_10
  have e_1164_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_11
  simp only [← c71865, ← c71866, k2, ← c71867] at e_1164_11
  have hr := e_1164_11
  simp only [e_1164_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10251)) := by
  have c71868 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71869 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71870 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71871 := (publicBatchWrapper89_copies2245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71872 := (publicBatchWrapper89_copies2246 a h).1
  have c71873 := (publicBatchWrapper89_copies2246 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_12
  simp only [← c71868, ← c71869, ← c71870] at e_1164_12
  have e_1164_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_13
  simp only [← c71871, ← c71872, k2, ← c71873] at e_1164_13
  have hr := e_1164_13
  simp only [e_1164_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10252)) := by
  have c71874 := (publicBatchWrapper89_copies2246 a h).2.2.1
  have c71875 := (publicBatchWrapper89_copies2246 a h).2.2.2.1
  have c71876 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.1
  have c71877 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.1
  have c71878 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.1
  have c71879 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_14
  simp only [← c71874, ← c71875, ← c71876] at e_1164_14
  have e_1164_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_15
  simp only [← c71877, ← c71878, k2, ← c71879] at e_1164_15
  have hr := e_1164_15
  simp only [e_1164_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10253)) := by
  have c71880 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.1
  have c71881 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.1
  have c71882 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71883 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71884 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71885 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_16
  simp only [← c71880, ← c71881, ← c71882] at e_1164_16
  have e_1164_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_17
  simp only [← c71883, ← c71884, k2, ← c71885] at e_1164_17
  have hr := e_1164_17
  simp only [e_1164_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1164 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10254)) := by
  have c71886 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71887 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71888 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71889 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71890 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71891 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1164_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_18
  simp only [← c71886, ← c71887, ← c71888] at e_1164_18
  have e_1164_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1164 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_19
  simp only [← c71889, ← c71890, k2, ← c71891] at e_1164_19
  have hr := e_1164_19
  simp only [e_1164_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10255)) := by
  have c71892 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71893 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71894 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71895 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71896 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71897 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_0
  simp only [← c71892, ← c71893, ← c71894] at e_1165_0
  have e_1165_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_1
  simp only [← c71895, ← c71896, k2, ← c71897] at e_1165_1
  have hr := e_1165_1
  simp only [e_1165_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10256)) := by
  have c71898 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71899 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71900 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71901 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71902 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71903 := (publicBatchWrapper89_copies2246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_2
  simp only [← c71898, ← c71899, ← c71900] at e_1165_2
  have e_1165_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_3
  simp only [← c71901, ← c71902, k2, ← c71903] at e_1165_3
  have hr := e_1165_3
  simp only [e_1165_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10257)) := by
  have c71904 := (publicBatchWrapper89_copies2247 a h).1
  have c71905 := (publicBatchWrapper89_copies2247 a h).2.1
  have c71906 := (publicBatchWrapper89_copies2247 a h).2.2.1
  have c71907 := (publicBatchWrapper89_copies2247 a h).2.2.2.1
  have c71908 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.1
  have c71909 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_4
  simp only [← c71904, ← c71905, ← c71906] at e_1165_4
  have e_1165_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_5
  simp only [← c71907, ← c71908, k2, ← c71909] at e_1165_5
  have hr := e_1165_5
  simp only [e_1165_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10258)) := by
  have c71910 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.1
  have c71911 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.1
  have c71912 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.1
  have c71913 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.1
  have c71914 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71915 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_6
  simp only [← c71910, ← c71911, ← c71912] at e_1165_6
  have e_1165_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_7
  simp only [← c71913, ← c71914, k2, ← c71915] at e_1165_7
  have hr := e_1165_7
  simp only [e_1165_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10259)) := by
  have c71916 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71917 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71918 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71919 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71920 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71921 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_8
  simp only [← c71916, ← c71917, ← c71918] at e_1165_8
  have e_1165_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_9
  simp only [← c71919, ← c71920, k2, ← c71921] at e_1165_9
  have hr := e_1165_9
  simp only [e_1165_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10260)) := by
  have c71922 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71923 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71924 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71925 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71926 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71927 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_10
  simp only [← c71922, ← c71923, ← c71924] at e_1165_10
  have e_1165_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_11
  simp only [← c71925, ← c71926, k2, ← c71927] at e_1165_11
  have hr := e_1165_11
  simp only [e_1165_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10261)) := by
  have c71928 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71929 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71930 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71931 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71932 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71933 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_12
  simp only [← c71928, ← c71929, ← c71930] at e_1165_12
  have e_1165_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_13
  simp only [← c71931, ← c71932, k2, ← c71933] at e_1165_13
  have hr := e_1165_13
  simp only [e_1165_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10262)) := by
  have c71934 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71935 := (publicBatchWrapper89_copies2247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71936 := (publicBatchWrapper89_copies2248 a h).1
  have c71937 := (publicBatchWrapper89_copies2248 a h).2.1
  have c71938 := (publicBatchWrapper89_copies2248 a h).2.2.1
  have c71939 := (publicBatchWrapper89_copies2248 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_14
  simp only [← c71934, ← c71935, ← c71936] at e_1165_14
  have e_1165_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_15
  simp only [← c71937, ← c71938, k2, ← c71939] at e_1165_15
  have hr := e_1165_15
  simp only [e_1165_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10263)) := by
  have c71940 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.1
  have c71941 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.1
  have c71942 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.1
  have c71943 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.1
  have c71944 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.1
  have c71945 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_16
  simp only [← c71940, ← c71941, ← c71942] at e_1165_16
  have e_1165_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_17
  simp only [← c71943, ← c71944, k2, ← c71945] at e_1165_17
  have hr := e_1165_17
  simp only [e_1165_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1165 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10264)) := by
  have c71946 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71947 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71948 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71949 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71950 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71951 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1165_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_18
  simp only [← c71946, ← c71947, ← c71948] at e_1165_18
  have e_1165_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1165 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_19
  simp only [← c71949, ← c71950, k2, ← c71951] at e_1165_19
  have hr := e_1165_19
  simp only [e_1165_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10265)) := by
  have c71952 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71953 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71954 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71955 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71956 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71957 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_0
  simp only [← c71952, ← c71953, ← c71954] at e_1166_0
  have e_1166_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_1
  simp only [← c71955, ← c71956, k2, ← c71957] at e_1166_1
  have hr := e_1166_1
  simp only [e_1166_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10266)) := by
  have c71958 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71959 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71960 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71961 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71962 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71963 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_2
  simp only [← c71958, ← c71959, ← c71960] at e_1166_2
  have e_1166_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_3
  simp only [← c71961, ← c71962, k2, ← c71963] at e_1166_3
  have hr := e_1166_3
  simp only [e_1166_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10267)) := by
  have c71964 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71965 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71966 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71967 := (publicBatchWrapper89_copies2248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c71968 := (publicBatchWrapper89_copies2249 a h).1
  have c71969 := (publicBatchWrapper89_copies2249 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_4
  simp only [← c71964, ← c71965, ← c71966] at e_1166_4
  have e_1166_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_5
  simp only [← c71967, ← c71968, k2, ← c71969] at e_1166_5
  have hr := e_1166_5
  simp only [e_1166_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10268)) := by
  have c71970 := (publicBatchWrapper89_copies2249 a h).2.2.1
  have c71971 := (publicBatchWrapper89_copies2249 a h).2.2.2.1
  have c71972 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.1
  have c71973 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.1
  have c71974 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.1
  have c71975 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_6
  simp only [← c71970, ← c71971, ← c71972] at e_1166_6
  have e_1166_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_7
  simp only [← c71973, ← c71974, k2, ← c71975] at e_1166_7
  have hr := e_1166_7
  simp only [e_1166_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10269)) := by
  have c71976 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.1
  have c71977 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.1
  have c71978 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c71979 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c71980 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71981 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_8
  simp only [← c71976, ← c71977, ← c71978] at e_1166_8
  have e_1166_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_9
  simp only [← c71979, ← c71980, k2, ← c71981] at e_1166_9
  have hr := e_1166_9
  simp only [e_1166_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10270)) := by
  have c71982 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71983 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71984 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71985 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71986 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71987 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_10
  simp only [← c71982, ← c71983, ← c71984] at e_1166_10
  have e_1166_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_11
  simp only [← c71985, ← c71986, k2, ← c71987] at e_1166_11
  have hr := e_1166_11
  simp only [e_1166_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10271)) := by
  have c71988 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71989 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71990 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71991 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71992 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71993 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_12
  simp only [← c71988, ← c71989, ← c71990] at e_1166_12
  have e_1166_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_13
  simp only [← c71991, ← c71992, k2, ← c71993] at e_1166_13
  have hr := e_1166_13
  simp only [e_1166_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10272)) := by
  have c71994 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71995 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71996 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71997 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71998 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c71999 := (publicBatchWrapper89_copies2249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_14
  simp only [← c71994, ← c71995, ← c71996] at e_1166_14
  have e_1166_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_15
  simp only [← c71997, ← c71998, k2, ← c71999] at e_1166_15
  have hr := e_1166_15
  simp only [e_1166_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10273)) := by
  have c72000 := (publicBatchWrapper89_copies2250 a h).1
  have c72001 := (publicBatchWrapper89_copies2250 a h).2.1
  have c72002 := (publicBatchWrapper89_copies2250 a h).2.2.1
  have c72003 := (publicBatchWrapper89_copies2250 a h).2.2.2.1
  have c72004 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.1
  have c72005 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_16
  simp only [← c72000, ← c72001, ← c72002] at e_1166_16
  have e_1166_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_17
  simp only [← c72003, ← c72004, k2, ← c72005] at e_1166_17
  have hr := e_1166_17
  simp only [e_1166_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1166 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10274)) := by
  have c72006 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.1
  have c72007 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.1
  have c72008 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.1
  have c72009 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.1
  have c72010 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72011 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1166_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_18
  simp only [← c72006, ← c72007, ← c72008] at e_1166_18
  have e_1166_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1166 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_19
  simp only [← c72009, ← c72010, k2, ← c72011] at e_1166_19
  have hr := e_1166_19
  simp only [e_1166_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10275)) := by
  have c72012 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72013 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72014 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72015 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72016 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72017 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_0
  simp only [← c72012, ← c72013, ← c72014] at e_1167_0
  have e_1167_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_1
  simp only [← c72015, ← c72016, k2, ← c72017] at e_1167_1
  have hr := e_1167_1
  simp only [e_1167_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10276)) := by
  have c72018 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72019 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72020 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72021 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72022 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72023 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_2
  simp only [← c72018, ← c72019, ← c72020] at e_1167_2
  have e_1167_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_3
  simp only [← c72021, ← c72022, k2, ← c72023] at e_1167_3
  have hr := e_1167_3
  simp only [e_1167_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10277)) := by
  have c72024 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72025 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72026 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72027 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72028 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72029 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_4
  simp only [← c72024, ← c72025, ← c72026] at e_1167_4
  have e_1167_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_5
  simp only [← c72027, ← c72028, k2, ← c72029] at e_1167_5
  have hr := e_1167_5
  simp only [e_1167_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10278)) := by
  have c72030 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72031 := (publicBatchWrapper89_copies2250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72032 := (publicBatchWrapper89_copies2251 a h).1
  have c72033 := (publicBatchWrapper89_copies2251 a h).2.1
  have c72034 := (publicBatchWrapper89_copies2251 a h).2.2.1
  have c72035 := (publicBatchWrapper89_copies2251 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_6
  simp only [← c72030, ← c72031, ← c72032] at e_1167_6
  have e_1167_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_7
  simp only [← c72033, ← c72034, k2, ← c72035] at e_1167_7
  have hr := e_1167_7
  simp only [e_1167_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10279)) := by
  have c72036 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.1
  have c72037 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.1
  have c72038 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.1
  have c72039 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.1
  have c72040 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.1
  have c72041 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_8
  simp only [← c72036, ← c72037, ← c72038] at e_1167_8
  have e_1167_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_9
  simp only [← c72039, ← c72040, k2, ← c72041] at e_1167_9
  have hr := e_1167_9
  simp only [e_1167_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10280)) := by
  have c72042 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72043 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72044 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72045 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72046 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72047 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_10
  simp only [← c72042, ← c72043, ← c72044] at e_1167_10
  have e_1167_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_11
  simp only [← c72045, ← c72046, k2, ← c72047] at e_1167_11
  have hr := e_1167_11
  simp only [e_1167_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10281)) := by
  have c72048 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72049 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72050 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72051 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72052 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72053 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_12
  simp only [← c72048, ← c72049, ← c72050] at e_1167_12
  have e_1167_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_13
  simp only [← c72051, ← c72052, k2, ← c72053] at e_1167_13
  have hr := e_1167_13
  simp only [e_1167_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10282)) := by
  have c72054 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72055 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72056 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72057 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72058 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72059 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_14
  simp only [← c72054, ← c72055, ← c72056] at e_1167_14
  have e_1167_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_15
  simp only [← c72057, ← c72058, k2, ← c72059] at e_1167_15
  have hr := e_1167_15
  simp only [e_1167_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10283)) := by
  have c72060 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72061 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72062 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72063 := (publicBatchWrapper89_copies2251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72064 := (publicBatchWrapper89_copies2252 a h).1
  have c72065 := (publicBatchWrapper89_copies2252 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_16
  simp only [← c72060, ← c72061, ← c72062] at e_1167_16
  have e_1167_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_17
  simp only [← c72063, ← c72064, k2, ← c72065] at e_1167_17
  have hr := e_1167_17
  simp only [e_1167_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1167 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10284)) := by
  have c72066 := (publicBatchWrapper89_copies2252 a h).2.2.1
  have c72067 := (publicBatchWrapper89_copies2252 a h).2.2.2.1
  have c72068 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.1
  have c72069 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.1
  have c72070 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.1
  have c72071 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1167_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_18
  simp only [← c72066, ← c72067, ← c72068] at e_1167_18
  have e_1167_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1167 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_19
  simp only [← c72069, ← c72070, k2, ← c72071] at e_1167_19
  have hr := e_1167_19
  simp only [e_1167_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10285)) := by
  have c72072 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.1
  have c72073 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.1
  have c72074 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72075 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72076 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72077 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_0
  simp only [← c72072, ← c72073, ← c72074] at e_1168_0
  have e_1168_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_1
  simp only [← c72075, ← c72076, k2, ← c72077] at e_1168_1
  have hr := e_1168_1
  simp only [e_1168_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10286)) := by
  have c72078 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72079 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72080 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72081 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72082 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72083 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_2
  simp only [← c72078, ← c72079, ← c72080] at e_1168_2
  have e_1168_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_3
  simp only [← c72081, ← c72082, k2, ← c72083] at e_1168_3
  have hr := e_1168_3
  simp only [e_1168_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10287)) := by
  have c72084 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72085 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72086 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72087 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72088 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72089 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_4
  simp only [← c72084, ← c72085, ← c72086] at e_1168_4
  have e_1168_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_5
  simp only [← c72087, ← c72088, k2, ← c72089] at e_1168_5
  have hr := e_1168_5
  simp only [e_1168_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10288)) := by
  have c72090 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72091 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72092 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72093 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72094 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72095 := (publicBatchWrapper89_copies2252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_6
  simp only [← c72090, ← c72091, ← c72092] at e_1168_6
  have e_1168_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_7
  simp only [← c72093, ← c72094, k2, ← c72095] at e_1168_7
  have hr := e_1168_7
  simp only [e_1168_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10289)) := by
  have c72096 := (publicBatchWrapper89_copies2253 a h).1
  have c72097 := (publicBatchWrapper89_copies2253 a h).2.1
  have c72098 := (publicBatchWrapper89_copies2253 a h).2.2.1
  have c72099 := (publicBatchWrapper89_copies2253 a h).2.2.2.1
  have c72100 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.1
  have c72101 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_8
  simp only [← c72096, ← c72097, ← c72098] at e_1168_8
  have e_1168_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_9
  simp only [← c72099, ← c72100, k2, ← c72101] at e_1168_9
  have hr := e_1168_9
  simp only [e_1168_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10290)) := by
  have c72102 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.1
  have c72103 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.1
  have c72104 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.1
  have c72105 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.1
  have c72106 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72107 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_10
  simp only [← c72102, ← c72103, ← c72104] at e_1168_10
  have e_1168_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_11
  simp only [← c72105, ← c72106, k2, ← c72107] at e_1168_11
  have hr := e_1168_11
  simp only [e_1168_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10291)) := by
  have c72108 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72109 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72110 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72111 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72112 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72113 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_12
  simp only [← c72108, ← c72109, ← c72110] at e_1168_12
  have e_1168_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_13
  simp only [← c72111, ← c72112, k2, ← c72113] at e_1168_13
  have hr := e_1168_13
  simp only [e_1168_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10292)) := by
  have c72114 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72115 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72116 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72117 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72118 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72119 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_14
  simp only [← c72114, ← c72115, ← c72116] at e_1168_14
  have e_1168_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_15
  simp only [← c72117, ← c72118, k2, ← c72119] at e_1168_15
  have hr := e_1168_15
  simp only [e_1168_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10293)) := by
  have c72120 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72121 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72122 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72123 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72124 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72125 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_16
  simp only [← c72120, ← c72121, ← c72122] at e_1168_16
  have e_1168_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_17
  simp only [← c72123, ← c72124, k2, ← c72125] at e_1168_17
  have hr := e_1168_17
  simp only [e_1168_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1168 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10294)) := by
  have c72126 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72127 := (publicBatchWrapper89_copies2253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72128 := (publicBatchWrapper89_copies2254 a h).1
  have c72129 := (publicBatchWrapper89_copies2254 a h).2.1
  have c72130 := (publicBatchWrapper89_copies2254 a h).2.2.1
  have c72131 := (publicBatchWrapper89_copies2254 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1168_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_18
  simp only [← c72126, ← c72127, ← c72128] at e_1168_18
  have e_1168_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1168 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_19
  simp only [← c72129, ← c72130, k2, ← c72131] at e_1168_19
  have hr := e_1168_19
  simp only [e_1168_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10295)) := by
  have c72132 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.1
  have c72133 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.1
  have c72134 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.1
  have c72135 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.1
  have c72136 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.1
  have c72137 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_0
  simp only [← c72132, ← c72133, ← c72134] at e_1169_0
  have e_1169_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_1
  simp only [← c72135, ← c72136, k2, ← c72137] at e_1169_1
  have hr := e_1169_1
  simp only [e_1169_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10296)) := by
  have c72138 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72139 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72140 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72141 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72142 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72143 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_2
  simp only [← c72138, ← c72139, ← c72140] at e_1169_2
  have e_1169_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_3
  simp only [← c72141, ← c72142, k2, ← c72143] at e_1169_3
  have hr := e_1169_3
  simp only [e_1169_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10297)) := by
  have c72144 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72145 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72146 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72147 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72148 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72149 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_4
  simp only [← c72144, ← c72145, ← c72146] at e_1169_4
  have e_1169_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_5
  simp only [← c72147, ← c72148, k2, ← c72149] at e_1169_5
  have hr := e_1169_5
  simp only [e_1169_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10298)) := by
  have c72150 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72151 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72152 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72153 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72154 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72155 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_6
  simp only [← c72150, ← c72151, ← c72152] at e_1169_6
  have e_1169_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_7
  simp only [← c72153, ← c72154, k2, ← c72155] at e_1169_7
  have hr := e_1169_7
  simp only [e_1169_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10299)) := by
  have c72156 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72157 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72158 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72159 := (publicBatchWrapper89_copies2254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72160 := (publicBatchWrapper89_copies2255 a h).1
  have c72161 := (publicBatchWrapper89_copies2255 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_8
  simp only [← c72156, ← c72157, ← c72158] at e_1169_8
  have e_1169_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_9
  simp only [← c72159, ← c72160, k2, ← c72161] at e_1169_9
  have hr := e_1169_9
  simp only [e_1169_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10300)) := by
  have c72162 := (publicBatchWrapper89_copies2255 a h).2.2.1
  have c72163 := (publicBatchWrapper89_copies2255 a h).2.2.2.1
  have c72164 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.1
  have c72165 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.1
  have c72166 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.1
  have c72167 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_10
  simp only [← c72162, ← c72163, ← c72164] at e_1169_10
  have e_1169_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_11
  simp only [← c72165, ← c72166, k2, ← c72167] at e_1169_11
  have hr := e_1169_11
  simp only [e_1169_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10301)) := by
  have c72168 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.1
  have c72169 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.1
  have c72170 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72171 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72172 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72173 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_12
  simp only [← c72168, ← c72169, ← c72170] at e_1169_12
  have e_1169_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_13
  simp only [← c72171, ← c72172, k2, ← c72173] at e_1169_13
  have hr := e_1169_13
  simp only [e_1169_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10302)) := by
  have c72174 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72175 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72176 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72177 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72178 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72179 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_14
  simp only [← c72174, ← c72175, ← c72176] at e_1169_14
  have e_1169_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_15
  simp only [← c72177, ← c72178, k2, ← c72179] at e_1169_15
  have hr := e_1169_15
  simp only [e_1169_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10303)) := by
  have c72180 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72181 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72182 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72183 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72184 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72185 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_16
  simp only [← c72180, ← c72181, ← c72182] at e_1169_16
  have e_1169_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_17
  simp only [← c72183, ← c72184, k2, ← c72185] at e_1169_17
  have hr := e_1169_17
  simp only [e_1169_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1169 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10304)) := by
  have c72186 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72187 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72188 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72189 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72190 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72191 := (publicBatchWrapper89_copies2255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1169_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_18
  simp only [← c72186, ← c72187, ← c72188] at e_1169_18
  have e_1169_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1169 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_19
  simp only [← c72189, ← c72190, k2, ← c72191] at e_1169_19
  have hr := e_1169_19
  simp only [e_1169_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10305)) := by
  have c72192 := (publicBatchWrapper89_copies2256 a h).1
  have c72193 := (publicBatchWrapper89_copies2256 a h).2.1
  have c72194 := (publicBatchWrapper89_copies2256 a h).2.2.1
  have c72195 := (publicBatchWrapper89_copies2256 a h).2.2.2.1
  have c72196 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.1
  have c72197 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_0
  simp only [← c72192, ← c72193, ← c72194] at e_1170_0
  have e_1170_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_1
  simp only [← c72195, ← c72196, k2, ← c72197] at e_1170_1
  have hr := e_1170_1
  simp only [e_1170_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10306)) := by
  have c72198 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.1
  have c72199 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.1
  have c72200 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.1
  have c72201 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.1
  have c72202 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72203 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_2
  simp only [← c72198, ← c72199, ← c72200] at e_1170_2
  have e_1170_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_3
  simp only [← c72201, ← c72202, k2, ← c72203] at e_1170_3
  have hr := e_1170_3
  simp only [e_1170_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10307)) := by
  have c72204 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72205 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72206 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72207 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72208 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72209 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_4
  simp only [← c72204, ← c72205, ← c72206] at e_1170_4
  have e_1170_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_5
  simp only [← c72207, ← c72208, k2, ← c72209] at e_1170_5
  have hr := e_1170_5
  simp only [e_1170_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10308)) := by
  have c72210 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72211 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72212 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72213 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72214 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72215 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_6
  simp only [← c72210, ← c72211, ← c72212] at e_1170_6
  have e_1170_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_7
  simp only [← c72213, ← c72214, k2, ← c72215] at e_1170_7
  have hr := e_1170_7
  simp only [e_1170_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10309)) := by
  have c72216 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72217 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72218 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72219 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72220 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72221 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_8
  simp only [← c72216, ← c72217, ← c72218] at e_1170_8
  have e_1170_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_9
  simp only [← c72219, ← c72220, k2, ← c72221] at e_1170_9
  have hr := e_1170_9
  simp only [e_1170_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10310)) := by
  have c72222 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72223 := (publicBatchWrapper89_copies2256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72224 := (publicBatchWrapper89_copies2257 a h).1
  have c72225 := (publicBatchWrapper89_copies2257 a h).2.1
  have c72226 := (publicBatchWrapper89_copies2257 a h).2.2.1
  have c72227 := (publicBatchWrapper89_copies2257 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_10
  simp only [← c72222, ← c72223, ← c72224] at e_1170_10
  have e_1170_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_11
  simp only [← c72225, ← c72226, k2, ← c72227] at e_1170_11
  have hr := e_1170_11
  simp only [e_1170_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10311)) := by
  have c72228 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.1
  have c72229 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.1
  have c72230 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.1
  have c72231 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.1
  have c72232 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.1
  have c72233 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_12
  simp only [← c72228, ← c72229, ← c72230] at e_1170_12
  have e_1170_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_13
  simp only [← c72231, ← c72232, k2, ← c72233] at e_1170_13
  have hr := e_1170_13
  simp only [e_1170_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10312)) := by
  have c72234 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72235 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72236 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72237 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72238 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72239 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_14
  simp only [← c72234, ← c72235, ← c72236] at e_1170_14
  have e_1170_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_15
  simp only [← c72237, ← c72238, k2, ← c72239] at e_1170_15
  have hr := e_1170_15
  simp only [e_1170_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10313)) := by
  have c72240 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72241 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72242 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72243 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72244 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72245 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_16
  simp only [← c72240, ← c72241, ← c72242] at e_1170_16
  have e_1170_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_17
  simp only [← c72243, ← c72244, k2, ← c72245] at e_1170_17
  have hr := e_1170_17
  simp only [e_1170_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1170 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10314)) := by
  have c72246 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72247 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72248 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72249 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72250 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72251 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1170_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_18
  simp only [← c72246, ← c72247, ← c72248] at e_1170_18
  have e_1170_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1170 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_19
  simp only [← c72249, ← c72250, k2, ← c72251] at e_1170_19
  have hr := e_1170_19
  simp only [e_1170_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10315)) := by
  have c72252 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72253 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72254 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72255 := (publicBatchWrapper89_copies2257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72256 := (publicBatchWrapper89_copies2258 a h).1
  have c72257 := (publicBatchWrapper89_copies2258 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_0
  simp only [← c72252, ← c72253, ← c72254] at e_1171_0
  have e_1171_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_1
  simp only [← c72255, ← c72256, k2, ← c72257] at e_1171_1
  have hr := e_1171_1
  simp only [e_1171_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10316)) := by
  have c72258 := (publicBatchWrapper89_copies2258 a h).2.2.1
  have c72259 := (publicBatchWrapper89_copies2258 a h).2.2.2.1
  have c72260 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.1
  have c72261 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.1
  have c72262 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.1
  have c72263 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_2
  simp only [← c72258, ← c72259, ← c72260] at e_1171_2
  have e_1171_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_3
  simp only [← c72261, ← c72262, k2, ← c72263] at e_1171_3
  have hr := e_1171_3
  simp only [e_1171_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10317)) := by
  have c72264 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.1
  have c72265 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.1
  have c72266 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72267 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72268 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72269 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_4
  simp only [← c72264, ← c72265, ← c72266] at e_1171_4
  have e_1171_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_5
  simp only [← c72267, ← c72268, k2, ← c72269] at e_1171_5
  have hr := e_1171_5
  simp only [e_1171_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10318)) := by
  have c72270 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72271 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72272 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72273 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72274 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72275 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_6
  simp only [← c72270, ← c72271, ← c72272] at e_1171_6
  have e_1171_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_7
  simp only [← c72273, ← c72274, k2, ← c72275] at e_1171_7
  have hr := e_1171_7
  simp only [e_1171_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10319)) := by
  have c72276 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72277 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72278 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72279 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72280 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72281 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_8
  simp only [← c72276, ← c72277, ← c72278] at e_1171_8
  have e_1171_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_9
  simp only [← c72279, ← c72280, k2, ← c72281] at e_1171_9
  have hr := e_1171_9
  simp only [e_1171_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10320)) := by
  have c72282 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72283 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72284 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72285 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72286 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72287 := (publicBatchWrapper89_copies2258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_10
  simp only [← c72282, ← c72283, ← c72284] at e_1171_10
  have e_1171_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_11
  simp only [← c72285, ← c72286, k2, ← c72287] at e_1171_11
  have hr := e_1171_11
  simp only [e_1171_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10321)) := by
  have c72288 := (publicBatchWrapper89_copies2259 a h).1
  have c72289 := (publicBatchWrapper89_copies2259 a h).2.1
  have c72290 := (publicBatchWrapper89_copies2259 a h).2.2.1
  have c72291 := (publicBatchWrapper89_copies2259 a h).2.2.2.1
  have c72292 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.1
  have c72293 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_12
  simp only [← c72288, ← c72289, ← c72290] at e_1171_12
  have e_1171_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_13
  simp only [← c72291, ← c72292, k2, ← c72293] at e_1171_13
  have hr := e_1171_13
  simp only [e_1171_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10322)) := by
  have c72294 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.1
  have c72295 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.1
  have c72296 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.1
  have c72297 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.1
  have c72298 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72299 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_14
  simp only [← c72294, ← c72295, ← c72296] at e_1171_14
  have e_1171_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_15
  simp only [← c72297, ← c72298, k2, ← c72299] at e_1171_15
  have hr := e_1171_15
  simp only [e_1171_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10323)) := by
  have c72300 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72301 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72302 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72303 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72304 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72305 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_16
  simp only [← c72300, ← c72301, ← c72302] at e_1171_16
  have e_1171_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_17
  simp only [← c72303, ← c72304, k2, ← c72305] at e_1171_17
  have hr := e_1171_17
  simp only [e_1171_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1171 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10324)) := by
  have c72306 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72307 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72308 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72309 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72310 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72311 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1171_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_18
  simp only [← c72306, ← c72307, ← c72308] at e_1171_18
  have e_1171_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1171 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_19
  simp only [← c72309, ← c72310, k2, ← c72311] at e_1171_19
  have hr := e_1171_19
  simp only [e_1171_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10325)) := by
  have c72312 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72313 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72314 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72315 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72316 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72317 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_0
  simp only [← c72312, ← c72313, ← c72314] at e_1172_0
  have e_1172_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_1
  simp only [← c72315, ← c72316, k2, ← c72317] at e_1172_1
  have hr := e_1172_1
  simp only [e_1172_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10326)) := by
  have c72318 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72319 := (publicBatchWrapper89_copies2259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72320 := (publicBatchWrapper89_copies2260 a h).1
  have c72321 := (publicBatchWrapper89_copies2260 a h).2.1
  have c72322 := (publicBatchWrapper89_copies2260 a h).2.2.1
  have c72323 := (publicBatchWrapper89_copies2260 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_2
  simp only [← c72318, ← c72319, ← c72320] at e_1172_2
  have e_1172_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_3
  simp only [← c72321, ← c72322, k2, ← c72323] at e_1172_3
  have hr := e_1172_3
  simp only [e_1172_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10327)) := by
  have c72324 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.1
  have c72325 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.1
  have c72326 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.1
  have c72327 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.1
  have c72328 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.1
  have c72329 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_4
  simp only [← c72324, ← c72325, ← c72326] at e_1172_4
  have e_1172_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_5
  simp only [← c72327, ← c72328, k2, ← c72329] at e_1172_5
  have hr := e_1172_5
  simp only [e_1172_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10328)) := by
  have c72330 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72331 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72332 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72333 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72334 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72335 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_6
  simp only [← c72330, ← c72331, ← c72332] at e_1172_6
  have e_1172_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_7
  simp only [← c72333, ← c72334, k2, ← c72335] at e_1172_7
  have hr := e_1172_7
  simp only [e_1172_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10329)) := by
  have c72336 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72337 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72338 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72339 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72340 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72341 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_8
  simp only [← c72336, ← c72337, ← c72338] at e_1172_8
  have e_1172_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_9
  simp only [← c72339, ← c72340, k2, ← c72341] at e_1172_9
  have hr := e_1172_9
  simp only [e_1172_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10330)) := by
  have c72342 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72343 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72344 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72345 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72346 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72347 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_10
  simp only [← c72342, ← c72343, ← c72344] at e_1172_10
  have e_1172_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_11
  simp only [← c72345, ← c72346, k2, ← c72347] at e_1172_11
  have hr := e_1172_11
  simp only [e_1172_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10331)) := by
  have c72348 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72349 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72350 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72351 := (publicBatchWrapper89_copies2260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72352 := (publicBatchWrapper89_copies2261 a h).1
  have c72353 := (publicBatchWrapper89_copies2261 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_12
  simp only [← c72348, ← c72349, ← c72350] at e_1172_12
  have e_1172_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_13
  simp only [← c72351, ← c72352, k2, ← c72353] at e_1172_13
  have hr := e_1172_13
  simp only [e_1172_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10332)) := by
  have c72354 := (publicBatchWrapper89_copies2261 a h).2.2.1
  have c72355 := (publicBatchWrapper89_copies2261 a h).2.2.2.1
  have c72356 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.1
  have c72357 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.1
  have c72358 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.1
  have c72359 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_14
  simp only [← c72354, ← c72355, ← c72356] at e_1172_14
  have e_1172_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_15
  simp only [← c72357, ← c72358, k2, ← c72359] at e_1172_15
  have hr := e_1172_15
  simp only [e_1172_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10333)) := by
  have c72360 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.1
  have c72361 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.1
  have c72362 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72363 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72364 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72365 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_16
  simp only [← c72360, ← c72361, ← c72362] at e_1172_16
  have e_1172_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_17
  simp only [← c72363, ← c72364, k2, ← c72365] at e_1172_17
  have hr := e_1172_17
  simp only [e_1172_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1172 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10334)) := by
  have c72366 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72367 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72368 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72369 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72370 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72371 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1172_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_18
  simp only [← c72366, ← c72367, ← c72368] at e_1172_18
  have e_1172_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1172 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_19
  simp only [← c72369, ← c72370, k2, ← c72371] at e_1172_19
  have hr := e_1172_19
  simp only [e_1172_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10335)) := by
  have c72372 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72373 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72374 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72375 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72376 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72377 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_0
  simp only [← c72372, ← c72373, ← c72374] at e_1173_0
  have e_1173_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_1
  simp only [← c72375, ← c72376, k2, ← c72377] at e_1173_1
  have hr := e_1173_1
  simp only [e_1173_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10336)) := by
  have c72378 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72379 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72380 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72381 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72382 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72383 := (publicBatchWrapper89_copies2261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_2
  simp only [← c72378, ← c72379, ← c72380] at e_1173_2
  have e_1173_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_3
  simp only [← c72381, ← c72382, k2, ← c72383] at e_1173_3
  have hr := e_1173_3
  simp only [e_1173_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10337)) := by
  have c72384 := (publicBatchWrapper89_copies2262 a h).1
  have c72385 := (publicBatchWrapper89_copies2262 a h).2.1
  have c72386 := (publicBatchWrapper89_copies2262 a h).2.2.1
  have c72387 := (publicBatchWrapper89_copies2262 a h).2.2.2.1
  have c72388 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.1
  have c72389 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_4
  simp only [← c72384, ← c72385, ← c72386] at e_1173_4
  have e_1173_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_5
  simp only [← c72387, ← c72388, k2, ← c72389] at e_1173_5
  have hr := e_1173_5
  simp only [e_1173_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10338)) := by
  have c72390 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.1
  have c72391 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.1
  have c72392 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.1
  have c72393 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.1
  have c72394 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72395 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_6
  simp only [← c72390, ← c72391, ← c72392] at e_1173_6
  have e_1173_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_7
  simp only [← c72393, ← c72394, k2, ← c72395] at e_1173_7
  have hr := e_1173_7
  simp only [e_1173_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10339)) := by
  have c72396 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72397 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72398 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72399 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72400 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72401 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_8
  simp only [← c72396, ← c72397, ← c72398] at e_1173_8
  have e_1173_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_9
  simp only [← c72399, ← c72400, k2, ← c72401] at e_1173_9
  have hr := e_1173_9
  simp only [e_1173_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10340)) := by
  have c72402 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72403 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72404 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72405 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72406 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72407 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_10
  simp only [← c72402, ← c72403, ← c72404] at e_1173_10
  have e_1173_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_11
  simp only [← c72405, ← c72406, k2, ← c72407] at e_1173_11
  have hr := e_1173_11
  simp only [e_1173_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10341)) := by
  have c72408 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72409 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72410 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72411 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72412 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72413 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_12
  simp only [← c72408, ← c72409, ← c72410] at e_1173_12
  have e_1173_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_13
  simp only [← c72411, ← c72412, k2, ← c72413] at e_1173_13
  have hr := e_1173_13
  simp only [e_1173_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10342)) := by
  have c72414 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72415 := (publicBatchWrapper89_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72416 := (publicBatchWrapper89_copies2263 a h).1
  have c72417 := (publicBatchWrapper89_copies2263 a h).2.1
  have c72418 := (publicBatchWrapper89_copies2263 a h).2.2.1
  have c72419 := (publicBatchWrapper89_copies2263 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_14
  simp only [← c72414, ← c72415, ← c72416] at e_1173_14
  have e_1173_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_15
  simp only [← c72417, ← c72418, k2, ← c72419] at e_1173_15
  have hr := e_1173_15
  simp only [e_1173_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10343)) := by
  have c72420 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.1
  have c72421 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.1
  have c72422 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.1
  have c72423 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.1
  have c72424 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.1
  have c72425 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_16
  simp only [← c72420, ← c72421, ← c72422] at e_1173_16
  have e_1173_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_17
  simp only [← c72423, ← c72424, k2, ← c72425] at e_1173_17
  have hr := e_1173_17
  simp only [e_1173_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1173 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10344)) := by
  have c72426 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72427 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72428 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72429 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72430 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72431 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1173_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_18
  simp only [← c72426, ← c72427, ← c72428] at e_1173_18
  have e_1173_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1173 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_19
  simp only [← c72429, ← c72430, k2, ← c72431] at e_1173_19
  have hr := e_1173_19
  simp only [e_1173_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10345)) := by
  have c72432 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72433 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72434 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72435 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72436 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72437 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_0
  simp only [← c72432, ← c72433, ← c72434] at e_1174_0
  have e_1174_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_1
  simp only [← c72435, ← c72436, k2, ← c72437] at e_1174_1
  have hr := e_1174_1
  simp only [e_1174_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10346)) := by
  have c72438 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72439 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72440 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72441 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72442 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72443 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_2
  simp only [← c72438, ← c72439, ← c72440] at e_1174_2
  have e_1174_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_3
  simp only [← c72441, ← c72442, k2, ← c72443] at e_1174_3
  have hr := e_1174_3
  simp only [e_1174_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10347)) := by
  have c72444 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72445 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72446 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72447 := (publicBatchWrapper89_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72448 := (publicBatchWrapper89_copies2264 a h).1
  have c72449 := (publicBatchWrapper89_copies2264 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_4
  simp only [← c72444, ← c72445, ← c72446] at e_1174_4
  have e_1174_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_5
  simp only [← c72447, ← c72448, k2, ← c72449] at e_1174_5
  have hr := e_1174_5
  simp only [e_1174_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10348)) := by
  have c72450 := (publicBatchWrapper89_copies2264 a h).2.2.1
  have c72451 := (publicBatchWrapper89_copies2264 a h).2.2.2.1
  have c72452 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.1
  have c72453 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.1
  have c72454 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.1
  have c72455 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_6
  simp only [← c72450, ← c72451, ← c72452] at e_1174_6
  have e_1174_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_7
  simp only [← c72453, ← c72454, k2, ← c72455] at e_1174_7
  have hr := e_1174_7
  simp only [e_1174_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10349)) := by
  have c72456 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.1
  have c72457 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.1
  have c72458 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72459 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72460 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72461 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_8
  simp only [← c72456, ← c72457, ← c72458] at e_1174_8
  have e_1174_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_9
  simp only [← c72459, ← c72460, k2, ← c72461] at e_1174_9
  have hr := e_1174_9
  simp only [e_1174_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10350)) := by
  have c72462 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72463 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72464 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72465 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72466 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72467 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_10
  simp only [← c72462, ← c72463, ← c72464] at e_1174_10
  have e_1174_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_11
  simp only [← c72465, ← c72466, k2, ← c72467] at e_1174_11
  have hr := e_1174_11
  simp only [e_1174_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10351)) := by
  have c72468 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72469 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72470 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72471 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72472 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72473 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_12
  simp only [← c72468, ← c72469, ← c72470] at e_1174_12
  have e_1174_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_13
  simp only [← c72471, ← c72472, k2, ← c72473] at e_1174_13
  have hr := e_1174_13
  simp only [e_1174_12] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
