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

theorem publicBatchWrapper86_f5917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3723)) := by
  have c39855 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39856 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39857 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39858 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39859 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39860 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_10
  simp only [← c39855, ← c39856, ← c39857] at e_632_10
  have e_632_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_11
  simp only [← c39858, ← c39859, k2, ← c39860] at e_632_11
  have hr := e_632_11
  simp only [e_632_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3724)) := by
  have c39861 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39862 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39863 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39864 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39865 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39866 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_12
  simp only [← c39861, ← c39862, ← c39863] at e_632_12
  have e_632_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_13
  simp only [← c39864, ← c39865, k2, ← c39866] at e_632_13
  have hr := e_632_13
  simp only [e_632_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3725)) := by
  have c39867 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39868 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39869 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39870 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39871 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39872 := (publicBatchWrapper86_copies1246 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_14
  simp only [← c39867, ← c39868, ← c39869] at e_632_14
  have e_632_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_15
  simp only [← c39870, ← c39871, k2, ← c39872] at e_632_15
  have hr := e_632_15
  simp only [e_632_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3726)) := by
  have c39873 := (publicBatchWrapper86_copies1246 a h).2.1
  have c39874 := (publicBatchWrapper86_copies1246 a h).2.2.1
  have c39875 := (publicBatchWrapper86_copies1246 a h).2.2.2.1
  have c39876 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.1
  have c39877 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.1
  have c39878 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_16
  simp only [← c39873, ← c39874, ← c39875] at e_632_16
  have e_632_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_17
  simp only [← c39876, ← c39877, k2, ← c39878] at e_632_17
  have hr := e_632_17
  simp only [e_632_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3727)) := by
  have c39879 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.1
  have c39880 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.1
  have c39881 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.1
  have c39882 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39883 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39884 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_18
  simp only [← c39879, ← c39880, ← c39881] at e_632_18
  have e_632_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_19
  simp only [← c39882, ← c39883, k2, ← c39884] at e_632_19
  have hr := e_632_19
  simp only [e_632_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3728)) := by
  have c39885 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39886 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39887 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39888 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39889 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39890 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_0
  simp only [← c39885, ← c39886, ← c39887] at e_633_0
  have e_633_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_1
  simp only [← c39888, ← c39889, k2, ← c39890] at e_633_1
  have hr := e_633_1
  simp only [e_633_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3729)) := by
  have c39891 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39892 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39893 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39894 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39895 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39896 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_2
  simp only [← c39891, ← c39892, ← c39893] at e_633_2
  have e_633_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_3
  simp only [← c39894, ← c39895, k2, ← c39896] at e_633_3
  have hr := e_633_3
  simp only [e_633_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3730)) := by
  have c39897 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39898 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39899 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39900 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39901 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39902 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_4
  simp only [← c39897, ← c39898, ← c39899] at e_633_4
  have e_633_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_5
  simp only [← c39900, ← c39901, k2, ← c39902] at e_633_5
  have hr := e_633_5
  simp only [e_633_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3731)) := by
  have c39903 := (publicBatchWrapper86_copies1246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39904 := (publicBatchWrapper86_copies1247 a h).1
  have c39905 := (publicBatchWrapper86_copies1247 a h).2.1
  have c39906 := (publicBatchWrapper86_copies1247 a h).2.2.1
  have c39907 := (publicBatchWrapper86_copies1247 a h).2.2.2.1
  have c39908 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_6
  simp only [← c39903, ← c39904, ← c39905] at e_633_6
  have e_633_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_7
  simp only [← c39906, ← c39907, k2, ← c39908] at e_633_7
  have hr := e_633_7
  simp only [e_633_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3732)) := by
  have c39909 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.1
  have c39910 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.1
  have c39911 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.1
  have c39912 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.1
  have c39913 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.1
  have c39914 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_8
  simp only [← c39909, ← c39910, ← c39911] at e_633_8
  have e_633_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_9
  simp only [← c39912, ← c39913, k2, ← c39914] at e_633_9
  have hr := e_633_9
  simp only [e_633_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3733)) := by
  have c39915 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39916 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39917 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39918 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39919 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39920 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_10
  simp only [← c39915, ← c39916, ← c39917] at e_633_10
  have e_633_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_11
  simp only [← c39918, ← c39919, k2, ← c39920] at e_633_11
  have hr := e_633_11
  simp only [e_633_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3734)) := by
  have c39921 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39922 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39923 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39924 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39925 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39926 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_12
  simp only [← c39921, ← c39922, ← c39923] at e_633_12
  have e_633_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_13
  simp only [← c39924, ← c39925, k2, ← c39926] at e_633_13
  have hr := e_633_13
  simp only [e_633_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3735)) := by
  have c39927 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39928 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39929 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39930 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39931 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39932 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_14
  simp only [← c39927, ← c39928, ← c39929] at e_633_14
  have e_633_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_15
  simp only [← c39930, ← c39931, k2, ← c39932] at e_633_15
  have hr := e_633_15
  simp only [e_633_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3736)) := by
  have c39933 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39934 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39935 := (publicBatchWrapper86_copies1247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39936 := (publicBatchWrapper86_copies1248 a h).1
  have c39937 := (publicBatchWrapper86_copies1248 a h).2.1
  have c39938 := (publicBatchWrapper86_copies1248 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_16
  simp only [← c39933, ← c39934, ← c39935] at e_633_16
  have e_633_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_17
  simp only [← c39936, ← c39937, k2, ← c39938] at e_633_17
  have hr := e_633_17
  simp only [e_633_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 633 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3737)) := by
  have c39939 := (publicBatchWrapper86_copies1248 a h).2.2.2.1
  have c39940 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.1
  have c39941 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.1
  have c39942 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.1
  have c39943 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.1
  have c39944 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_633_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_18
  simp only [← c39939, ← c39940, ← c39941] at e_633_18
  have e_633_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_633_19
  simp only [← c39942, ← c39943, k2, ← c39944] at e_633_19
  have hr := e_633_19
  simp only [e_633_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3738)) := by
  have c39945 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.1
  have c39946 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39947 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39948 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39949 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39950 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_0
  simp only [← c39945, ← c39946, ← c39947] at e_634_0
  have e_634_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_1
  simp only [← c39948, ← c39949, k2, ← c39950] at e_634_1
  have hr := e_634_1
  simp only [e_634_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3739)) := by
  have c39951 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39952 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39953 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39954 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39955 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39956 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_2
  simp only [← c39951, ← c39952, ← c39953] at e_634_2
  have e_634_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_3
  simp only [← c39954, ← c39955, k2, ← c39956] at e_634_3
  have hr := e_634_3
  simp only [e_634_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3740)) := by
  have c39957 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39958 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39959 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39960 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39961 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39962 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_4
  simp only [← c39957, ← c39958, ← c39959] at e_634_4
  have e_634_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_5
  simp only [← c39960, ← c39961, k2, ← c39962] at e_634_5
  have hr := e_634_5
  simp only [e_634_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3741)) := by
  have c39963 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39964 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39965 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39966 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39967 := (publicBatchWrapper86_copies1248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39968 := (publicBatchWrapper86_copies1249 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_6
  simp only [← c39963, ← c39964, ← c39965] at e_634_6
  have e_634_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_7
  simp only [← c39966, ← c39967, k2, ← c39968] at e_634_7
  have hr := e_634_7
  simp only [e_634_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3742)) := by
  have c39969 := (publicBatchWrapper86_copies1249 a h).2.1
  have c39970 := (publicBatchWrapper86_copies1249 a h).2.2.1
  have c39971 := (publicBatchWrapper86_copies1249 a h).2.2.2.1
  have c39972 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.1
  have c39973 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.1
  have c39974 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_8
  simp only [← c39969, ← c39970, ← c39971] at e_634_8
  have e_634_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_9
  simp only [← c39972, ← c39973, k2, ← c39974] at e_634_9
  have hr := e_634_9
  simp only [e_634_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3743)) := by
  have c39975 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.1
  have c39976 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.1
  have c39977 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.1
  have c39978 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39979 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39980 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_10
  simp only [← c39975, ← c39976, ← c39977] at e_634_10
  have e_634_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_11
  simp only [← c39978, ← c39979, k2, ← c39980] at e_634_11
  have hr := e_634_11
  simp only [e_634_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3744)) := by
  have c39981 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39982 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39983 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39984 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39985 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39986 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_12
  simp only [← c39981, ← c39982, ← c39983] at e_634_12
  have e_634_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_13
  simp only [← c39984, ← c39985, k2, ← c39986] at e_634_13
  have hr := e_634_13
  simp only [e_634_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3745)) := by
  have c39987 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39988 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39989 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39990 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39991 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39992 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_14
  simp only [← c39987, ← c39988, ← c39989] at e_634_14
  have e_634_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_15
  simp only [← c39990, ← c39991, k2, ← c39992] at e_634_15
  have hr := e_634_15
  simp only [e_634_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3746)) := by
  have c39993 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39994 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39995 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39996 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39997 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39998 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_16
  simp only [← c39993, ← c39994, ← c39995] at e_634_16
  have e_634_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_17
  simp only [← c39996, ← c39997, k2, ← c39998] at e_634_17
  have hr := e_634_17
  simp only [e_634_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 634 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3747)) := by
  have c39999 := (publicBatchWrapper86_copies1249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40000 := (publicBatchWrapper86_copies1250 a h).1
  have c40001 := (publicBatchWrapper86_copies1250 a h).2.1
  have c40002 := (publicBatchWrapper86_copies1250 a h).2.2.1
  have c40003 := (publicBatchWrapper86_copies1250 a h).2.2.2.1
  have c40004 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_634_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_18
  simp only [← c39999, ← c40000, ← c40001] at e_634_18
  have e_634_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_634_19
  simp only [← c40002, ← c40003, k2, ← c40004] at e_634_19
  have hr := e_634_19
  simp only [e_634_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3748)) := by
  have c40005 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.1
  have c40006 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.1
  have c40007 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.1
  have c40008 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.1
  have c40009 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.1
  have c40010 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_0
  simp only [← c40005, ← c40006, ← c40007] at e_635_0
  have e_635_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_1
  simp only [← c40008, ← c40009, k2, ← c40010] at e_635_1
  have hr := e_635_1
  simp only [e_635_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3749)) := by
  have c40011 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40012 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40013 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40014 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40015 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40016 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_2
  simp only [← c40011, ← c40012, ← c40013] at e_635_2
  have e_635_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_3
  simp only [← c40014, ← c40015, k2, ← c40016] at e_635_3
  have hr := e_635_3
  simp only [e_635_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3750)) := by
  have c40017 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40018 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40019 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40020 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40021 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40022 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_4
  simp only [← c40017, ← c40018, ← c40019] at e_635_4
  have e_635_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_5
  simp only [← c40020, ← c40021, k2, ← c40022] at e_635_5
  have hr := e_635_5
  simp only [e_635_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3751)) := by
  have c40023 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40024 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40025 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40026 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40027 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40028 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_6
  simp only [← c40023, ← c40024, ← c40025] at e_635_6
  have e_635_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_7
  simp only [← c40026, ← c40027, k2, ← c40028] at e_635_7
  have hr := e_635_7
  simp only [e_635_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3752)) := by
  have c40029 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40030 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40031 := (publicBatchWrapper86_copies1250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40032 := (publicBatchWrapper86_copies1251 a h).1
  have c40033 := (publicBatchWrapper86_copies1251 a h).2.1
  have c40034 := (publicBatchWrapper86_copies1251 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_8
  simp only [← c40029, ← c40030, ← c40031] at e_635_8
  have e_635_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_9
  simp only [← c40032, ← c40033, k2, ← c40034] at e_635_9
  have hr := e_635_9
  simp only [e_635_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3753)) := by
  have c40035 := (publicBatchWrapper86_copies1251 a h).2.2.2.1
  have c40036 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.1
  have c40037 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.1
  have c40038 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.1
  have c40039 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.1
  have c40040 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_10
  simp only [← c40035, ← c40036, ← c40037] at e_635_10
  have e_635_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_11
  simp only [← c40038, ← c40039, k2, ← c40040] at e_635_11
  have hr := e_635_11
  simp only [e_635_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3754)) := by
  have c40041 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.1
  have c40042 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40043 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40044 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40045 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40046 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_12
  simp only [← c40041, ← c40042, ← c40043] at e_635_12
  have e_635_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_13
  simp only [← c40044, ← c40045, k2, ← c40046] at e_635_13
  have hr := e_635_13
  simp only [e_635_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3755)) := by
  have c40047 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40048 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40049 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40050 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40051 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40052 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_14
  simp only [← c40047, ← c40048, ← c40049] at e_635_14
  have e_635_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_15
  simp only [← c40050, ← c40051, k2, ← c40052] at e_635_15
  have hr := e_635_15
  simp only [e_635_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3756)) := by
  have c40053 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40054 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40055 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40056 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40057 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40058 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_16
  simp only [← c40053, ← c40054, ← c40055] at e_635_16
  have e_635_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_17
  simp only [← c40056, ← c40057, k2, ← c40058] at e_635_17
  have hr := e_635_17
  simp only [e_635_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 635 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3757)) := by
  have c40059 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40060 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40061 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40062 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40063 := (publicBatchWrapper86_copies1251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40064 := (publicBatchWrapper86_copies1252 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_635_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_18
  simp only [← c40059, ← c40060, ← c40061] at e_635_18
  have e_635_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_635_19
  simp only [← c40062, ← c40063, k2, ← c40064] at e_635_19
  have hr := e_635_19
  simp only [e_635_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3758)) := by
  have c40065 := (publicBatchWrapper86_copies1252 a h).2.1
  have c40066 := (publicBatchWrapper86_copies1252 a h).2.2.1
  have c40067 := (publicBatchWrapper86_copies1252 a h).2.2.2.1
  have c40068 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.1
  have c40069 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.1
  have c40070 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_0
  simp only [← c40065, ← c40066, ← c40067] at e_636_0
  have e_636_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_1
  simp only [← c40068, ← c40069, k2, ← c40070] at e_636_1
  have hr := e_636_1
  simp only [e_636_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3759)) := by
  have c40071 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.1
  have c40072 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.1
  have c40073 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.1
  have c40074 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40075 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40076 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_2
  simp only [← c40071, ← c40072, ← c40073] at e_636_2
  have e_636_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_3
  simp only [← c40074, ← c40075, k2, ← c40076] at e_636_3
  have hr := e_636_3
  simp only [e_636_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3760)) := by
  have c40077 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40078 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40079 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40080 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40081 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40082 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_4
  simp only [← c40077, ← c40078, ← c40079] at e_636_4
  have e_636_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_5
  simp only [← c40080, ← c40081, k2, ← c40082] at e_636_5
  have hr := e_636_5
  simp only [e_636_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3761)) := by
  have c40083 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40084 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40085 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40086 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40087 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40088 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_6
  simp only [← c40083, ← c40084, ← c40085] at e_636_6
  have e_636_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_7
  simp only [← c40086, ← c40087, k2, ← c40088] at e_636_7
  have hr := e_636_7
  simp only [e_636_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3762)) := by
  have c40089 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40090 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40091 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40092 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40093 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40094 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_8
  simp only [← c40089, ← c40090, ← c40091] at e_636_8
  have e_636_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_9
  simp only [← c40092, ← c40093, k2, ← c40094] at e_636_9
  have hr := e_636_9
  simp only [e_636_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3763)) := by
  have c40095 := (publicBatchWrapper86_copies1252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40096 := (publicBatchWrapper86_copies1253 a h).1
  have c40097 := (publicBatchWrapper86_copies1253 a h).2.1
  have c40098 := (publicBatchWrapper86_copies1253 a h).2.2.1
  have c40099 := (publicBatchWrapper86_copies1253 a h).2.2.2.1
  have c40100 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_10
  simp only [← c40095, ← c40096, ← c40097] at e_636_10
  have e_636_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_11
  simp only [← c40098, ← c40099, k2, ← c40100] at e_636_11
  have hr := e_636_11
  simp only [e_636_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3764)) := by
  have c40101 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.1
  have c40102 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.1
  have c40103 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.1
  have c40104 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.1
  have c40105 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.1
  have c40106 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_12
  simp only [← c40101, ← c40102, ← c40103] at e_636_12
  have e_636_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_13
  simp only [← c40104, ← c40105, k2, ← c40106] at e_636_13
  have hr := e_636_13
  simp only [e_636_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3765)) := by
  have c40107 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40108 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40109 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40110 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40111 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40112 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_14
  simp only [← c40107, ← c40108, ← c40109] at e_636_14
  have e_636_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_15
  simp only [← c40110, ← c40111, k2, ← c40112] at e_636_15
  have hr := e_636_15
  simp only [e_636_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3766)) := by
  have c40113 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40114 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40115 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40116 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40117 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40118 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_16
  simp only [← c40113, ← c40114, ← c40115] at e_636_16
  have e_636_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_17
  simp only [← c40116, ← c40117, k2, ← c40118] at e_636_17
  have hr := e_636_17
  simp only [e_636_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 636 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3767)) := by
  have c40119 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40120 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40121 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40122 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40123 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40124 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_636_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_18
  simp only [← c40119, ← c40120, ← c40121] at e_636_18
  have e_636_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_636_19
  simp only [← c40122, ← c40123, k2, ← c40124] at e_636_19
  have hr := e_636_19
  simp only [e_636_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3768)) := by
  have c40125 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40126 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40127 := (publicBatchWrapper86_copies1253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40128 := (publicBatchWrapper86_copies1254 a h).1
  have c40129 := (publicBatchWrapper86_copies1254 a h).2.1
  have c40130 := (publicBatchWrapper86_copies1254 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_0
  simp only [← c40125, ← c40126, ← c40127] at e_637_0
  have e_637_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_1
  simp only [← c40128, ← c40129, k2, ← c40130] at e_637_1
  have hr := e_637_1
  simp only [e_637_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3769)) := by
  have c40131 := (publicBatchWrapper86_copies1254 a h).2.2.2.1
  have c40132 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.1
  have c40133 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.1
  have c40134 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.1
  have c40135 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.1
  have c40136 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_2
  simp only [← c40131, ← c40132, ← c40133] at e_637_2
  have e_637_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_3
  simp only [← c40134, ← c40135, k2, ← c40136] at e_637_3
  have hr := e_637_3
  simp only [e_637_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3810)) := by
  have c40137 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.1
  have c40138 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40139 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40140 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40141 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40142 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_4
  simp only [← c40137, ← c40138, ← c40139] at e_637_4
  have e_637_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_5
  simp only [← c40140, ← c40141, k2, ← c40142] at e_637_5
  have hr := e_637_5
  simp only [e_637_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3811)) := by
  have c40143 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40144 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40145 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40146 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40147 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40148 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_6
  simp only [← c40143, ← c40144, ← c40145] at e_637_6
  have e_637_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_7
  simp only [← c40146, ← c40147, k2, ← c40148] at e_637_7
  have hr := e_637_7
  simp only [e_637_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3812)) := by
  have c40149 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40150 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40151 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40152 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40153 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40154 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_8
  simp only [← c40149, ← c40150, ← c40151] at e_637_8
  have e_637_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_9
  simp only [← c40152, ← c40153, k2, ← c40154] at e_637_9
  have hr := e_637_9
  simp only [e_637_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3813)) := by
  have c40155 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40156 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40157 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40158 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40159 := (publicBatchWrapper86_copies1254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40160 := (publicBatchWrapper86_copies1255 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_10
  simp only [← c40155, ← c40156, ← c40157] at e_637_10
  have e_637_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_11
  simp only [← c40158, ← c40159, k2, ← c40160] at e_637_11
  have hr := e_637_11
  simp only [e_637_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3814)) := by
  have c40161 := (publicBatchWrapper86_copies1255 a h).2.1
  have c40162 := (publicBatchWrapper86_copies1255 a h).2.2.1
  have c40163 := (publicBatchWrapper86_copies1255 a h).2.2.2.1
  have c40164 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.1
  have c40165 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.1
  have c40166 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_12
  simp only [← c40161, ← c40162, ← c40163] at e_637_12
  have e_637_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_13
  simp only [← c40164, ← c40165, k2, ← c40166] at e_637_13
  have hr := e_637_13
  simp only [e_637_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3815)) := by
  have c40167 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.1
  have c40168 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.1
  have c40169 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.1
  have c40170 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40171 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40172 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_14
  simp only [← c40167, ← c40168, ← c40169] at e_637_14
  have e_637_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_15
  simp only [← c40170, ← c40171, k2, ← c40172] at e_637_15
  have hr := e_637_15
  simp only [e_637_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3816)) := by
  have c40173 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40174 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40175 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40176 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40177 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40178 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_16
  simp only [← c40173, ← c40174, ← c40175] at e_637_16
  have e_637_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_17
  simp only [← c40176, ← c40177, k2, ← c40178] at e_637_17
  have hr := e_637_17
  simp only [e_637_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 637 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3817)) := by
  have c40179 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40180 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40181 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40182 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40183 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40184 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_637_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_18
  simp only [← c40179, ← c40180, ← c40181] at e_637_18
  have e_637_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_637_19
  simp only [← c40182, ← c40183, k2, ← c40184] at e_637_19
  have hr := e_637_19
  simp only [e_637_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3818)) := by
  have c40185 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40186 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40187 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40188 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40189 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40190 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_0
  simp only [← c40185, ← c40186, ← c40187] at e_638_0
  have e_638_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_1
  simp only [← c40188, ← c40189, k2, ← c40190] at e_638_1
  have hr := e_638_1
  simp only [e_638_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3819)) := by
  have c40191 := (publicBatchWrapper86_copies1255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40192 := (publicBatchWrapper86_copies1256 a h).1
  have c40193 := (publicBatchWrapper86_copies1256 a h).2.1
  have c40194 := (publicBatchWrapper86_copies1256 a h).2.2.1
  have c40195 := (publicBatchWrapper86_copies1256 a h).2.2.2.1
  have c40196 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_2
  simp only [← c40191, ← c40192, ← c40193] at e_638_2
  have e_638_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_3
  simp only [← c40194, ← c40195, k2, ← c40196] at e_638_3
  have hr := e_638_3
  simp only [e_638_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3820)) := by
  have c40197 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.1
  have c40198 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.1
  have c40199 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.1
  have c40200 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.1
  have c40201 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.1
  have c40202 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_4
  simp only [← c40197, ← c40198, ← c40199] at e_638_4
  have e_638_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_5
  simp only [← c40200, ← c40201, k2, ← c40202] at e_638_5
  have hr := e_638_5
  simp only [e_638_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3821)) := by
  have c40203 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40204 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40205 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40206 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40207 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40208 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_6
  simp only [← c40203, ← c40204, ← c40205] at e_638_6
  have e_638_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_7
  simp only [← c40206, ← c40207, k2, ← c40208] at e_638_7
  have hr := e_638_7
  simp only [e_638_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3822)) := by
  have c40209 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40210 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40211 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40212 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40213 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40214 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_8
  simp only [← c40209, ← c40210, ← c40211] at e_638_8
  have e_638_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_9
  simp only [← c40212, ← c40213, k2, ← c40214] at e_638_9
  have hr := e_638_9
  simp only [e_638_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3823)) := by
  have c40215 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40216 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40217 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40218 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40219 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40220 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_10
  simp only [← c40215, ← c40216, ← c40217] at e_638_10
  have e_638_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_11
  simp only [← c40218, ← c40219, k2, ← c40220] at e_638_11
  have hr := e_638_11
  simp only [e_638_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3824)) := by
  have c40221 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40222 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40223 := (publicBatchWrapper86_copies1256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40224 := (publicBatchWrapper86_copies1257 a h).1
  have c40225 := (publicBatchWrapper86_copies1257 a h).2.1
  have c40226 := (publicBatchWrapper86_copies1257 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_12
  simp only [← c40221, ← c40222, ← c40223] at e_638_12
  have e_638_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_13
  simp only [← c40224, ← c40225, k2, ← c40226] at e_638_13
  have hr := e_638_13
  simp only [e_638_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3825)) := by
  have c40227 := (publicBatchWrapper86_copies1257 a h).2.2.2.1
  have c40228 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.1
  have c40229 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.1
  have c40230 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.1
  have c40231 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.1
  have c40232 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_14
  simp only [← c40227, ← c40228, ← c40229] at e_638_14
  have e_638_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_15
  simp only [← c40230, ← c40231, k2, ← c40232] at e_638_15
  have hr := e_638_15
  simp only [e_638_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3826)) := by
  have c40233 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.1
  have c40234 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40235 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40236 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40237 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40238 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_16
  simp only [← c40233, ← c40234, ← c40235] at e_638_16
  have e_638_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_17
  simp only [← c40236, ← c40237, k2, ← c40238] at e_638_17
  have hr := e_638_17
  simp only [e_638_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 638 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3827)) := by
  have c40239 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40240 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40241 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40242 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40243 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40244 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_638_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_18
  simp only [← c40239, ← c40240, ← c40241] at e_638_18
  have e_638_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_638_19
  simp only [← c40242, ← c40243, k2, ← c40244] at e_638_19
  have hr := e_638_19
  simp only [e_638_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3828)) := by
  have c40245 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40246 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40247 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40248 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40249 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40250 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_0
  simp only [← c40245, ← c40246, ← c40247] at e_639_0
  have e_639_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_1
  simp only [← c40248, ← c40249, k2, ← c40250] at e_639_1
  have hr := e_639_1
  simp only [e_639_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3829)) := by
  have c40251 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40252 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40253 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40254 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40255 := (publicBatchWrapper86_copies1257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40256 := (publicBatchWrapper86_copies1258 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_2
  simp only [← c40251, ← c40252, ← c40253] at e_639_2
  have e_639_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_3
  simp only [← c40254, ← c40255, k2, ← c40256] at e_639_3
  have hr := e_639_3
  simp only [e_639_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3830)) := by
  have c40257 := (publicBatchWrapper86_copies1258 a h).2.1
  have c40258 := (publicBatchWrapper86_copies1258 a h).2.2.1
  have c40259 := (publicBatchWrapper86_copies1258 a h).2.2.2.1
  have c40260 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.1
  have c40261 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.1
  have c40262 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_4
  simp only [← c40257, ← c40258, ← c40259] at e_639_4
  have e_639_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_5
  simp only [← c40260, ← c40261, k2, ← c40262] at e_639_5
  have hr := e_639_5
  simp only [e_639_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3831)) := by
  have c40263 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.1
  have c40264 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.1
  have c40265 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.1
  have c40266 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40267 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40268 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_6
  simp only [← c40263, ← c40264, ← c40265] at e_639_6
  have e_639_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_7
  simp only [← c40266, ← c40267, k2, ← c40268] at e_639_7
  have hr := e_639_7
  simp only [e_639_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3832)) := by
  have c40269 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40270 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40271 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40272 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40273 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40274 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_8
  simp only [← c40269, ← c40270, ← c40271] at e_639_8
  have e_639_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_9
  simp only [← c40272, ← c40273, k2, ← c40274] at e_639_9
  have hr := e_639_9
  simp only [e_639_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3833)) := by
  have c40275 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40276 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40277 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40278 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40279 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40280 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_10
  simp only [← c40275, ← c40276, ← c40277] at e_639_10
  have e_639_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_11
  simp only [← c40278, ← c40279, k2, ← c40280] at e_639_11
  have hr := e_639_11
  simp only [e_639_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3834)) := by
  have c40281 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40282 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40283 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40284 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40285 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40286 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_12
  simp only [← c40281, ← c40282, ← c40283] at e_639_12
  have e_639_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_13
  simp only [← c40284, ← c40285, k2, ← c40286] at e_639_13
  have hr := e_639_13
  simp only [e_639_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3835)) := by
  have c40287 := (publicBatchWrapper86_copies1258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40288 := (publicBatchWrapper86_copies1259 a h).1
  have c40289 := (publicBatchWrapper86_copies1259 a h).2.1
  have c40290 := (publicBatchWrapper86_copies1259 a h).2.2.1
  have c40291 := (publicBatchWrapper86_copies1259 a h).2.2.2.1
  have c40292 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_14
  simp only [← c40287, ← c40288, ← c40289] at e_639_14
  have e_639_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_15
  simp only [← c40290, ← c40291, k2, ← c40292] at e_639_15
  have hr := e_639_15
  simp only [e_639_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3836)) := by
  have c40293 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.1
  have c40294 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.1
  have c40295 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.1
  have c40296 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.1
  have c40297 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.1
  have c40298 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_16
  simp only [← c40293, ← c40294, ← c40295] at e_639_16
  have e_639_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_17
  simp only [← c40296, ← c40297, k2, ← c40298] at e_639_17
  have hr := e_639_17
  simp only [e_639_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 639 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3837)) := by
  have c40299 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40300 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40301 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40302 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40303 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40304 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_639_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_18
  simp only [← c40299, ← c40300, ← c40301] at e_639_18
  have e_639_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_639_19
  simp only [← c40302, ← c40303, k2, ← c40304] at e_639_19
  have hr := e_639_19
  simp only [e_639_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3838)) := by
  have c40305 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40306 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40307 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40308 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40309 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40310 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_0
  simp only [← c40305, ← c40306, ← c40307] at e_640_0
  have e_640_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_1
  simp only [← c40308, ← c40309, k2, ← c40310] at e_640_1
  have hr := e_640_1
  simp only [e_640_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3839)) := by
  have c40311 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40312 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40313 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40314 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40315 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40316 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_2
  simp only [← c40311, ← c40312, ← c40313] at e_640_2
  have e_640_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_3
  simp only [← c40314, ← c40315, k2, ← c40316] at e_640_3
  have hr := e_640_3
  simp only [e_640_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3840)) := by
  have c40317 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40318 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40319 := (publicBatchWrapper86_copies1259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40320 := (publicBatchWrapper86_copies1260 a h).1
  have c40321 := (publicBatchWrapper86_copies1260 a h).2.1
  have c40322 := (publicBatchWrapper86_copies1260 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_4
  simp only [← c40317, ← c40318, ← c40319] at e_640_4
  have e_640_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_5
  simp only [← c40320, ← c40321, k2, ← c40322] at e_640_5
  have hr := e_640_5
  simp only [e_640_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3841)) := by
  have c40323 := (publicBatchWrapper86_copies1260 a h).2.2.2.1
  have c40324 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.1
  have c40325 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.1
  have c40326 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.1
  have c40327 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.1
  have c40328 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_6
  simp only [← c40323, ← c40324, ← c40325] at e_640_6
  have e_640_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_7
  simp only [← c40326, ← c40327, k2, ← c40328] at e_640_7
  have hr := e_640_7
  simp only [e_640_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3842)) := by
  have c40329 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.1
  have c40330 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40331 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40332 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40333 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40334 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_8
  simp only [← c40329, ← c40330, ← c40331] at e_640_8
  have e_640_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_9
  simp only [← c40332, ← c40333, k2, ← c40334] at e_640_9
  have hr := e_640_9
  simp only [e_640_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3843)) := by
  have c40335 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40336 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40337 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40338 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40339 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40340 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_10
  simp only [← c40335, ← c40336, ← c40337] at e_640_10
  have e_640_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_11
  simp only [← c40338, ← c40339, k2, ← c40340] at e_640_11
  have hr := e_640_11
  simp only [e_640_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3844)) := by
  have c40341 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40342 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40343 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40344 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40345 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40346 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_12
  simp only [← c40341, ← c40342, ← c40343] at e_640_12
  have e_640_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_13
  simp only [← c40344, ← c40345, k2, ← c40346] at e_640_13
  have hr := e_640_13
  simp only [e_640_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3845)) := by
  have c40347 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40348 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40349 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40350 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40351 := (publicBatchWrapper86_copies1260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40352 := (publicBatchWrapper86_copies1261 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_14
  simp only [← c40347, ← c40348, ← c40349] at e_640_14
  have e_640_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_15
  simp only [← c40350, ← c40351, k2, ← c40352] at e_640_15
  have hr := e_640_15
  simp only [e_640_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3846)) := by
  have c40353 := (publicBatchWrapper86_copies1261 a h).2.1
  have c40354 := (publicBatchWrapper86_copies1261 a h).2.2.1
  have c40355 := (publicBatchWrapper86_copies1261 a h).2.2.2.1
  have c40356 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.1
  have c40357 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.1
  have c40358 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_16
  simp only [← c40353, ← c40354, ← c40355] at e_640_16
  have e_640_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_17
  simp only [← c40356, ← c40357, k2, ← c40358] at e_640_17
  have hr := e_640_17
  simp only [e_640_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 640 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3847)) := by
  have c40359 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.1
  have c40360 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.1
  have c40361 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.1
  have c40362 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40363 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40364 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_640_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_18
  simp only [← c40359, ← c40360, ← c40361] at e_640_18
  have e_640_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_640_19
  simp only [← c40362, ← c40363, k2, ← c40364] at e_640_19
  have hr := e_640_19
  simp only [e_640_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3848)) := by
  have c40365 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40366 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40367 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40368 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40369 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40370 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_0
  simp only [← c40365, ← c40366, ← c40367] at e_641_0
  have e_641_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_1
  simp only [← c40368, ← c40369, k2, ← c40370] at e_641_1
  have hr := e_641_1
  simp only [e_641_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3849)) := by
  have c40371 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40372 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40373 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40374 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40375 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40376 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_2
  simp only [← c40371, ← c40372, ← c40373] at e_641_2
  have e_641_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_3
  simp only [← c40374, ← c40375, k2, ← c40376] at e_641_3
  have hr := e_641_3
  simp only [e_641_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3850)) := by
  have c40377 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40378 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40379 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40380 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40381 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40382 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_4
  simp only [← c40377, ← c40378, ← c40379] at e_641_4
  have e_641_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_5
  simp only [← c40380, ← c40381, k2, ← c40382] at e_641_5
  have hr := e_641_5
  simp only [e_641_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3851)) := by
  have c40383 := (publicBatchWrapper86_copies1261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40384 := (publicBatchWrapper86_copies1262 a h).1
  have c40385 := (publicBatchWrapper86_copies1262 a h).2.1
  have c40386 := (publicBatchWrapper86_copies1262 a h).2.2.1
  have c40387 := (publicBatchWrapper86_copies1262 a h).2.2.2.1
  have c40388 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_6
  simp only [← c40383, ← c40384, ← c40385] at e_641_6
  have e_641_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_7
  simp only [← c40386, ← c40387, k2, ← c40388] at e_641_7
  have hr := e_641_7
  simp only [e_641_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3852)) := by
  have c40389 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.1
  have c40390 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.1
  have c40391 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.1
  have c40392 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.1
  have c40393 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.1
  have c40394 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_8
  simp only [← c40389, ← c40390, ← c40391] at e_641_8
  have e_641_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_9
  simp only [← c40392, ← c40393, k2, ← c40394] at e_641_9
  have hr := e_641_9
  simp only [e_641_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3853)) := by
  have c40395 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40396 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40397 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40398 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40399 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40400 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_10
  simp only [← c40395, ← c40396, ← c40397] at e_641_10
  have e_641_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_11
  simp only [← c40398, ← c40399, k2, ← c40400] at e_641_11
  have hr := e_641_11
  simp only [e_641_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3854)) := by
  have c40401 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40402 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40403 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40404 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40405 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40406 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_12
  simp only [← c40401, ← c40402, ← c40403] at e_641_12
  have e_641_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_13
  simp only [← c40404, ← c40405, k2, ← c40406] at e_641_13
  have hr := e_641_13
  simp only [e_641_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3855)) := by
  have c40407 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40408 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40409 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40410 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40411 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40412 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_14
  simp only [← c40407, ← c40408, ← c40409] at e_641_14
  have e_641_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_15
  simp only [← c40410, ← c40411, k2, ← c40412] at e_641_15
  have hr := e_641_15
  simp only [e_641_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3856)) := by
  have c40413 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40414 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40415 := (publicBatchWrapper86_copies1262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40416 := (publicBatchWrapper86_copies1263 a h).1
  have c40417 := (publicBatchWrapper86_copies1263 a h).2.1
  have c40418 := (publicBatchWrapper86_copies1263 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_16
  simp only [← c40413, ← c40414, ← c40415] at e_641_16
  have e_641_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_17
  simp only [← c40416, ← c40417, k2, ← c40418] at e_641_17
  have hr := e_641_17
  simp only [e_641_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 641 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3857)) := by
  have c40419 := (publicBatchWrapper86_copies1263 a h).2.2.2.1
  have c40420 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.1
  have c40421 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.1
  have c40422 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.1
  have c40423 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.1
  have c40424 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_641_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_18
  simp only [← c40419, ← c40420, ← c40421] at e_641_18
  have e_641_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_641_19
  simp only [← c40422, ← c40423, k2, ← c40424] at e_641_19
  have hr := e_641_19
  simp only [e_641_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3858)) := by
  have c40425 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.1
  have c40426 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40427 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40428 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40429 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40430 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_0
  simp only [← c40425, ← c40426, ← c40427] at e_642_0
  have e_642_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_1
  simp only [← c40428, ← c40429, k2, ← c40430] at e_642_1
  have hr := e_642_1
  simp only [e_642_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3859)) := by
  have c40431 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40432 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40433 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40434 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40435 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40436 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_2
  simp only [← c40431, ← c40432, ← c40433] at e_642_2
  have e_642_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_3
  simp only [← c40434, ← c40435, k2, ← c40436] at e_642_3
  have hr := e_642_3
  simp only [e_642_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3860)) := by
  have c40437 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40438 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40439 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40440 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40441 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40442 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_4
  simp only [← c40437, ← c40438, ← c40439] at e_642_4
  have e_642_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_5
  simp only [← c40440, ← c40441, k2, ← c40442] at e_642_5
  have hr := e_642_5
  simp only [e_642_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3861)) := by
  have c40443 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40444 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40445 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40446 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40447 := (publicBatchWrapper86_copies1263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40448 := (publicBatchWrapper86_copies1264 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_6
  simp only [← c40443, ← c40444, ← c40445] at e_642_6
  have e_642_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_7
  simp only [← c40446, ← c40447, k2, ← c40448] at e_642_7
  have hr := e_642_7
  simp only [e_642_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3862)) := by
  have c40449 := (publicBatchWrapper86_copies1264 a h).2.1
  have c40450 := (publicBatchWrapper86_copies1264 a h).2.2.1
  have c40451 := (publicBatchWrapper86_copies1264 a h).2.2.2.1
  have c40452 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.1
  have c40453 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.1
  have c40454 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_8
  simp only [← c40449, ← c40450, ← c40451] at e_642_8
  have e_642_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_9
  simp only [← c40452, ← c40453, k2, ← c40454] at e_642_9
  have hr := e_642_9
  simp only [e_642_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3863)) := by
  have c40455 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.1
  have c40456 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.1
  have c40457 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.1
  have c40458 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40459 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40460 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_10
  simp only [← c40455, ← c40456, ← c40457] at e_642_10
  have e_642_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_11
  simp only [← c40458, ← c40459, k2, ← c40460] at e_642_11
  have hr := e_642_11
  simp only [e_642_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3864)) := by
  have c40461 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40462 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40463 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40464 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40465 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40466 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_12
  simp only [← c40461, ← c40462, ← c40463] at e_642_12
  have e_642_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_13
  simp only [← c40464, ← c40465, k2, ← c40466] at e_642_13
  have hr := e_642_13
  simp only [e_642_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3865)) := by
  have c40467 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40468 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40469 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40470 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40471 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40472 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_14
  simp only [← c40467, ← c40468, ← c40469] at e_642_14
  have e_642_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_15
  simp only [← c40470, ← c40471, k2, ← c40472] at e_642_15
  have hr := e_642_15
  simp only [e_642_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3866)) := by
  have c40473 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40474 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40475 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40476 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40477 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40478 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_16
  simp only [← c40473, ← c40474, ← c40475] at e_642_16
  have e_642_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_17
  simp only [← c40476, ← c40477, k2, ← c40478] at e_642_17
  have hr := e_642_17
  simp only [e_642_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 642 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3867)) := by
  have c40479 := (publicBatchWrapper86_copies1264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40480 := (publicBatchWrapper86_copies1265 a h).1
  have c40481 := (publicBatchWrapper86_copies1265 a h).2.1
  have c40482 := (publicBatchWrapper86_copies1265 a h).2.2.1
  have c40483 := (publicBatchWrapper86_copies1265 a h).2.2.2.1
  have c40484 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_642_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_18
  simp only [← c40479, ← c40480, ← c40481] at e_642_18
  have e_642_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_642_19
  simp only [← c40482, ← c40483, k2, ← c40484] at e_642_19
  have hr := e_642_19
  simp only [e_642_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3868)) := by
  have c40485 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.1
  have c40486 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.1
  have c40487 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.1
  have c40488 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.1
  have c40489 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.1
  have c40490 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_0
  simp only [← c40485, ← c40486, ← c40487] at e_643_0
  have e_643_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_1
  simp only [← c40488, ← c40489, k2, ← c40490] at e_643_1
  have hr := e_643_1
  simp only [e_643_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3869)) := by
  have c40491 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40492 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40493 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40494 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40495 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40496 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_2
  simp only [← c40491, ← c40492, ← c40493] at e_643_2
  have e_643_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_3
  simp only [← c40494, ← c40495, k2, ← c40496] at e_643_3
  have hr := e_643_3
  simp only [e_643_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3870)) := by
  have c40497 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40498 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40499 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40500 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40501 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40502 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_4
  simp only [← c40497, ← c40498, ← c40499] at e_643_4
  have e_643_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_5
  simp only [← c40500, ← c40501, k2, ← c40502] at e_643_5
  have hr := e_643_5
  simp only [e_643_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3871)) := by
  have c40503 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40504 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40505 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40506 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40507 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40508 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_6
  simp only [← c40503, ← c40504, ← c40505] at e_643_6
  have e_643_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_7
  simp only [← c40506, ← c40507, k2, ← c40508] at e_643_7
  have hr := e_643_7
  simp only [e_643_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3872)) := by
  have c40509 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40510 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40511 := (publicBatchWrapper86_copies1265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40512 := (publicBatchWrapper86_copies1266 a h).1
  have c40513 := (publicBatchWrapper86_copies1266 a h).2.1
  have c40514 := (publicBatchWrapper86_copies1266 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_8
  simp only [← c40509, ← c40510, ← c40511] at e_643_8
  have e_643_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_9
  simp only [← c40512, ← c40513, k2, ← c40514] at e_643_9
  have hr := e_643_9
  simp only [e_643_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3873)) := by
  have c40515 := (publicBatchWrapper86_copies1266 a h).2.2.2.1
  have c40516 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.1
  have c40517 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.1
  have c40518 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.1
  have c40519 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.1
  have c40520 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_10
  simp only [← c40515, ← c40516, ← c40517] at e_643_10
  have e_643_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_11
  simp only [← c40518, ← c40519, k2, ← c40520] at e_643_11
  have hr := e_643_11
  simp only [e_643_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3874)) := by
  have c40521 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.1
  have c40522 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40523 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40524 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40525 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40526 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_12
  simp only [← c40521, ← c40522, ← c40523] at e_643_12
  have e_643_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_13
  simp only [← c40524, ← c40525, k2, ← c40526] at e_643_13
  have hr := e_643_13
  simp only [e_643_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3875)) := by
  have c40527 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40528 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40529 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40530 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40531 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40532 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_14
  simp only [← c40527, ← c40528, ← c40529] at e_643_14
  have e_643_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_15
  simp only [← c40530, ← c40531, k2, ← c40532] at e_643_15
  have hr := e_643_15
  simp only [e_643_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3876)) := by
  have c40533 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40534 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40535 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40536 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40537 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40538 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_16
  simp only [← c40533, ← c40534, ← c40535] at e_643_16
  have e_643_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_17
  simp only [← c40536, ← c40537, k2, ← c40538] at e_643_17
  have hr := e_643_17
  simp only [e_643_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 643 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3877)) := by
  have c40539 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40540 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40541 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40542 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40543 := (publicBatchWrapper86_copies1266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40544 := (publicBatchWrapper86_copies1267 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_643_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_18
  simp only [← c40539, ← c40540, ← c40541] at e_643_18
  have e_643_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_19
  simp only [← c40542, ← c40543, k2, ← c40544] at e_643_19
  have hr := e_643_19
  simp only [e_643_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3878)) := by
  have c40545 := (publicBatchWrapper86_copies1267 a h).2.1
  have c40546 := (publicBatchWrapper86_copies1267 a h).2.2.1
  have c40547 := (publicBatchWrapper86_copies1267 a h).2.2.2.1
  have c40548 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.1
  have c40549 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.1
  have c40550 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_0
  simp only [← c40545, ← c40546, ← c40547] at e_644_0
  have e_644_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_1
  simp only [← c40548, ← c40549, k2, ← c40550] at e_644_1
  have hr := e_644_1
  simp only [e_644_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3879)) := by
  have c40551 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.1
  have c40552 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.1
  have c40553 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.1
  have c40554 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40555 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40556 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_2
  simp only [← c40551, ← c40552, ← c40553] at e_644_2
  have e_644_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_3
  simp only [← c40554, ← c40555, k2, ← c40556] at e_644_3
  have hr := e_644_3
  simp only [e_644_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3880)) := by
  have c40557 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40558 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40559 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40560 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40561 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40562 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_4
  simp only [← c40557, ← c40558, ← c40559] at e_644_4
  have e_644_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_5
  simp only [← c40560, ← c40561, k2, ← c40562] at e_644_5
  have hr := e_644_5
  simp only [e_644_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3881)) := by
  have c40563 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40564 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40565 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40566 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40567 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40568 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_6
  simp only [← c40563, ← c40564, ← c40565] at e_644_6
  have e_644_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_7
  simp only [← c40566, ← c40567, k2, ← c40568] at e_644_7
  have hr := e_644_7
  simp only [e_644_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3882)) := by
  have c40569 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40570 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40571 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40572 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40573 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40574 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_8
  simp only [← c40569, ← c40570, ← c40571] at e_644_8
  have e_644_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_9
  simp only [← c40572, ← c40573, k2, ← c40574] at e_644_9
  have hr := e_644_9
  simp only [e_644_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3883)) := by
  have c40575 := (publicBatchWrapper86_copies1267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40576 := (publicBatchWrapper86_copies1268 a h).1
  have c40577 := (publicBatchWrapper86_copies1268 a h).2.1
  have c40578 := (publicBatchWrapper86_copies1268 a h).2.2.1
  have c40579 := (publicBatchWrapper86_copies1268 a h).2.2.2.1
  have c40580 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_10
  simp only [← c40575, ← c40576, ← c40577] at e_644_10
  have e_644_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_11
  simp only [← c40578, ← c40579, k2, ← c40580] at e_644_11
  have hr := e_644_11
  simp only [e_644_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3884)) := by
  have c40581 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.1
  have c40582 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.1
  have c40583 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.1
  have c40584 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.1
  have c40585 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.1
  have c40586 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_12
  simp only [← c40581, ← c40582, ← c40583] at e_644_12
  have e_644_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_13
  simp only [← c40584, ← c40585, k2, ← c40586] at e_644_13
  have hr := e_644_13
  simp only [e_644_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3885)) := by
  have c40587 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40588 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40589 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40590 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40591 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40592 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_14
  simp only [← c40587, ← c40588, ← c40589] at e_644_14
  have e_644_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_15
  simp only [← c40590, ← c40591, k2, ← c40592] at e_644_15
  have hr := e_644_15
  simp only [e_644_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3886)) := by
  have c40593 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40594 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40595 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40596 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40597 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40598 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_16
  simp only [← c40593, ← c40594, ← c40595] at e_644_16
  have e_644_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_17
  simp only [← c40596, ← c40597, k2, ← c40598] at e_644_17
  have hr := e_644_17
  simp only [e_644_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 644 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3887)) := by
  have c40599 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40600 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40601 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40602 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40603 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40604 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_644_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_18
  simp only [← c40599, ← c40600, ← c40601] at e_644_18
  have e_644_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_19
  simp only [← c40602, ← c40603, k2, ← c40604] at e_644_19
  have hr := e_644_19
  simp only [e_644_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3888)) := by
  have c40605 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40606 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40607 := (publicBatchWrapper86_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40608 := (publicBatchWrapper86_copies1269 a h).1
  have c40609 := (publicBatchWrapper86_copies1269 a h).2.1
  have c40610 := (publicBatchWrapper86_copies1269 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_0
  simp only [← c40605, ← c40606, ← c40607] at e_645_0
  have e_645_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_1
  simp only [← c40608, ← c40609, k2, ← c40610] at e_645_1
  have hr := e_645_1
  simp only [e_645_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3889)) := by
  have c40611 := (publicBatchWrapper86_copies1269 a h).2.2.2.1
  have c40612 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.1
  have c40613 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.1
  have c40614 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.1
  have c40615 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.1
  have c40616 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_2
  simp only [← c40611, ← c40612, ← c40613] at e_645_2
  have e_645_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_3
  simp only [← c40614, ← c40615, k2, ← c40616] at e_645_3
  have hr := e_645_3
  simp only [e_645_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3890)) := by
  have c40617 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.1
  have c40618 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40619 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40620 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40621 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40622 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_4
  simp only [← c40617, ← c40618, ← c40619] at e_645_4
  have e_645_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_5
  simp only [← c40620, ← c40621, k2, ← c40622] at e_645_5
  have hr := e_645_5
  simp only [e_645_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3891)) := by
  have c40623 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40624 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40625 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40626 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40627 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40628 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_6
  simp only [← c40623, ← c40624, ← c40625] at e_645_6
  have e_645_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_7
  simp only [← c40626, ← c40627, k2, ← c40628] at e_645_7
  have hr := e_645_7
  simp only [e_645_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3892)) := by
  have c40629 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40630 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40631 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40632 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40633 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40634 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_8
  simp only [← c40629, ← c40630, ← c40631] at e_645_8
  have e_645_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_9
  simp only [← c40632, ← c40633, k2, ← c40634] at e_645_9
  have hr := e_645_9
  simp only [e_645_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3893)) := by
  have c40635 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40636 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40637 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40638 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40639 := (publicBatchWrapper86_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40640 := (publicBatchWrapper86_copies1270 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_10
  simp only [← c40635, ← c40636, ← c40637] at e_645_10
  have e_645_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_11
  simp only [← c40638, ← c40639, k2, ← c40640] at e_645_11
  have hr := e_645_11
  simp only [e_645_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3894)) := by
  have c40641 := (publicBatchWrapper86_copies1270 a h).2.1
  have c40642 := (publicBatchWrapper86_copies1270 a h).2.2.1
  have c40643 := (publicBatchWrapper86_copies1270 a h).2.2.2.1
  have c40644 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.1
  have c40645 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.1
  have c40646 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_12
  simp only [← c40641, ← c40642, ← c40643] at e_645_12
  have e_645_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_13
  simp only [← c40644, ← c40645, k2, ← c40646] at e_645_13
  have hr := e_645_13
  simp only [e_645_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3895)) := by
  have c40647 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.1
  have c40648 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.1
  have c40649 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.1
  have c40650 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40651 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40652 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_14
  simp only [← c40647, ← c40648, ← c40649] at e_645_14
  have e_645_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_15
  simp only [← c40650, ← c40651, k2, ← c40652] at e_645_15
  have hr := e_645_15
  simp only [e_645_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3896)) := by
  have c40653 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40654 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40655 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40656 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40657 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40658 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_16
  simp only [← c40653, ← c40654, ← c40655] at e_645_16
  have e_645_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_17
  simp only [← c40656, ← c40657, k2, ← c40658] at e_645_17
  have hr := e_645_17
  simp only [e_645_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 645 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3897)) := by
  have c40659 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40660 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40661 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40662 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40663 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40664 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_645_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_18
  simp only [← c40659, ← c40660, ← c40661] at e_645_18
  have e_645_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_19
  simp only [← c40662, ← c40663, k2, ← c40664] at e_645_19
  have hr := e_645_19
  simp only [e_645_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3898)) := by
  have c40665 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40666 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40667 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40668 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40669 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40670 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_0
  simp only [← c40665, ← c40666, ← c40667] at e_646_0
  have e_646_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_1
  simp only [← c40668, ← c40669, k2, ← c40670] at e_646_1
  have hr := e_646_1
  simp only [e_646_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3899)) := by
  have c40671 := (publicBatchWrapper86_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40672 := (publicBatchWrapper86_copies1271 a h).1
  have c40673 := (publicBatchWrapper86_copies1271 a h).2.1
  have c40674 := (publicBatchWrapper86_copies1271 a h).2.2.1
  have c40675 := (publicBatchWrapper86_copies1271 a h).2.2.2.1
  have c40676 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_2
  simp only [← c40671, ← c40672, ← c40673] at e_646_2
  have e_646_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_3
  simp only [← c40674, ← c40675, k2, ← c40676] at e_646_3
  have hr := e_646_3
  simp only [e_646_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3900)) := by
  have c40677 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.1
  have c40678 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.1
  have c40679 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.1
  have c40680 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.1
  have c40681 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.1
  have c40682 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_4
  simp only [← c40677, ← c40678, ← c40679] at e_646_4
  have e_646_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_5
  simp only [← c40680, ← c40681, k2, ← c40682] at e_646_5
  have hr := e_646_5
  simp only [e_646_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3901)) := by
  have c40683 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40684 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40685 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40686 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40687 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40688 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_6
  simp only [← c40683, ← c40684, ← c40685] at e_646_6
  have e_646_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_7
  simp only [← c40686, ← c40687, k2, ← c40688] at e_646_7
  have hr := e_646_7
  simp only [e_646_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3902)) := by
  have c40689 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40690 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40691 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40692 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40693 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40694 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_8
  simp only [← c40689, ← c40690, ← c40691] at e_646_8
  have e_646_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_9
  simp only [← c40692, ← c40693, k2, ← c40694] at e_646_9
  have hr := e_646_9
  simp only [e_646_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3903)) := by
  have c40695 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40696 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40697 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40698 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40699 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40700 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_10
  simp only [← c40695, ← c40696, ← c40697] at e_646_10
  have e_646_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_11
  simp only [← c40698, ← c40699, k2, ← c40700] at e_646_11
  have hr := e_646_11
  simp only [e_646_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3904)) := by
  have c40701 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40702 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40703 := (publicBatchWrapper86_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40704 := (publicBatchWrapper86_copies1272 a h).1
  have c40705 := (publicBatchWrapper86_copies1272 a h).2.1
  have c40706 := (publicBatchWrapper86_copies1272 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_12
  simp only [← c40701, ← c40702, ← c40703] at e_646_12
  have e_646_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_13
  simp only [← c40704, ← c40705, k2, ← c40706] at e_646_13
  have hr := e_646_13
  simp only [e_646_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3905)) := by
  have c40707 := (publicBatchWrapper86_copies1272 a h).2.2.2.1
  have c40708 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.1
  have c40709 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.1
  have c40710 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.1
  have c40711 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.1
  have c40712 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_14
  simp only [← c40707, ← c40708, ← c40709] at e_646_14
  have e_646_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_15
  simp only [← c40710, ← c40711, k2, ← c40712] at e_646_15
  have hr := e_646_15
  simp only [e_646_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3906)) := by
  have c40713 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.1
  have c40714 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40715 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40716 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40717 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40718 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_16
  simp only [← c40713, ← c40714, ← c40715] at e_646_16
  have e_646_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_17
  simp only [← c40716, ← c40717, k2, ← c40718] at e_646_17
  have hr := e_646_17
  simp only [e_646_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 646 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3907)) := by
  have c40719 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40720 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40721 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40722 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40723 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40724 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_646_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_18
  simp only [← c40719, ← c40720, ← c40721] at e_646_18
  have e_646_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_19
  simp only [← c40722, ← c40723, k2, ← c40724] at e_646_19
  have hr := e_646_19
  simp only [e_646_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3908)) := by
  have c40725 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40726 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40727 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40728 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40729 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40730 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_0
  simp only [← c40725, ← c40726, ← c40727] at e_647_0
  have e_647_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_1
  simp only [← c40728, ← c40729, k2, ← c40730] at e_647_1
  have hr := e_647_1
  simp only [e_647_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3909)) := by
  have c40731 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40732 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40733 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40734 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40735 := (publicBatchWrapper86_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40736 := (publicBatchWrapper86_copies1273 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_2
  simp only [← c40731, ← c40732, ← c40733] at e_647_2
  have e_647_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_3
  simp only [← c40734, ← c40735, k2, ← c40736] at e_647_3
  have hr := e_647_3
  simp only [e_647_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3910)) := by
  have c40737 := (publicBatchWrapper86_copies1273 a h).2.1
  have c40738 := (publicBatchWrapper86_copies1273 a h).2.2.1
  have c40739 := (publicBatchWrapper86_copies1273 a h).2.2.2.1
  have c40740 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.1
  have c40741 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.1
  have c40742 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_4
  simp only [← c40737, ← c40738, ← c40739] at e_647_4
  have e_647_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_5
  simp only [← c40740, ← c40741, k2, ← c40742] at e_647_5
  have hr := e_647_5
  simp only [e_647_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3911)) := by
  have c40743 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.1
  have c40744 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.1
  have c40745 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.1
  have c40746 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40747 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40748 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_6
  simp only [← c40743, ← c40744, ← c40745] at e_647_6
  have e_647_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_7
  simp only [← c40746, ← c40747, k2, ← c40748] at e_647_7
  have hr := e_647_7
  simp only [e_647_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3912)) := by
  have c40749 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40750 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40751 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40752 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40753 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40754 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_8
  simp only [← c40749, ← c40750, ← c40751] at e_647_8
  have e_647_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_9
  simp only [← c40752, ← c40753, k2, ← c40754] at e_647_9
  have hr := e_647_9
  simp only [e_647_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3913)) := by
  have c40755 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40756 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40757 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40758 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40759 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40760 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_10
  simp only [← c40755, ← c40756, ← c40757] at e_647_10
  have e_647_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_11
  simp only [← c40758, ← c40759, k2, ← c40760] at e_647_11
  have hr := e_647_11
  simp only [e_647_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3914)) := by
  have c40761 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40762 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40763 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40764 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40765 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40766 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_12
  simp only [← c40761, ← c40762, ← c40763] at e_647_12
  have e_647_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_13
  simp only [← c40764, ← c40765, k2, ← c40766] at e_647_13
  have hr := e_647_13
  simp only [e_647_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3915)) := by
  have c40767 := (publicBatchWrapper86_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40768 := (publicBatchWrapper86_copies1274 a h).1
  have c40769 := (publicBatchWrapper86_copies1274 a h).2.1
  have c40770 := (publicBatchWrapper86_copies1274 a h).2.2.1
  have c40771 := (publicBatchWrapper86_copies1274 a h).2.2.2.1
  have c40772 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_14
  simp only [← c40767, ← c40768, ← c40769] at e_647_14
  have e_647_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_15
  simp only [← c40770, ← c40771, k2, ← c40772] at e_647_15
  have hr := e_647_15
  simp only [e_647_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3916)) := by
  have c40773 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.1
  have c40774 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.1
  have c40775 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.1
  have c40776 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.1
  have c40777 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.1
  have c40778 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_16
  simp only [← c40773, ← c40774, ← c40775] at e_647_16
  have e_647_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_17
  simp only [← c40776, ← c40777, k2, ← c40778] at e_647_17
  have hr := e_647_17
  simp only [e_647_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 647 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3917)) := by
  have c40779 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40780 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40781 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40782 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40783 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40784 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_647_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_18
  simp only [← c40779, ← c40780, ← c40781] at e_647_18
  have e_647_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_19
  simp only [← c40782, ← c40783, k2, ← c40784] at e_647_19
  have hr := e_647_19
  simp only [e_647_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3918)) := by
  have c40785 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40786 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40787 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40788 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40789 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40790 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_0
  simp only [← c40785, ← c40786, ← c40787] at e_648_0
  have e_648_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_1
  simp only [← c40788, ← c40789, k2, ← c40790] at e_648_1
  have hr := e_648_1
  simp only [e_648_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3919)) := by
  have c40791 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40792 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40793 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40794 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40795 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40796 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_2
  simp only [← c40791, ← c40792, ← c40793] at e_648_2
  have e_648_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_3
  simp only [← c40794, ← c40795, k2, ← c40796] at e_648_3
  have hr := e_648_3
  simp only [e_648_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3920)) := by
  have c40797 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40798 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40799 := (publicBatchWrapper86_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40800 := (publicBatchWrapper86_copies1275 a h).1
  have c40801 := (publicBatchWrapper86_copies1275 a h).2.1
  have c40802 := (publicBatchWrapper86_copies1275 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_4
  simp only [← c40797, ← c40798, ← c40799] at e_648_4
  have e_648_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_5
  simp only [← c40800, ← c40801, k2, ← c40802] at e_648_5
  have hr := e_648_5
  simp only [e_648_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3921)) := by
  have c40803 := (publicBatchWrapper86_copies1275 a h).2.2.2.1
  have c40804 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.1
  have c40805 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.1
  have c40806 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.1
  have c40807 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.1
  have c40808 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_6
  simp only [← c40803, ← c40804, ← c40805] at e_648_6
  have e_648_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_7
  simp only [← c40806, ← c40807, k2, ← c40808] at e_648_7
  have hr := e_648_7
  simp only [e_648_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3922)) := by
  have c40809 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.1
  have c40810 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40811 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40812 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40813 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40814 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_8
  simp only [← c40809, ← c40810, ← c40811] at e_648_8
  have e_648_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_9
  simp only [← c40812, ← c40813, k2, ← c40814] at e_648_9
  have hr := e_648_9
  simp only [e_648_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3923)) := by
  have c40815 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40816 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40817 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40818 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40819 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40820 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_10
  simp only [← c40815, ← c40816, ← c40817] at e_648_10
  have e_648_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_11
  simp only [← c40818, ← c40819, k2, ← c40820] at e_648_11
  have hr := e_648_11
  simp only [e_648_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3924)) := by
  have c40821 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40822 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40823 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40824 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40825 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40826 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_12
  simp only [← c40821, ← c40822, ← c40823] at e_648_12
  have e_648_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_13
  simp only [← c40824, ← c40825, k2, ← c40826] at e_648_13
  have hr := e_648_13
  simp only [e_648_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3925)) := by
  have c40827 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40828 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40829 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40830 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40831 := (publicBatchWrapper86_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40832 := (publicBatchWrapper86_copies1276 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_14
  simp only [← c40827, ← c40828, ← c40829] at e_648_14
  have e_648_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_15
  simp only [← c40830, ← c40831, k2, ← c40832] at e_648_15
  have hr := e_648_15
  simp only [e_648_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3926)) := by
  have c40833 := (publicBatchWrapper86_copies1276 a h).2.1
  have c40834 := (publicBatchWrapper86_copies1276 a h).2.2.1
  have c40835 := (publicBatchWrapper86_copies1276 a h).2.2.2.1
  have c40836 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.1
  have c40837 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.1
  have c40838 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_16
  simp only [← c40833, ← c40834, ← c40835] at e_648_16
  have e_648_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_17
  simp only [← c40836, ← c40837, k2, ← c40838] at e_648_17
  have hr := e_648_17
  simp only [e_648_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 648 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3927)) := by
  have c40839 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.1
  have c40840 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.1
  have c40841 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.1
  have c40842 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40843 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40844 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_648_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_18
  simp only [← c40839, ← c40840, ← c40841] at e_648_18
  have e_648_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_19
  simp only [← c40842, ← c40843, k2, ← c40844] at e_648_19
  have hr := e_648_19
  simp only [e_648_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3928)) := by
  have c40845 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40846 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40847 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40848 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40849 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40850 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_0
  simp only [← c40845, ← c40846, ← c40847] at e_649_0
  have e_649_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_1
  simp only [← c40848, ← c40849, k2, ← c40850] at e_649_1
  have hr := e_649_1
  simp only [e_649_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3929)) := by
  have c40851 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40852 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40853 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40854 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40855 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40856 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_2
  simp only [← c40851, ← c40852, ← c40853] at e_649_2
  have e_649_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_3
  simp only [← c40854, ← c40855, k2, ← c40856] at e_649_3
  have hr := e_649_3
  simp only [e_649_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3930)) := by
  have c40857 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40858 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40859 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40860 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40861 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40862 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_4
  simp only [← c40857, ← c40858, ← c40859] at e_649_4
  have e_649_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_5
  simp only [← c40860, ← c40861, k2, ← c40862] at e_649_5
  have hr := e_649_5
  simp only [e_649_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3931)) := by
  have c40863 := (publicBatchWrapper86_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40864 := (publicBatchWrapper86_copies1277 a h).1
  have c40865 := (publicBatchWrapper86_copies1277 a h).2.1
  have c40866 := (publicBatchWrapper86_copies1277 a h).2.2.1
  have c40867 := (publicBatchWrapper86_copies1277 a h).2.2.2.1
  have c40868 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_6
  simp only [← c40863, ← c40864, ← c40865] at e_649_6
  have e_649_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_7
  simp only [← c40866, ← c40867, k2, ← c40868] at e_649_7
  have hr := e_649_7
  simp only [e_649_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3932)) := by
  have c40869 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.1
  have c40870 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.1
  have c40871 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.1
  have c40872 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.1
  have c40873 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.1
  have c40874 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_8
  simp only [← c40869, ← c40870, ← c40871] at e_649_8
  have e_649_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_9
  simp only [← c40872, ← c40873, k2, ← c40874] at e_649_9
  have hr := e_649_9
  simp only [e_649_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3933)) := by
  have c40875 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40876 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40877 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40878 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40879 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40880 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_10
  simp only [← c40875, ← c40876, ← c40877] at e_649_10
  have e_649_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_11
  simp only [← c40878, ← c40879, k2, ← c40880] at e_649_11
  have hr := e_649_11
  simp only [e_649_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3934)) := by
  have c40881 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40882 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40883 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40884 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40885 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40886 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_12
  simp only [← c40881, ← c40882, ← c40883] at e_649_12
  have e_649_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_13
  simp only [← c40884, ← c40885, k2, ← c40886] at e_649_13
  have hr := e_649_13
  simp only [e_649_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3935)) := by
  have c40887 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40888 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40889 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40890 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40891 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40892 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_14
  simp only [← c40887, ← c40888, ← c40889] at e_649_14
  have e_649_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_15
  simp only [← c40890, ← c40891, k2, ← c40892] at e_649_15
  have hr := e_649_15
  simp only [e_649_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3936)) := by
  have c40893 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40894 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40895 := (publicBatchWrapper86_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40896 := (publicBatchWrapper86_copies1278 a h).1
  have c40897 := (publicBatchWrapper86_copies1278 a h).2.1
  have c40898 := (publicBatchWrapper86_copies1278 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_16
  simp only [← c40893, ← c40894, ← c40895] at e_649_16
  have e_649_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_17
  simp only [← c40896, ← c40897, k2, ← c40898] at e_649_17
  have hr := e_649_17
  simp only [e_649_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 649 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3937)) := by
  have c40899 := (publicBatchWrapper86_copies1278 a h).2.2.2.1
  have c40900 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.1
  have c40901 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.1
  have c40902 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.1
  have c40903 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.1
  have c40904 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_649_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_18
  simp only [← c40899, ← c40900, ← c40901] at e_649_18
  have e_649_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_19
  simp only [← c40902, ← c40903, k2, ← c40904] at e_649_19
  have hr := e_649_19
  simp only [e_649_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3938)) := by
  have c40905 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.1
  have c40906 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40907 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40908 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40909 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40910 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_0
  simp only [← c40905, ← c40906, ← c40907] at e_650_0
  have e_650_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_1
  simp only [← c40908, ← c40909, k2, ← c40910] at e_650_1
  have hr := e_650_1
  simp only [e_650_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3939)) := by
  have c40911 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40912 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40913 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40914 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40915 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40916 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_2
  simp only [← c40911, ← c40912, ← c40913] at e_650_2
  have e_650_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_3
  simp only [← c40914, ← c40915, k2, ← c40916] at e_650_3
  have hr := e_650_3
  simp only [e_650_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3940)) := by
  have c40917 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40918 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40919 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40920 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40921 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40922 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_4
  simp only [← c40917, ← c40918, ← c40919] at e_650_4
  have e_650_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_5
  simp only [← c40920, ← c40921, k2, ← c40922] at e_650_5
  have hr := e_650_5
  simp only [e_650_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3941)) := by
  have c40923 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40924 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40925 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40926 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40927 := (publicBatchWrapper86_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40928 := (publicBatchWrapper86_copies1279 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_6
  simp only [← c40923, ← c40924, ← c40925] at e_650_6
  have e_650_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_7
  simp only [← c40926, ← c40927, k2, ← c40928] at e_650_7
  have hr := e_650_7
  simp only [e_650_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3942)) := by
  have c40929 := (publicBatchWrapper86_copies1279 a h).2.1
  have c40930 := (publicBatchWrapper86_copies1279 a h).2.2.1
  have c40931 := (publicBatchWrapper86_copies1279 a h).2.2.2.1
  have c40932 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.1
  have c40933 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.1
  have c40934 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_8
  simp only [← c40929, ← c40930, ← c40931] at e_650_8
  have e_650_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_9
  simp only [← c40932, ← c40933, k2, ← c40934] at e_650_9
  have hr := e_650_9
  simp only [e_650_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3943)) := by
  have c40935 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.1
  have c40936 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.1
  have c40937 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.1
  have c40938 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40939 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40940 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_10
  simp only [← c40935, ← c40936, ← c40937] at e_650_10
  have e_650_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_11
  simp only [← c40938, ← c40939, k2, ← c40940] at e_650_11
  have hr := e_650_11
  simp only [e_650_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3944)) := by
  have c40941 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40942 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40943 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40944 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40945 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40946 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_12
  simp only [← c40941, ← c40942, ← c40943] at e_650_12
  have e_650_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_13
  simp only [← c40944, ← c40945, k2, ← c40946] at e_650_13
  have hr := e_650_13
  simp only [e_650_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3945)) := by
  have c40947 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40948 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40949 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40950 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40951 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40952 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_14
  simp only [← c40947, ← c40948, ← c40949] at e_650_14
  have e_650_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_15
  simp only [← c40950, ← c40951, k2, ← c40952] at e_650_15
  have hr := e_650_15
  simp only [e_650_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3946)) := by
  have c40953 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40954 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40955 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40956 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40957 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40958 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_16
  simp only [← c40953, ← c40954, ← c40955] at e_650_16
  have e_650_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_17
  simp only [← c40956, ← c40957, k2, ← c40958] at e_650_17
  have hr := e_650_17
  simp only [e_650_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 650 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3947)) := by
  have c40959 := (publicBatchWrapper86_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40960 := (publicBatchWrapper86_copies1280 a h).1
  have c40961 := (publicBatchWrapper86_copies1280 a h).2.1
  have c40962 := (publicBatchWrapper86_copies1280 a h).2.2.1
  have c40963 := (publicBatchWrapper86_copies1280 a h).2.2.2.1
  have c40964 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_650_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_18
  simp only [← c40959, ← c40960, ← c40961] at e_650_18
  have e_650_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_19
  simp only [← c40962, ← c40963, k2, ← c40964] at e_650_19
  have hr := e_650_19
  simp only [e_650_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3948)) := by
  have c40965 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.1
  have c40966 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.1
  have c40967 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.1
  have c40968 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.1
  have c40969 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.1
  have c40970 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_0
  simp only [← c40965, ← c40966, ← c40967] at e_651_0
  have e_651_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_1
  simp only [← c40968, ← c40969, k2, ← c40970] at e_651_1
  have hr := e_651_1
  simp only [e_651_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3949)) := by
  have c40971 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40972 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40973 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40974 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40975 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40976 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_2
  simp only [← c40971, ← c40972, ← c40973] at e_651_2
  have e_651_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_3
  simp only [← c40974, ← c40975, k2, ← c40976] at e_651_3
  have hr := e_651_3
  simp only [e_651_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3950)) := by
  have c40977 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40978 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40979 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40980 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40981 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40982 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_4
  simp only [← c40977, ← c40978, ← c40979] at e_651_4
  have e_651_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_5
  simp only [← c40980, ← c40981, k2, ← c40982] at e_651_5
  have hr := e_651_5
  simp only [e_651_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3951)) := by
  have c40983 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40984 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40985 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40986 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40987 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40988 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_6
  simp only [← c40983, ← c40984, ← c40985] at e_651_6
  have e_651_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_7
  simp only [← c40986, ← c40987, k2, ← c40988] at e_651_7
  have hr := e_651_7
  simp only [e_651_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3952)) := by
  have c40989 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40990 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40991 := (publicBatchWrapper86_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40992 := (publicBatchWrapper86_copies1281 a h).1
  have c40993 := (publicBatchWrapper86_copies1281 a h).2.1
  have c40994 := (publicBatchWrapper86_copies1281 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_8
  simp only [← c40989, ← c40990, ← c40991] at e_651_8
  have e_651_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_9
  simp only [← c40992, ← c40993, k2, ← c40994] at e_651_9
  have hr := e_651_9
  simp only [e_651_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3953)) := by
  have c40995 := (publicBatchWrapper86_copies1281 a h).2.2.2.1
  have c40996 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.1
  have c40997 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.1
  have c40998 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.1
  have c40999 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.1
  have c41000 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_10
  simp only [← c40995, ← c40996, ← c40997] at e_651_10
  have e_651_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_11
  simp only [← c40998, ← c40999, k2, ← c41000] at e_651_11
  have hr := e_651_11
  simp only [e_651_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3954)) := by
  have c41001 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.1
  have c41002 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41003 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41004 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41005 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41006 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_12
  simp only [← c41001, ← c41002, ← c41003] at e_651_12
  have e_651_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_13
  simp only [← c41004, ← c41005, k2, ← c41006] at e_651_13
  have hr := e_651_13
  simp only [e_651_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3955)) := by
  have c41007 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41008 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41009 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41010 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41011 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41012 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_14
  simp only [← c41007, ← c41008, ← c41009] at e_651_14
  have e_651_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_15
  simp only [← c41010, ← c41011, k2, ← c41012] at e_651_15
  have hr := e_651_15
  simp only [e_651_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3956)) := by
  have c41013 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41014 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41015 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41016 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41017 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41018 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_16
  simp only [← c41013, ← c41014, ← c41015] at e_651_16
  have e_651_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_17
  simp only [← c41016, ← c41017, k2, ← c41018] at e_651_17
  have hr := e_651_17
  simp only [e_651_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 651 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3957)) := by
  have c41019 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41020 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41021 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41022 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41023 := (publicBatchWrapper86_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41024 := (publicBatchWrapper86_copies1282 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_651_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_18
  simp only [← c41019, ← c41020, ← c41021] at e_651_18
  have e_651_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_19
  simp only [← c41022, ← c41023, k2, ← c41024] at e_651_19
  have hr := e_651_19
  simp only [e_651_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3958)) := by
  have c41025 := (publicBatchWrapper86_copies1282 a h).2.1
  have c41026 := (publicBatchWrapper86_copies1282 a h).2.2.1
  have c41027 := (publicBatchWrapper86_copies1282 a h).2.2.2.1
  have c41028 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.1
  have c41029 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.1
  have c41030 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_0
  simp only [← c41025, ← c41026, ← c41027] at e_652_0
  have e_652_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_1
  simp only [← c41028, ← c41029, k2, ← c41030] at e_652_1
  have hr := e_652_1
  simp only [e_652_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3959)) := by
  have c41031 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.1
  have c41032 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.1
  have c41033 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.1
  have c41034 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41035 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41036 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_2
  simp only [← c41031, ← c41032, ← c41033] at e_652_2
  have e_652_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_3
  simp only [← c41034, ← c41035, k2, ← c41036] at e_652_3
  have hr := e_652_3
  simp only [e_652_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 23) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3960)) := by
  have c41037 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41038 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41039 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41040 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41041 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41042 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_4
  simp only [← c41037, ← c41038, ← c41039] at e_652_4
  have e_652_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_5
  simp only [← c41040, ← c41041, k2, ← c41042] at e_652_5
  have hr := e_652_5
  simp only [e_652_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 31) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3961)) := by
  have c41043 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41044 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41045 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41046 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41047 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41048 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_6
  simp only [← c41043, ← c41044, ← c41045] at e_652_6
  have e_652_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_7
  simp only [← c41046, ← c41047, k2, ← c41048] at e_652_7
  have hr := e_652_7
  simp only [e_652_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 39) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3962)) := by
  have c41049 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41050 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41051 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41052 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41053 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41054 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_8
  simp only [← c41049, ← c41050, ← c41051] at e_652_8
  have e_652_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_9
  simp only [← c41052, ← c41053, k2, ← c41054] at e_652_9
  have hr := e_652_9
  simp only [e_652_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 47) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3963)) := by
  have c41055 := (publicBatchWrapper86_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41056 := (publicBatchWrapper86_copies1283 a h).1
  have c41057 := (publicBatchWrapper86_copies1283 a h).2.1
  have c41058 := (publicBatchWrapper86_copies1283 a h).2.2.1
  have c41059 := (publicBatchWrapper86_copies1283 a h).2.2.2.1
  have c41060 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_10
  simp only [← c41055, ← c41056, ← c41057] at e_652_10
  have e_652_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_11
  simp only [← c41058, ← c41059, k2, ← c41060] at e_652_11
  have hr := e_652_11
  simp only [e_652_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 55) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3964)) := by
  have c41061 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.1
  have c41062 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.1
  have c41063 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.1
  have c41064 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.1
  have c41065 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.1
  have c41066 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_12
  simp only [← c41061, ← c41062, ← c41063] at e_652_12
  have e_652_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_13
  simp only [← c41064, ← c41065, k2, ← c41066] at e_652_13
  have hr := e_652_13
  simp only [e_652_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 63) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3965)) := by
  have c41067 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41068 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41069 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41070 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41071 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41072 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_14
  simp only [← c41067, ← c41068, ← c41069] at e_652_14
  have e_652_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_15
  simp only [← c41070, ← c41071, k2, ← c41072] at e_652_15
  have hr := e_652_15
  simp only [e_652_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 71) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3966)) := by
  have c41073 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41074 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41075 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41076 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41077 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41078 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_16
  simp only [← c41073, ← c41074, ← c41075] at e_652_16
  have e_652_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_17
  simp only [← c41076, ← c41077, k2, ← c41078] at e_652_17
  have hr := e_652_17
  simp only [e_652_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 652 79) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3967)) := by
  have c41079 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41080 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41081 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41082 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41083 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41084 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_652_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_18
  simp only [← c41079, ← c41080, ← c41081] at e_652_18
  have e_652_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_19
  simp only [← c41082, ← c41083, k2, ← c41084] at e_652_19
  have hr := e_652_19
  simp only [e_652_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 7) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3968)) := by
  have c41085 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41086 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41087 := (publicBatchWrapper86_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41088 := (publicBatchWrapper86_copies1284 a h).1
  have c41089 := (publicBatchWrapper86_copies1284 a h).2.1
  have c41090 := (publicBatchWrapper86_copies1284 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_0
  simp only [← c41085, ← c41086, ← c41087] at e_653_0
  have e_653_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_1
  simp only [← c41088, ← c41089, k2, ← c41090] at e_653_1
  have hr := e_653_1
  simp only [e_653_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 15) = bselect (a (.wire 20 63)) (a (.virt 200)) (a (.virt 3969)) := by
  have c41091 := (publicBatchWrapper86_copies1284 a h).2.2.2.1
  have c41092 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.1
  have c41093 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.1
  have c41094 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.1
  have c41095 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.1
  have c41096 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_2
  simp only [← c41091, ← c41092, ← c41093] at e_653_2
  have e_653_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_3
  simp only [← c41094, ← c41095, k2, ← c41096] at e_653_3
  have hr := e_653_3
  simp only [e_653_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4010)) := by
  have c41097 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.1
  have c41098 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41099 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41100 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41101 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41102 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_4
  simp only [← c41097, ← c41098, ← c41099] at e_653_4
  have e_653_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_5
  simp only [← c41100, ← c41101, k2, ← c41102] at e_653_5
  have hr := e_653_5
  simp only [e_653_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4011)) := by
  have c41103 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41104 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41105 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41106 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41107 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41108 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_6
  simp only [← c41103, ← c41104, ← c41105] at e_653_6
  have e_653_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_7
  simp only [← c41106, ← c41107, k2, ← c41108] at e_653_7
  have hr := e_653_7
  simp only [e_653_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4012)) := by
  have c41109 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41110 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41111 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41112 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41113 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41114 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_8
  simp only [← c41109, ← c41110, ← c41111] at e_653_8
  have e_653_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_9
  simp only [← c41112, ← c41113, k2, ← c41114] at e_653_9
  have hr := e_653_9
  simp only [e_653_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4013)) := by
  have c41115 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41116 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41117 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41118 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41119 := (publicBatchWrapper86_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41120 := (publicBatchWrapper86_copies1285 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_10
  simp only [← c41115, ← c41116, ← c41117] at e_653_10
  have e_653_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_11
  simp only [← c41118, ← c41119, k2, ← c41120] at e_653_11
  have hr := e_653_11
  simp only [e_653_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4014)) := by
  have c41121 := (publicBatchWrapper86_copies1285 a h).2.1
  have c41122 := (publicBatchWrapper86_copies1285 a h).2.2.1
  have c41123 := (publicBatchWrapper86_copies1285 a h).2.2.2.1
  have c41124 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.1
  have c41125 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.1
  have c41126 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_12
  simp only [← c41121, ← c41122, ← c41123] at e_653_12
  have e_653_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_13
  simp only [← c41124, ← c41125, k2, ← c41126] at e_653_13
  have hr := e_653_13
  simp only [e_653_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4015)) := by
  have c41127 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.1
  have c41128 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.1
  have c41129 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.1
  have c41130 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41131 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41132 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_14
  simp only [← c41127, ← c41128, ← c41129] at e_653_14
  have e_653_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_15
  simp only [← c41130, ← c41131, k2, ← c41132] at e_653_15
  have hr := e_653_15
  simp only [e_653_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4016)) := by
  have c41133 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41134 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41135 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41136 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41137 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41138 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_16
  simp only [← c41133, ← c41134, ← c41135] at e_653_16
  have e_653_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_17
  simp only [← c41136, ← c41137, k2, ← c41138] at e_653_17
  have hr := e_653_17
  simp only [e_653_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 653 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4017)) := by
  have c41139 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41140 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41141 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41142 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41143 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41144 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_653_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_18
  simp only [← c41139, ← c41140, ← c41141] at e_653_18
  have e_653_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_19
  simp only [← c41142, ← c41143, k2, ← c41144] at e_653_19
  have hr := e_653_19
  simp only [e_653_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4018)) := by
  have c41145 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41146 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41147 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41148 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41149 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41150 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_0
  simp only [← c41145, ← c41146, ← c41147] at e_654_0
  have e_654_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_1
  simp only [← c41148, ← c41149, k2, ← c41150] at e_654_1
  have hr := e_654_1
  simp only [e_654_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4019)) := by
  have c41151 := (publicBatchWrapper86_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41152 := (publicBatchWrapper86_copies1286 a h).1
  have c41153 := (publicBatchWrapper86_copies1286 a h).2.1
  have c41154 := (publicBatchWrapper86_copies1286 a h).2.2.1
  have c41155 := (publicBatchWrapper86_copies1286 a h).2.2.2.1
  have c41156 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_2
  simp only [← c41151, ← c41152, ← c41153] at e_654_2
  have e_654_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_3
  simp only [← c41154, ← c41155, k2, ← c41156] at e_654_3
  have hr := e_654_3
  simp only [e_654_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4020)) := by
  have c41157 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.1
  have c41158 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.1
  have c41159 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.1
  have c41160 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.1
  have c41161 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.1
  have c41162 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_4
  simp only [← c41157, ← c41158, ← c41159] at e_654_4
  have e_654_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_5
  simp only [← c41160, ← c41161, k2, ← c41162] at e_654_5
  have hr := e_654_5
  simp only [e_654_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4021)) := by
  have c41163 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41164 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41165 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41166 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41167 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41168 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_6
  simp only [← c41163, ← c41164, ← c41165] at e_654_6
  have e_654_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_7
  simp only [← c41166, ← c41167, k2, ← c41168] at e_654_7
  have hr := e_654_7
  simp only [e_654_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4022)) := by
  have c41169 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41170 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41171 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41172 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41173 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41174 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_8
  simp only [← c41169, ← c41170, ← c41171] at e_654_8
  have e_654_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_9
  simp only [← c41172, ← c41173, k2, ← c41174] at e_654_9
  have hr := e_654_9
  simp only [e_654_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4023)) := by
  have c41175 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41176 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41177 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41178 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41179 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41180 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_10
  simp only [← c41175, ← c41176, ← c41177] at e_654_10
  have e_654_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_11
  simp only [← c41178, ← c41179, k2, ← c41180] at e_654_11
  have hr := e_654_11
  simp only [e_654_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4024)) := by
  have c41181 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41182 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41183 := (publicBatchWrapper86_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41184 := (publicBatchWrapper86_copies1287 a h).1
  have c41185 := (publicBatchWrapper86_copies1287 a h).2.1
  have c41186 := (publicBatchWrapper86_copies1287 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_12
  simp only [← c41181, ← c41182, ← c41183] at e_654_12
  have e_654_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_13
  simp only [← c41184, ← c41185, k2, ← c41186] at e_654_13
  have hr := e_654_13
  simp only [e_654_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4025)) := by
  have c41187 := (publicBatchWrapper86_copies1287 a h).2.2.2.1
  have c41188 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.1
  have c41189 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.1
  have c41190 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.1
  have c41191 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.1
  have c41192 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_14
  simp only [← c41187, ← c41188, ← c41189] at e_654_14
  have e_654_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_15
  simp only [← c41190, ← c41191, k2, ← c41192] at e_654_15
  have hr := e_654_15
  simp only [e_654_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4026)) := by
  have c41193 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.1
  have c41194 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41195 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41196 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41197 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41198 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_16
  simp only [← c41193, ← c41194, ← c41195] at e_654_16
  have e_654_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_17
  simp only [← c41196, ← c41197, k2, ← c41198] at e_654_17
  have hr := e_654_17
  simp only [e_654_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 654 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4027)) := by
  have c41199 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41200 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41201 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41202 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41203 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41204 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_654_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_18
  simp only [← c41199, ← c41200, ← c41201] at e_654_18
  have e_654_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_19
  simp only [← c41202, ← c41203, k2, ← c41204] at e_654_19
  have hr := e_654_19
  simp only [e_654_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4028)) := by
  have c41205 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41206 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41207 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41208 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41209 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41210 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_0
  simp only [← c41205, ← c41206, ← c41207] at e_655_0
  have e_655_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_1
  simp only [← c41208, ← c41209, k2, ← c41210] at e_655_1
  have hr := e_655_1
  simp only [e_655_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4029)) := by
  have c41211 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41212 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41213 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41214 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41215 := (publicBatchWrapper86_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41216 := (publicBatchWrapper86_copies1288 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_2
  simp only [← c41211, ← c41212, ← c41213] at e_655_2
  have e_655_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_3
  simp only [← c41214, ← c41215, k2, ← c41216] at e_655_3
  have hr := e_655_3
  simp only [e_655_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4030)) := by
  have c41217 := (publicBatchWrapper86_copies1288 a h).2.1
  have c41218 := (publicBatchWrapper86_copies1288 a h).2.2.1
  have c41219 := (publicBatchWrapper86_copies1288 a h).2.2.2.1
  have c41220 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.1
  have c41221 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.1
  have c41222 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_4
  simp only [← c41217, ← c41218, ← c41219] at e_655_4
  have e_655_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_5
  simp only [← c41220, ← c41221, k2, ← c41222] at e_655_5
  have hr := e_655_5
  simp only [e_655_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4031)) := by
  have c41223 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.1
  have c41224 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.1
  have c41225 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.1
  have c41226 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41227 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41228 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_6
  simp only [← c41223, ← c41224, ← c41225] at e_655_6
  have e_655_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_7
  simp only [← c41226, ← c41227, k2, ← c41228] at e_655_7
  have hr := e_655_7
  simp only [e_655_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4032)) := by
  have c41229 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41230 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41231 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41232 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41233 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41234 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_8
  simp only [← c41229, ← c41230, ← c41231] at e_655_8
  have e_655_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_9
  simp only [← c41232, ← c41233, k2, ← c41234] at e_655_9
  have hr := e_655_9
  simp only [e_655_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4033)) := by
  have c41235 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41236 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41237 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41238 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41239 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41240 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_10
  simp only [← c41235, ← c41236, ← c41237] at e_655_10
  have e_655_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_11
  simp only [← c41238, ← c41239, k2, ← c41240] at e_655_11
  have hr := e_655_11
  simp only [e_655_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4034)) := by
  have c41241 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41242 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41243 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41244 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41245 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41246 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_12
  simp only [← c41241, ← c41242, ← c41243] at e_655_12
  have e_655_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_13
  simp only [← c41244, ← c41245, k2, ← c41246] at e_655_13
  have hr := e_655_13
  simp only [e_655_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4035)) := by
  have c41247 := (publicBatchWrapper86_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41248 := (publicBatchWrapper86_copies1289 a h).1
  have c41249 := (publicBatchWrapper86_copies1289 a h).2.1
  have c41250 := (publicBatchWrapper86_copies1289 a h).2.2.1
  have c41251 := (publicBatchWrapper86_copies1289 a h).2.2.2.1
  have c41252 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_14
  simp only [← c41247, ← c41248, ← c41249] at e_655_14
  have e_655_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_15
  simp only [← c41250, ← c41251, k2, ← c41252] at e_655_15
  have hr := e_655_15
  simp only [e_655_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4036)) := by
  have c41253 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.1
  have c41254 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.1
  have c41255 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.1
  have c41256 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.1
  have c41257 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.1
  have c41258 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_16
  simp only [← c41253, ← c41254, ← c41255] at e_655_16
  have e_655_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_17
  simp only [← c41256, ← c41257, k2, ← c41258] at e_655_17
  have hr := e_655_17
  simp only [e_655_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 655 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4037)) := by
  have c41259 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41260 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41261 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41262 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41263 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41264 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_655_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_18
  simp only [← c41259, ← c41260, ← c41261] at e_655_18
  have e_655_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_19
  simp only [← c41262, ← c41263, k2, ← c41264] at e_655_19
  have hr := e_655_19
  simp only [e_655_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4038)) := by
  have c41265 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41266 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41267 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41268 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41269 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41270 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_0
  simp only [← c41265, ← c41266, ← c41267] at e_656_0
  have e_656_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_1
  simp only [← c41268, ← c41269, k2, ← c41270] at e_656_1
  have hr := e_656_1
  simp only [e_656_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4039)) := by
  have c41271 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41272 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41273 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41274 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41275 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41276 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_2
  simp only [← c41271, ← c41272, ← c41273] at e_656_2
  have e_656_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_3
  simp only [← c41274, ← c41275, k2, ← c41276] at e_656_3
  have hr := e_656_3
  simp only [e_656_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4040)) := by
  have c41277 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41278 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41279 := (publicBatchWrapper86_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41280 := (publicBatchWrapper86_copies1290 a h).1
  have c41281 := (publicBatchWrapper86_copies1290 a h).2.1
  have c41282 := (publicBatchWrapper86_copies1290 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_4
  simp only [← c41277, ← c41278, ← c41279] at e_656_4
  have e_656_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_5
  simp only [← c41280, ← c41281, k2, ← c41282] at e_656_5
  have hr := e_656_5
  simp only [e_656_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4041)) := by
  have c41283 := (publicBatchWrapper86_copies1290 a h).2.2.2.1
  have c41284 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.1
  have c41285 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.1
  have c41286 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.1
  have c41287 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.1
  have c41288 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_6
  simp only [← c41283, ← c41284, ← c41285] at e_656_6
  have e_656_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_7
  simp only [← c41286, ← c41287, k2, ← c41288] at e_656_7
  have hr := e_656_7
  simp only [e_656_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4042)) := by
  have c41289 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.1
  have c41290 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41291 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41292 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41293 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41294 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_8
  simp only [← c41289, ← c41290, ← c41291] at e_656_8
  have e_656_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_9
  simp only [← c41292, ← c41293, k2, ← c41294] at e_656_9
  have hr := e_656_9
  simp only [e_656_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4043)) := by
  have c41295 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41296 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41297 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41298 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41299 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41300 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_10
  simp only [← c41295, ← c41296, ← c41297] at e_656_10
  have e_656_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_11
  simp only [← c41298, ← c41299, k2, ← c41300] at e_656_11
  have hr := e_656_11
  simp only [e_656_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4044)) := by
  have c41301 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41302 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41303 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41304 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41305 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41306 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_12
  simp only [← c41301, ← c41302, ← c41303] at e_656_12
  have e_656_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_13
  simp only [← c41304, ← c41305, k2, ← c41306] at e_656_13
  have hr := e_656_13
  simp only [e_656_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4045)) := by
  have c41307 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41308 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41309 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41310 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41311 := (publicBatchWrapper86_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41312 := (publicBatchWrapper86_copies1291 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_14
  simp only [← c41307, ← c41308, ← c41309] at e_656_14
  have e_656_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_15
  simp only [← c41310, ← c41311, k2, ← c41312] at e_656_15
  have hr := e_656_15
  simp only [e_656_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4046)) := by
  have c41313 := (publicBatchWrapper86_copies1291 a h).2.1
  have c41314 := (publicBatchWrapper86_copies1291 a h).2.2.1
  have c41315 := (publicBatchWrapper86_copies1291 a h).2.2.2.1
  have c41316 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.1
  have c41317 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.1
  have c41318 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_16
  simp only [← c41313, ← c41314, ← c41315] at e_656_16
  have e_656_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_17
  simp only [← c41316, ← c41317, k2, ← c41318] at e_656_17
  have hr := e_656_17
  simp only [e_656_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 656 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4047)) := by
  have c41319 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.1
  have c41320 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.1
  have c41321 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.1
  have c41322 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41323 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41324 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_656_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_18
  simp only [← c41319, ← c41320, ← c41321] at e_656_18
  have e_656_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_19
  simp only [← c41322, ← c41323, k2, ← c41324] at e_656_19
  have hr := e_656_19
  simp only [e_656_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4048)) := by
  have c41325 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41326 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41327 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41328 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41329 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41330 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_0
  simp only [← c41325, ← c41326, ← c41327] at e_657_0
  have e_657_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_1
  simp only [← c41328, ← c41329, k2, ← c41330] at e_657_1
  have hr := e_657_1
  simp only [e_657_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4049)) := by
  have c41331 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41332 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41333 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41334 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41335 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41336 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_2
  simp only [← c41331, ← c41332, ← c41333] at e_657_2
  have e_657_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_3
  simp only [← c41334, ← c41335, k2, ← c41336] at e_657_3
  have hr := e_657_3
  simp only [e_657_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4050)) := by
  have c41337 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41338 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41339 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41340 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41341 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41342 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_4
  simp only [← c41337, ← c41338, ← c41339] at e_657_4
  have e_657_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_5
  simp only [← c41340, ← c41341, k2, ← c41342] at e_657_5
  have hr := e_657_5
  simp only [e_657_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4051)) := by
  have c41343 := (publicBatchWrapper86_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41344 := (publicBatchWrapper86_copies1292 a h).1
  have c41345 := (publicBatchWrapper86_copies1292 a h).2.1
  have c41346 := (publicBatchWrapper86_copies1292 a h).2.2.1
  have c41347 := (publicBatchWrapper86_copies1292 a h).2.2.2.1
  have c41348 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_6
  simp only [← c41343, ← c41344, ← c41345] at e_657_6
  have e_657_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_7
  simp only [← c41346, ← c41347, k2, ← c41348] at e_657_7
  have hr := e_657_7
  simp only [e_657_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4052)) := by
  have c41349 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.1
  have c41350 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.1
  have c41351 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.1
  have c41352 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.1
  have c41353 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.1
  have c41354 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_8
  simp only [← c41349, ← c41350, ← c41351] at e_657_8
  have e_657_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_9
  simp only [← c41352, ← c41353, k2, ← c41354] at e_657_9
  have hr := e_657_9
  simp only [e_657_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4053)) := by
  have c41355 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41356 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41357 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41358 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41359 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41360 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_10
  simp only [← c41355, ← c41356, ← c41357] at e_657_10
  have e_657_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_11
  simp only [← c41358, ← c41359, k2, ← c41360] at e_657_11
  have hr := e_657_11
  simp only [e_657_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4054)) := by
  have c41361 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41362 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41363 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41364 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41365 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41366 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_12
  simp only [← c41361, ← c41362, ← c41363] at e_657_12
  have e_657_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_13
  simp only [← c41364, ← c41365, k2, ← c41366] at e_657_13
  have hr := e_657_13
  simp only [e_657_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4055)) := by
  have c41367 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41368 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41369 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41370 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41371 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41372 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_14
  simp only [← c41367, ← c41368, ← c41369] at e_657_14
  have e_657_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_15
  simp only [← c41370, ← c41371, k2, ← c41372] at e_657_15
  have hr := e_657_15
  simp only [e_657_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4056)) := by
  have c41373 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41374 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41375 := (publicBatchWrapper86_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41376 := (publicBatchWrapper86_copies1293 a h).1
  have c41377 := (publicBatchWrapper86_copies1293 a h).2.1
  have c41378 := (publicBatchWrapper86_copies1293 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_16
  simp only [← c41373, ← c41374, ← c41375] at e_657_16
  have e_657_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_17
  simp only [← c41376, ← c41377, k2, ← c41378] at e_657_17
  have hr := e_657_17
  simp only [e_657_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 657 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4057)) := by
  have c41379 := (publicBatchWrapper86_copies1293 a h).2.2.2.1
  have c41380 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.1
  have c41381 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.1
  have c41382 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.1
  have c41383 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.1
  have c41384 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_657_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_18
  simp only [← c41379, ← c41380, ← c41381] at e_657_18
  have e_657_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_19
  simp only [← c41382, ← c41383, k2, ← c41384] at e_657_19
  have hr := e_657_19
  simp only [e_657_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4058)) := by
  have c41385 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.1
  have c41386 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41387 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41388 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41389 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41390 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_0
  simp only [← c41385, ← c41386, ← c41387] at e_658_0
  have e_658_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_1
  simp only [← c41388, ← c41389, k2, ← c41390] at e_658_1
  have hr := e_658_1
  simp only [e_658_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4059)) := by
  have c41391 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41392 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41393 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41394 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41395 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41396 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_2
  simp only [← c41391, ← c41392, ← c41393] at e_658_2
  have e_658_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_3
  simp only [← c41394, ← c41395, k2, ← c41396] at e_658_3
  have hr := e_658_3
  simp only [e_658_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4060)) := by
  have c41397 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41398 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41399 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41400 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41401 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41402 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_4
  simp only [← c41397, ← c41398, ← c41399] at e_658_4
  have e_658_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_5
  simp only [← c41400, ← c41401, k2, ← c41402] at e_658_5
  have hr := e_658_5
  simp only [e_658_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4061)) := by
  have c41403 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41404 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41405 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41406 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41407 := (publicBatchWrapper86_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41408 := (publicBatchWrapper86_copies1294 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_6
  simp only [← c41403, ← c41404, ← c41405] at e_658_6
  have e_658_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_7
  simp only [← c41406, ← c41407, k2, ← c41408] at e_658_7
  have hr := e_658_7
  simp only [e_658_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4062)) := by
  have c41409 := (publicBatchWrapper86_copies1294 a h).2.1
  have c41410 := (publicBatchWrapper86_copies1294 a h).2.2.1
  have c41411 := (publicBatchWrapper86_copies1294 a h).2.2.2.1
  have c41412 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.1
  have c41413 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.1
  have c41414 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_8
  simp only [← c41409, ← c41410, ← c41411] at e_658_8
  have e_658_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_9
  simp only [← c41412, ← c41413, k2, ← c41414] at e_658_9
  have hr := e_658_9
  simp only [e_658_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4063)) := by
  have c41415 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.1
  have c41416 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.1
  have c41417 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.1
  have c41418 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41419 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41420 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_10
  simp only [← c41415, ← c41416, ← c41417] at e_658_10
  have e_658_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_11
  simp only [← c41418, ← c41419, k2, ← c41420] at e_658_11
  have hr := e_658_11
  simp only [e_658_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4064)) := by
  have c41421 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41422 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41423 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41424 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41425 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41426 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_12
  simp only [← c41421, ← c41422, ← c41423] at e_658_12
  have e_658_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_13
  simp only [← c41424, ← c41425, k2, ← c41426] at e_658_13
  have hr := e_658_13
  simp only [e_658_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4065)) := by
  have c41427 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41428 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41429 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41430 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41431 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41432 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_14
  simp only [← c41427, ← c41428, ← c41429] at e_658_14
  have e_658_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_15
  simp only [← c41430, ← c41431, k2, ← c41432] at e_658_15
  have hr := e_658_15
  simp only [e_658_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4066)) := by
  have c41433 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41434 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41435 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41436 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41437 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41438 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_16
  simp only [← c41433, ← c41434, ← c41435] at e_658_16
  have e_658_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_17
  simp only [← c41436, ← c41437, k2, ← c41438] at e_658_17
  have hr := e_658_17
  simp only [e_658_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 658 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4067)) := by
  have c41439 := (publicBatchWrapper86_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41440 := (publicBatchWrapper86_copies1295 a h).1
  have c41441 := (publicBatchWrapper86_copies1295 a h).2.1
  have c41442 := (publicBatchWrapper86_copies1295 a h).2.2.1
  have c41443 := (publicBatchWrapper86_copies1295 a h).2.2.2.1
  have c41444 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_658_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_18
  simp only [← c41439, ← c41440, ← c41441] at e_658_18
  have e_658_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_19
  simp only [← c41442, ← c41443, k2, ← c41444] at e_658_19
  have hr := e_658_19
  simp only [e_658_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4068)) := by
  have c41445 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.1
  have c41446 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.1
  have c41447 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.1
  have c41448 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.1
  have c41449 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.1
  have c41450 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_0
  simp only [← c41445, ← c41446, ← c41447] at e_659_0
  have e_659_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_1
  simp only [← c41448, ← c41449, k2, ← c41450] at e_659_1
  have hr := e_659_1
  simp only [e_659_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4069)) := by
  have c41451 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41452 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41453 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41454 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41455 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41456 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_2
  simp only [← c41451, ← c41452, ← c41453] at e_659_2
  have e_659_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_3
  simp only [← c41454, ← c41455, k2, ← c41456] at e_659_3
  have hr := e_659_3
  simp only [e_659_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4070)) := by
  have c41457 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41458 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41459 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41460 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41461 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41462 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_4
  simp only [← c41457, ← c41458, ← c41459] at e_659_4
  have e_659_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_5
  simp only [← c41460, ← c41461, k2, ← c41462] at e_659_5
  have hr := e_659_5
  simp only [e_659_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4071)) := by
  have c41463 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41464 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41465 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41466 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41467 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41468 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_6
  simp only [← c41463, ← c41464, ← c41465] at e_659_6
  have e_659_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_7
  simp only [← c41466, ← c41467, k2, ← c41468] at e_659_7
  have hr := e_659_7
  simp only [e_659_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4072)) := by
  have c41469 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41470 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41471 := (publicBatchWrapper86_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41472 := (publicBatchWrapper86_copies1296 a h).1
  have c41473 := (publicBatchWrapper86_copies1296 a h).2.1
  have c41474 := (publicBatchWrapper86_copies1296 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_8
  simp only [← c41469, ← c41470, ← c41471] at e_659_8
  have e_659_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_9
  simp only [← c41472, ← c41473, k2, ← c41474] at e_659_9
  have hr := e_659_9
  simp only [e_659_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4073)) := by
  have c41475 := (publicBatchWrapper86_copies1296 a h).2.2.2.1
  have c41476 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.1
  have c41477 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.1
  have c41478 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.1
  have c41479 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.1
  have c41480 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_10
  simp only [← c41475, ← c41476, ← c41477] at e_659_10
  have e_659_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_11
  simp only [← c41478, ← c41479, k2, ← c41480] at e_659_11
  have hr := e_659_11
  simp only [e_659_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4074)) := by
  have c41481 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.1
  have c41482 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41483 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41484 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41485 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41486 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_12
  simp only [← c41481, ← c41482, ← c41483] at e_659_12
  have e_659_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_13
  simp only [← c41484, ← c41485, k2, ← c41486] at e_659_13
  have hr := e_659_13
  simp only [e_659_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4075)) := by
  have c41487 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41488 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41489 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41490 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41491 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41492 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_14
  simp only [← c41487, ← c41488, ← c41489] at e_659_14
  have e_659_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_15
  simp only [← c41490, ← c41491, k2, ← c41492] at e_659_15
  have hr := e_659_15
  simp only [e_659_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4076)) := by
  have c41493 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41494 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41495 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41496 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41497 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41498 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_16
  simp only [← c41493, ← c41494, ← c41495] at e_659_16
  have e_659_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_17
  simp only [← c41496, ← c41497, k2, ← c41498] at e_659_17
  have hr := e_659_17
  simp only [e_659_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 659 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4077)) := by
  have c41499 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41500 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41501 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41502 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41503 := (publicBatchWrapper86_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41504 := (publicBatchWrapper86_copies1297 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_659_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_18
  simp only [← c41499, ← c41500, ← c41501] at e_659_18
  have e_659_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_19
  simp only [← c41502, ← c41503, k2, ← c41504] at e_659_19
  have hr := e_659_19
  simp only [e_659_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4078)) := by
  have c41505 := (publicBatchWrapper86_copies1297 a h).2.1
  have c41506 := (publicBatchWrapper86_copies1297 a h).2.2.1
  have c41507 := (publicBatchWrapper86_copies1297 a h).2.2.2.1
  have c41508 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.1
  have c41509 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.1
  have c41510 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_0
  simp only [← c41505, ← c41506, ← c41507] at e_660_0
  have e_660_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_1
  simp only [← c41508, ← c41509, k2, ← c41510] at e_660_1
  have hr := e_660_1
  simp only [e_660_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4079)) := by
  have c41511 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.1
  have c41512 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.1
  have c41513 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.1
  have c41514 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41515 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41516 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_2
  simp only [← c41511, ← c41512, ← c41513] at e_660_2
  have e_660_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_3
  simp only [← c41514, ← c41515, k2, ← c41516] at e_660_3
  have hr := e_660_3
  simp only [e_660_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4080)) := by
  have c41517 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41518 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41519 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41520 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41521 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41522 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_4
  simp only [← c41517, ← c41518, ← c41519] at e_660_4
  have e_660_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_5
  simp only [← c41520, ← c41521, k2, ← c41522] at e_660_5
  have hr := e_660_5
  simp only [e_660_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4081)) := by
  have c41523 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41524 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41525 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41526 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41527 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41528 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_6
  simp only [← c41523, ← c41524, ← c41525] at e_660_6
  have e_660_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_7
  simp only [← c41526, ← c41527, k2, ← c41528] at e_660_7
  have hr := e_660_7
  simp only [e_660_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4082)) := by
  have c41529 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41530 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41531 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41532 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41533 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41534 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_8
  simp only [← c41529, ← c41530, ← c41531] at e_660_8
  have e_660_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_9
  simp only [← c41532, ← c41533, k2, ← c41534] at e_660_9
  have hr := e_660_9
  simp only [e_660_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4083)) := by
  have c41535 := (publicBatchWrapper86_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41536 := (publicBatchWrapper86_copies1298 a h).1
  have c41537 := (publicBatchWrapper86_copies1298 a h).2.1
  have c41538 := (publicBatchWrapper86_copies1298 a h).2.2.1
  have c41539 := (publicBatchWrapper86_copies1298 a h).2.2.2.1
  have c41540 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_10
  simp only [← c41535, ← c41536, ← c41537] at e_660_10
  have e_660_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_11
  simp only [← c41538, ← c41539, k2, ← c41540] at e_660_11
  have hr := e_660_11
  simp only [e_660_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4084)) := by
  have c41541 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.1
  have c41542 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.1
  have c41543 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.1
  have c41544 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.1
  have c41545 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.1
  have c41546 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_12
  simp only [← c41541, ← c41542, ← c41543] at e_660_12
  have e_660_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_13
  simp only [← c41544, ← c41545, k2, ← c41546] at e_660_13
  have hr := e_660_13
  simp only [e_660_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4085)) := by
  have c41547 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41548 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41549 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41550 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41551 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41552 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_14
  simp only [← c41547, ← c41548, ← c41549] at e_660_14
  have e_660_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_15
  simp only [← c41550, ← c41551, k2, ← c41552] at e_660_15
  have hr := e_660_15
  simp only [e_660_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4086)) := by
  have c41553 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41554 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41555 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41556 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41557 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41558 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_16
  simp only [← c41553, ← c41554, ← c41555] at e_660_16
  have e_660_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_17
  simp only [← c41556, ← c41557, k2, ← c41558] at e_660_17
  have hr := e_660_17
  simp only [e_660_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 660 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4087)) := by
  have c41559 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41560 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41561 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41562 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41563 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41564 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_660_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_18
  simp only [← c41559, ← c41560, ← c41561] at e_660_18
  have e_660_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_19
  simp only [← c41562, ← c41563, k2, ← c41564] at e_660_19
  have hr := e_660_19
  simp only [e_660_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4088)) := by
  have c41565 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41566 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41567 := (publicBatchWrapper86_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41568 := (publicBatchWrapper86_copies1299 a h).1
  have c41569 := (publicBatchWrapper86_copies1299 a h).2.1
  have c41570 := (publicBatchWrapper86_copies1299 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_0
  simp only [← c41565, ← c41566, ← c41567] at e_661_0
  have e_661_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_1
  simp only [← c41568, ← c41569, k2, ← c41570] at e_661_1
  have hr := e_661_1
  simp only [e_661_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4089)) := by
  have c41571 := (publicBatchWrapper86_copies1299 a h).2.2.2.1
  have c41572 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.1
  have c41573 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.1
  have c41574 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.1
  have c41575 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.1
  have c41576 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_2
  simp only [← c41571, ← c41572, ← c41573] at e_661_2
  have e_661_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_3
  simp only [← c41574, ← c41575, k2, ← c41576] at e_661_3
  have hr := e_661_3
  simp only [e_661_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4090)) := by
  have c41577 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.1
  have c41578 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41579 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41580 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41581 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41582 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_4
  simp only [← c41577, ← c41578, ← c41579] at e_661_4
  have e_661_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_5
  simp only [← c41580, ← c41581, k2, ← c41582] at e_661_5
  have hr := e_661_5
  simp only [e_661_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4091)) := by
  have c41583 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41584 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41585 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41586 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41587 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41588 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_6
  simp only [← c41583, ← c41584, ← c41585] at e_661_6
  have e_661_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_7
  simp only [← c41586, ← c41587, k2, ← c41588] at e_661_7
  have hr := e_661_7
  simp only [e_661_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4092)) := by
  have c41589 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41590 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41591 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41592 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41593 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41594 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_8
  simp only [← c41589, ← c41590, ← c41591] at e_661_8
  have e_661_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_9
  simp only [← c41592, ← c41593, k2, ← c41594] at e_661_9
  have hr := e_661_9
  simp only [e_661_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4093)) := by
  have c41595 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41596 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41597 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41598 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41599 := (publicBatchWrapper86_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41600 := (publicBatchWrapper86_copies1300 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_10
  simp only [← c41595, ← c41596, ← c41597] at e_661_10
  have e_661_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_11
  simp only [← c41598, ← c41599, k2, ← c41600] at e_661_11
  have hr := e_661_11
  simp only [e_661_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4094)) := by
  have c41601 := (publicBatchWrapper86_copies1300 a h).2.1
  have c41602 := (publicBatchWrapper86_copies1300 a h).2.2.1
  have c41603 := (publicBatchWrapper86_copies1300 a h).2.2.2.1
  have c41604 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.1
  have c41605 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.1
  have c41606 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_12
  simp only [← c41601, ← c41602, ← c41603] at e_661_12
  have e_661_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_13
  simp only [← c41604, ← c41605, k2, ← c41606] at e_661_13
  have hr := e_661_13
  simp only [e_661_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4095)) := by
  have c41607 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.1
  have c41608 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.1
  have c41609 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.1
  have c41610 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41611 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41612 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_14
  simp only [← c41607, ← c41608, ← c41609] at e_661_14
  have e_661_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_15
  simp only [← c41610, ← c41611, k2, ← c41612] at e_661_15
  have hr := e_661_15
  simp only [e_661_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4096)) := by
  have c41613 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41614 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41615 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41616 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41617 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41618 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_16
  simp only [← c41613, ← c41614, ← c41615] at e_661_16
  have e_661_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_17
  simp only [← c41616, ← c41617, k2, ← c41618] at e_661_17
  have hr := e_661_17
  simp only [e_661_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 661 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4097)) := by
  have c41619 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41620 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41621 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41622 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41623 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41624 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_661_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_18
  simp only [← c41619, ← c41620, ← c41621] at e_661_18
  have e_661_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_19
  simp only [← c41622, ← c41623, k2, ← c41624] at e_661_19
  have hr := e_661_19
  simp only [e_661_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4098)) := by
  have c41625 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41626 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41627 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41628 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41629 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41630 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_0
  simp only [← c41625, ← c41626, ← c41627] at e_662_0
  have e_662_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_1
  simp only [← c41628, ← c41629, k2, ← c41630] at e_662_1
  have hr := e_662_1
  simp only [e_662_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4099)) := by
  have c41631 := (publicBatchWrapper86_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41632 := (publicBatchWrapper86_copies1301 a h).1
  have c41633 := (publicBatchWrapper86_copies1301 a h).2.1
  have c41634 := (publicBatchWrapper86_copies1301 a h).2.2.1
  have c41635 := (publicBatchWrapper86_copies1301 a h).2.2.2.1
  have c41636 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_2
  simp only [← c41631, ← c41632, ← c41633] at e_662_2
  have e_662_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_3
  simp only [← c41634, ← c41635, k2, ← c41636] at e_662_3
  have hr := e_662_3
  simp only [e_662_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4100)) := by
  have c41637 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.1
  have c41638 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.1
  have c41639 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.1
  have c41640 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.1
  have c41641 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.1
  have c41642 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_4
  simp only [← c41637, ← c41638, ← c41639] at e_662_4
  have e_662_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_5
  simp only [← c41640, ← c41641, k2, ← c41642] at e_662_5
  have hr := e_662_5
  simp only [e_662_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4101)) := by
  have c41643 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41644 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41645 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41646 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41647 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41648 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_6
  simp only [← c41643, ← c41644, ← c41645] at e_662_6
  have e_662_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_7
  simp only [← c41646, ← c41647, k2, ← c41648] at e_662_7
  have hr := e_662_7
  simp only [e_662_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4102)) := by
  have c41649 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41650 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41651 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41652 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41653 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41654 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_8
  simp only [← c41649, ← c41650, ← c41651] at e_662_8
  have e_662_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_9
  simp only [← c41652, ← c41653, k2, ← c41654] at e_662_9
  have hr := e_662_9
  simp only [e_662_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4103)) := by
  have c41655 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41656 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41657 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41658 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41659 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41660 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_10
  simp only [← c41655, ← c41656, ← c41657] at e_662_10
  have e_662_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_11
  simp only [← c41658, ← c41659, k2, ← c41660] at e_662_11
  have hr := e_662_11
  simp only [e_662_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4104)) := by
  have c41661 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41662 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41663 := (publicBatchWrapper86_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41664 := (publicBatchWrapper86_copies1302 a h).1
  have c41665 := (publicBatchWrapper86_copies1302 a h).2.1
  have c41666 := (publicBatchWrapper86_copies1302 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_12
  simp only [← c41661, ← c41662, ← c41663] at e_662_12
  have e_662_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_13
  simp only [← c41664, ← c41665, k2, ← c41666] at e_662_13
  have hr := e_662_13
  simp only [e_662_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4105)) := by
  have c41667 := (publicBatchWrapper86_copies1302 a h).2.2.2.1
  have c41668 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.1
  have c41669 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.1
  have c41670 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.1
  have c41671 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.1
  have c41672 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_14
  simp only [← c41667, ← c41668, ← c41669] at e_662_14
  have e_662_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_15
  simp only [← c41670, ← c41671, k2, ← c41672] at e_662_15
  have hr := e_662_15
  simp only [e_662_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4106)) := by
  have c41673 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.1
  have c41674 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41675 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41676 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41677 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41678 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_16
  simp only [← c41673, ← c41674, ← c41675] at e_662_16
  have e_662_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_17
  simp only [← c41676, ← c41677, k2, ← c41678] at e_662_17
  have hr := e_662_17
  simp only [e_662_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 662 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4107)) := by
  have c41679 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41680 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41681 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41682 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41683 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41684 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_662_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_18
  simp only [← c41679, ← c41680, ← c41681] at e_662_18
  have e_662_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_19
  simp only [← c41682, ← c41683, k2, ← c41684] at e_662_19
  have hr := e_662_19
  simp only [e_662_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4108)) := by
  have c41685 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41686 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41687 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41688 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41689 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41690 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_0
  simp only [← c41685, ← c41686, ← c41687] at e_663_0
  have e_663_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_1
  simp only [← c41688, ← c41689, k2, ← c41690] at e_663_1
  have hr := e_663_1
  simp only [e_663_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4109)) := by
  have c41691 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41692 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41693 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41694 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41695 := (publicBatchWrapper86_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41696 := (publicBatchWrapper86_copies1303 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_2
  simp only [← c41691, ← c41692, ← c41693] at e_663_2
  have e_663_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_3
  simp only [← c41694, ← c41695, k2, ← c41696] at e_663_3
  have hr := e_663_3
  simp only [e_663_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4110)) := by
  have c41697 := (publicBatchWrapper86_copies1303 a h).2.1
  have c41698 := (publicBatchWrapper86_copies1303 a h).2.2.1
  have c41699 := (publicBatchWrapper86_copies1303 a h).2.2.2.1
  have c41700 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.1
  have c41701 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.1
  have c41702 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_4
  simp only [← c41697, ← c41698, ← c41699] at e_663_4
  have e_663_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_5
  simp only [← c41700, ← c41701, k2, ← c41702] at e_663_5
  have hr := e_663_5
  simp only [e_663_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4111)) := by
  have c41703 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.1
  have c41704 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.1
  have c41705 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.1
  have c41706 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41707 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41708 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_6
  simp only [← c41703, ← c41704, ← c41705] at e_663_6
  have e_663_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_7
  simp only [← c41706, ← c41707, k2, ← c41708] at e_663_7
  have hr := e_663_7
  simp only [e_663_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4112)) := by
  have c41709 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41710 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41711 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41712 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41713 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41714 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_8
  simp only [← c41709, ← c41710, ← c41711] at e_663_8
  have e_663_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_9
  simp only [← c41712, ← c41713, k2, ← c41714] at e_663_9
  have hr := e_663_9
  simp only [e_663_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4113)) := by
  have c41715 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41716 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41717 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41718 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41719 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41720 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_10
  simp only [← c41715, ← c41716, ← c41717] at e_663_10
  have e_663_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_11
  simp only [← c41718, ← c41719, k2, ← c41720] at e_663_11
  have hr := e_663_11
  simp only [e_663_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4114)) := by
  have c41721 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41722 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41723 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41724 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41725 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41726 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_12
  simp only [← c41721, ← c41722, ← c41723] at e_663_12
  have e_663_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_13
  simp only [← c41724, ← c41725, k2, ← c41726] at e_663_13
  have hr := e_663_13
  simp only [e_663_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4115)) := by
  have c41727 := (publicBatchWrapper86_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41728 := (publicBatchWrapper86_copies1304 a h).1
  have c41729 := (publicBatchWrapper86_copies1304 a h).2.1
  have c41730 := (publicBatchWrapper86_copies1304 a h).2.2.1
  have c41731 := (publicBatchWrapper86_copies1304 a h).2.2.2.1
  have c41732 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_14
  simp only [← c41727, ← c41728, ← c41729] at e_663_14
  have e_663_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_15
  simp only [← c41730, ← c41731, k2, ← c41732] at e_663_15
  have hr := e_663_15
  simp only [e_663_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4116)) := by
  have c41733 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.1
  have c41734 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.1
  have c41735 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.1
  have c41736 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.1
  have c41737 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.1
  have c41738 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_16
  simp only [← c41733, ← c41734, ← c41735] at e_663_16
  have e_663_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_17
  simp only [← c41736, ← c41737, k2, ← c41738] at e_663_17
  have hr := e_663_17
  simp only [e_663_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 663 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4117)) := by
  have c41739 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41740 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41741 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41742 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41743 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41744 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_663_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_18
  simp only [← c41739, ← c41740, ← c41741] at e_663_18
  have e_663_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_19
  simp only [← c41742, ← c41743, k2, ← c41744] at e_663_19
  have hr := e_663_19
  simp only [e_663_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4118)) := by
  have c41745 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41746 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41747 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41748 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41749 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41750 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_0
  simp only [← c41745, ← c41746, ← c41747] at e_664_0
  have e_664_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_1
  simp only [← c41748, ← c41749, k2, ← c41750] at e_664_1
  have hr := e_664_1
  simp only [e_664_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4119)) := by
  have c41751 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41752 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41753 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41754 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41755 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41756 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_2
  simp only [← c41751, ← c41752, ← c41753] at e_664_2
  have e_664_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_3
  simp only [← c41754, ← c41755, k2, ← c41756] at e_664_3
  have hr := e_664_3
  simp only [e_664_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4120)) := by
  have c41757 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41758 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41759 := (publicBatchWrapper86_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41760 := (publicBatchWrapper86_copies1305 a h).1
  have c41761 := (publicBatchWrapper86_copies1305 a h).2.1
  have c41762 := (publicBatchWrapper86_copies1305 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_4
  simp only [← c41757, ← c41758, ← c41759] at e_664_4
  have e_664_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_5
  simp only [← c41760, ← c41761, k2, ← c41762] at e_664_5
  have hr := e_664_5
  simp only [e_664_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4121)) := by
  have c41763 := (publicBatchWrapper86_copies1305 a h).2.2.2.1
  have c41764 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.1
  have c41765 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.1
  have c41766 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.1
  have c41767 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.1
  have c41768 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_6
  simp only [← c41763, ← c41764, ← c41765] at e_664_6
  have e_664_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_7
  simp only [← c41766, ← c41767, k2, ← c41768] at e_664_7
  have hr := e_664_7
  simp only [e_664_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4122)) := by
  have c41769 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.1
  have c41770 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41771 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41772 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41773 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41774 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_8
  simp only [← c41769, ← c41770, ← c41771] at e_664_8
  have e_664_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_9
  simp only [← c41772, ← c41773, k2, ← c41774] at e_664_9
  have hr := e_664_9
  simp only [e_664_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4123)) := by
  have c41775 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41776 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41777 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41778 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41779 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41780 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_10
  simp only [← c41775, ← c41776, ← c41777] at e_664_10
  have e_664_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_11
  simp only [← c41778, ← c41779, k2, ← c41780] at e_664_11
  have hr := e_664_11
  simp only [e_664_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4124)) := by
  have c41781 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41782 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41783 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41784 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41785 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41786 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_12
  simp only [← c41781, ← c41782, ← c41783] at e_664_12
  have e_664_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_13
  simp only [← c41784, ← c41785, k2, ← c41786] at e_664_13
  have hr := e_664_13
  simp only [e_664_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4125)) := by
  have c41787 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41788 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41789 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41790 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41791 := (publicBatchWrapper86_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41792 := (publicBatchWrapper86_copies1306 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_14
  simp only [← c41787, ← c41788, ← c41789] at e_664_14
  have e_664_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_15
  simp only [← c41790, ← c41791, k2, ← c41792] at e_664_15
  have hr := e_664_15
  simp only [e_664_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4126)) := by
  have c41793 := (publicBatchWrapper86_copies1306 a h).2.1
  have c41794 := (publicBatchWrapper86_copies1306 a h).2.2.1
  have c41795 := (publicBatchWrapper86_copies1306 a h).2.2.2.1
  have c41796 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.1
  have c41797 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.1
  have c41798 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_16
  simp only [← c41793, ← c41794, ← c41795] at e_664_16
  have e_664_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_17
  simp only [← c41796, ← c41797, k2, ← c41798] at e_664_17
  have hr := e_664_17
  simp only [e_664_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 664 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4127)) := by
  have c41799 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.1
  have c41800 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.1
  have c41801 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.1
  have c41802 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41803 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41804 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_664_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_18
  simp only [← c41799, ← c41800, ← c41801] at e_664_18
  have e_664_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_19
  simp only [← c41802, ← c41803, k2, ← c41804] at e_664_19
  have hr := e_664_19
  simp only [e_664_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4128)) := by
  have c41805 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41806 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41807 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41808 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41809 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41810 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_0
  simp only [← c41805, ← c41806, ← c41807] at e_665_0
  have e_665_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_1
  simp only [← c41808, ← c41809, k2, ← c41810] at e_665_1
  have hr := e_665_1
  simp only [e_665_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4129)) := by
  have c41811 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41812 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41813 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41814 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41815 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41816 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_2
  simp only [← c41811, ← c41812, ← c41813] at e_665_2
  have e_665_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_3
  simp only [← c41814, ← c41815, k2, ← c41816] at e_665_3
  have hr := e_665_3
  simp only [e_665_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4130)) := by
  have c41817 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41818 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41819 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41820 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41821 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41822 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_4
  simp only [← c41817, ← c41818, ← c41819] at e_665_4
  have e_665_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_5
  simp only [← c41820, ← c41821, k2, ← c41822] at e_665_5
  have hr := e_665_5
  simp only [e_665_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4131)) := by
  have c41823 := (publicBatchWrapper86_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41824 := (publicBatchWrapper86_copies1307 a h).1
  have c41825 := (publicBatchWrapper86_copies1307 a h).2.1
  have c41826 := (publicBatchWrapper86_copies1307 a h).2.2.1
  have c41827 := (publicBatchWrapper86_copies1307 a h).2.2.2.1
  have c41828 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_6
  simp only [← c41823, ← c41824, ← c41825] at e_665_6
  have e_665_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_7
  simp only [← c41826, ← c41827, k2, ← c41828] at e_665_7
  have hr := e_665_7
  simp only [e_665_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4132)) := by
  have c41829 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.1
  have c41830 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.1
  have c41831 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.1
  have c41832 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.1
  have c41833 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.1
  have c41834 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_8
  simp only [← c41829, ← c41830, ← c41831] at e_665_8
  have e_665_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_9
  simp only [← c41832, ← c41833, k2, ← c41834] at e_665_9
  have hr := e_665_9
  simp only [e_665_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4133)) := by
  have c41835 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41836 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41837 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41838 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41839 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41840 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_10
  simp only [← c41835, ← c41836, ← c41837] at e_665_10
  have e_665_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_11
  simp only [← c41838, ← c41839, k2, ← c41840] at e_665_11
  have hr := e_665_11
  simp only [e_665_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4134)) := by
  have c41841 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41842 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41843 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41844 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41845 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41846 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_12
  simp only [← c41841, ← c41842, ← c41843] at e_665_12
  have e_665_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_13
  simp only [← c41844, ← c41845, k2, ← c41846] at e_665_13
  have hr := e_665_13
  simp only [e_665_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4135)) := by
  have c41847 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41848 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41849 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41850 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41851 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41852 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_14
  simp only [← c41847, ← c41848, ← c41849] at e_665_14
  have e_665_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_15
  simp only [← c41850, ← c41851, k2, ← c41852] at e_665_15
  have hr := e_665_15
  simp only [e_665_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4136)) := by
  have c41853 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41854 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41855 := (publicBatchWrapper86_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41856 := (publicBatchWrapper86_copies1308 a h).1
  have c41857 := (publicBatchWrapper86_copies1308 a h).2.1
  have c41858 := (publicBatchWrapper86_copies1308 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_16
  simp only [← c41853, ← c41854, ← c41855] at e_665_16
  have e_665_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_17
  simp only [← c41856, ← c41857, k2, ← c41858] at e_665_17
  have hr := e_665_17
  simp only [e_665_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 665 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4137)) := by
  have c41859 := (publicBatchWrapper86_copies1308 a h).2.2.2.1
  have c41860 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.1
  have c41861 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.1
  have c41862 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.1
  have c41863 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.1
  have c41864 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_665_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_18
  simp only [← c41859, ← c41860, ← c41861] at e_665_18
  have e_665_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_19
  simp only [← c41862, ← c41863, k2, ← c41864] at e_665_19
  have hr := e_665_19
  simp only [e_665_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4138)) := by
  have c41865 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.1
  have c41866 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41867 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41868 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41869 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41870 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_0
  simp only [← c41865, ← c41866, ← c41867] at e_666_0
  have e_666_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_1
  simp only [← c41868, ← c41869, k2, ← c41870] at e_666_1
  have hr := e_666_1
  simp only [e_666_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4139)) := by
  have c41871 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41872 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41873 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41874 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41875 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41876 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_2
  simp only [← c41871, ← c41872, ← c41873] at e_666_2
  have e_666_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_3
  simp only [← c41874, ← c41875, k2, ← c41876] at e_666_3
  have hr := e_666_3
  simp only [e_666_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4140)) := by
  have c41877 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41878 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41879 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41880 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41881 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41882 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_4
  simp only [← c41877, ← c41878, ← c41879] at e_666_4
  have e_666_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_5
  simp only [← c41880, ← c41881, k2, ← c41882] at e_666_5
  have hr := e_666_5
  simp only [e_666_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4141)) := by
  have c41883 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41884 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41885 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41886 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41887 := (publicBatchWrapper86_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41888 := (publicBatchWrapper86_copies1309 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_6
  simp only [← c41883, ← c41884, ← c41885] at e_666_6
  have e_666_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_7
  simp only [← c41886, ← c41887, k2, ← c41888] at e_666_7
  have hr := e_666_7
  simp only [e_666_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4142)) := by
  have c41889 := (publicBatchWrapper86_copies1309 a h).2.1
  have c41890 := (publicBatchWrapper86_copies1309 a h).2.2.1
  have c41891 := (publicBatchWrapper86_copies1309 a h).2.2.2.1
  have c41892 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.1
  have c41893 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.1
  have c41894 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_8
  simp only [← c41889, ← c41890, ← c41891] at e_666_8
  have e_666_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_9
  simp only [← c41892, ← c41893, k2, ← c41894] at e_666_9
  have hr := e_666_9
  simp only [e_666_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4143)) := by
  have c41895 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.1
  have c41896 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.1
  have c41897 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.1
  have c41898 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41899 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41900 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_10
  simp only [← c41895, ← c41896, ← c41897] at e_666_10
  have e_666_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_11
  simp only [← c41898, ← c41899, k2, ← c41900] at e_666_11
  have hr := e_666_11
  simp only [e_666_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4144)) := by
  have c41901 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41902 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41903 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41904 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41905 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41906 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_12
  simp only [← c41901, ← c41902, ← c41903] at e_666_12
  have e_666_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_13
  simp only [← c41904, ← c41905, k2, ← c41906] at e_666_13
  have hr := e_666_13
  simp only [e_666_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4145)) := by
  have c41907 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41908 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41909 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41910 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41911 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41912 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_14
  simp only [← c41907, ← c41908, ← c41909] at e_666_14
  have e_666_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_15
  simp only [← c41910, ← c41911, k2, ← c41912] at e_666_15
  have hr := e_666_15
  simp only [e_666_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4146)) := by
  have c41913 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41914 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41915 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41916 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41917 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41918 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_16
  simp only [← c41913, ← c41914, ← c41915] at e_666_16
  have e_666_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_17
  simp only [← c41916, ← c41917, k2, ← c41918] at e_666_17
  have hr := e_666_17
  simp only [e_666_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 666 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4147)) := by
  have c41919 := (publicBatchWrapper86_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41920 := (publicBatchWrapper86_copies1310 a h).1
  have c41921 := (publicBatchWrapper86_copies1310 a h).2.1
  have c41922 := (publicBatchWrapper86_copies1310 a h).2.2.1
  have c41923 := (publicBatchWrapper86_copies1310 a h).2.2.2.1
  have c41924 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_666_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_18
  simp only [← c41919, ← c41920, ← c41921] at e_666_18
  have e_666_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_19
  simp only [← c41922, ← c41923, k2, ← c41924] at e_666_19
  have hr := e_666_19
  simp only [e_666_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4148)) := by
  have c41925 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.1
  have c41926 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.1
  have c41927 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.1
  have c41928 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.1
  have c41929 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.1
  have c41930 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_0
  simp only [← c41925, ← c41926, ← c41927] at e_667_0
  have e_667_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_1
  simp only [← c41928, ← c41929, k2, ← c41930] at e_667_1
  have hr := e_667_1
  simp only [e_667_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4149)) := by
  have c41931 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41932 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41933 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41934 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41935 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41936 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_2
  simp only [← c41931, ← c41932, ← c41933] at e_667_2
  have e_667_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_3
  simp only [← c41934, ← c41935, k2, ← c41936] at e_667_3
  have hr := e_667_3
  simp only [e_667_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4150)) := by
  have c41937 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41938 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41939 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41940 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41941 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41942 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_4
  simp only [← c41937, ← c41938, ← c41939] at e_667_4
  have e_667_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_5
  simp only [← c41940, ← c41941, k2, ← c41942] at e_667_5
  have hr := e_667_5
  simp only [e_667_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4151)) := by
  have c41943 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41944 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41945 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41946 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41947 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41948 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_6
  simp only [← c41943, ← c41944, ← c41945] at e_667_6
  have e_667_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_7
  simp only [← c41946, ← c41947, k2, ← c41948] at e_667_7
  have hr := e_667_7
  simp only [e_667_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4152)) := by
  have c41949 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41950 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41951 := (publicBatchWrapper86_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41952 := (publicBatchWrapper86_copies1311 a h).1
  have c41953 := (publicBatchWrapper86_copies1311 a h).2.1
  have c41954 := (publicBatchWrapper86_copies1311 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_8
  simp only [← c41949, ← c41950, ← c41951] at e_667_8
  have e_667_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_9
  simp only [← c41952, ← c41953, k2, ← c41954] at e_667_9
  have hr := e_667_9
  simp only [e_667_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4153)) := by
  have c41955 := (publicBatchWrapper86_copies1311 a h).2.2.2.1
  have c41956 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.1
  have c41957 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.1
  have c41958 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.1
  have c41959 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.1
  have c41960 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_10
  simp only [← c41955, ← c41956, ← c41957] at e_667_10
  have e_667_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_11
  simp only [← c41958, ← c41959, k2, ← c41960] at e_667_11
  have hr := e_667_11
  simp only [e_667_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4154)) := by
  have c41961 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.1
  have c41962 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41963 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41964 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41965 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41966 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_12
  simp only [← c41961, ← c41962, ← c41963] at e_667_12
  have e_667_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_13
  simp only [← c41964, ← c41965, k2, ← c41966] at e_667_13
  have hr := e_667_13
  simp only [e_667_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4155)) := by
  have c41967 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41968 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41969 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41970 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41971 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41972 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_14
  simp only [← c41967, ← c41968, ← c41969] at e_667_14
  have e_667_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_15
  simp only [← c41970, ← c41971, k2, ← c41972] at e_667_15
  have hr := e_667_15
  simp only [e_667_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4156)) := by
  have c41973 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41974 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41975 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41976 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41977 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41978 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_16
  simp only [← c41973, ← c41974, ← c41975] at e_667_16
  have e_667_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_17
  simp only [← c41976, ← c41977, k2, ← c41978] at e_667_17
  have hr := e_667_17
  simp only [e_667_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 667 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4157)) := by
  have c41979 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41980 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41981 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41982 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41983 := (publicBatchWrapper86_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41984 := (publicBatchWrapper86_copies1312 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_667_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_18
  simp only [← c41979, ← c41980, ← c41981] at e_667_18
  have e_667_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_19
  simp only [← c41982, ← c41983, k2, ← c41984] at e_667_19
  have hr := e_667_19
  simp only [e_667_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4158)) := by
  have c41985 := (publicBatchWrapper86_copies1312 a h).2.1
  have c41986 := (publicBatchWrapper86_copies1312 a h).2.2.1
  have c41987 := (publicBatchWrapper86_copies1312 a h).2.2.2.1
  have c41988 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.1
  have c41989 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.1
  have c41990 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_0
  simp only [← c41985, ← c41986, ← c41987] at e_668_0
  have e_668_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_1
  simp only [← c41988, ← c41989, k2, ← c41990] at e_668_1
  have hr := e_668_1
  simp only [e_668_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4159)) := by
  have c41991 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.1
  have c41992 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.1
  have c41993 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.1
  have c41994 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41995 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41996 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_2
  simp only [← c41991, ← c41992, ← c41993] at e_668_2
  have e_668_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_3
  simp only [← c41994, ← c41995, k2, ← c41996] at e_668_3
  have hr := e_668_3
  simp only [e_668_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 23) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4160)) := by
  have c41997 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41998 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41999 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42000 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42001 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42002 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_4
  simp only [← c41997, ← c41998, ← c41999] at e_668_4
  have e_668_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_5
  simp only [← c42000, ← c42001, k2, ← c42002] at e_668_5
  have hr := e_668_5
  simp only [e_668_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 31) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4161)) := by
  have c42003 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42004 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42005 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42006 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42007 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42008 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_6
  simp only [← c42003, ← c42004, ← c42005] at e_668_6
  have e_668_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_7
  simp only [← c42006, ← c42007, k2, ← c42008] at e_668_7
  have hr := e_668_7
  simp only [e_668_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 39) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4162)) := by
  have c42009 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42010 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42011 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42012 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42013 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42014 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_8
  simp only [← c42009, ← c42010, ← c42011] at e_668_8
  have e_668_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_9
  simp only [← c42012, ← c42013, k2, ← c42014] at e_668_9
  have hr := e_668_9
  simp only [e_668_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 47) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4163)) := by
  have c42015 := (publicBatchWrapper86_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42016 := (publicBatchWrapper86_copies1313 a h).1
  have c42017 := (publicBatchWrapper86_copies1313 a h).2.1
  have c42018 := (publicBatchWrapper86_copies1313 a h).2.2.1
  have c42019 := (publicBatchWrapper86_copies1313 a h).2.2.2.1
  have c42020 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_10
  simp only [← c42015, ← c42016, ← c42017] at e_668_10
  have e_668_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_11
  simp only [← c42018, ← c42019, k2, ← c42020] at e_668_11
  have hr := e_668_11
  simp only [e_668_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 55) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4164)) := by
  have c42021 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.1
  have c42022 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.1
  have c42023 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.1
  have c42024 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.1
  have c42025 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.1
  have c42026 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_12
  simp only [← c42021, ← c42022, ← c42023] at e_668_12
  have e_668_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_13
  simp only [← c42024, ← c42025, k2, ← c42026] at e_668_13
  have hr := e_668_13
  simp only [e_668_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 63) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4165)) := by
  have c42027 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42028 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42029 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42030 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42031 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42032 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_14
  simp only [← c42027, ← c42028, ← c42029] at e_668_14
  have e_668_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_15
  simp only [← c42030, ← c42031, k2, ← c42032] at e_668_15
  have hr := e_668_15
  simp only [e_668_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 71) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4166)) := by
  have c42033 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42034 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42035 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42036 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42037 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42038 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_16
  simp only [← c42033, ← c42034, ← c42035] at e_668_16
  have e_668_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_17
  simp only [← c42036, ← c42037, k2, ← c42038] at e_668_17
  have hr := e_668_17
  simp only [e_668_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 668 79) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4167)) := by
  have c42039 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42040 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42041 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42042 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42043 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42044 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_668_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_18
  simp only [← c42039, ← c42040, ← c42041] at e_668_18
  have e_668_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_19
  simp only [← c42042, ← c42043, k2, ← c42044] at e_668_19
  have hr := e_668_19
  simp only [e_668_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 7) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4168)) := by
  have c42045 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42046 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42047 := (publicBatchWrapper86_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42048 := (publicBatchWrapper86_copies1314 a h).1
  have c42049 := (publicBatchWrapper86_copies1314 a h).2.1
  have c42050 := (publicBatchWrapper86_copies1314 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_0
  simp only [← c42045, ← c42046, ← c42047] at e_669_0
  have e_669_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_1
  simp only [← c42048, ← c42049, k2, ← c42050] at e_669_1
  have hr := e_669_1
  simp only [e_669_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 15) = bselect (a (.wire 22 27)) (a (.virt 200)) (a (.virt 4169)) := by
  have c42051 := (publicBatchWrapper86_copies1314 a h).2.2.2.1
  have c42052 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.1
  have c42053 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.1
  have c42054 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.1
  have c42055 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.1
  have c42056 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_2
  simp only [← c42051, ← c42052, ← c42053] at e_669_2
  have e_669_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_3
  simp only [← c42054, ← c42055, k2, ← c42056] at e_669_3
  have hr := e_669_3
  simp only [e_669_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4210)) := by
  have c42057 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.1
  have c42058 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42059 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42060 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42061 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42062 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_4
  simp only [← c42057, ← c42058, ← c42059] at e_669_4
  have e_669_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_5
  simp only [← c42060, ← c42061, k2, ← c42062] at e_669_5
  have hr := e_669_5
  simp only [e_669_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4211)) := by
  have c42063 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42064 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42065 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42066 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42067 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42068 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_6
  simp only [← c42063, ← c42064, ← c42065] at e_669_6
  have e_669_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_7
  simp only [← c42066, ← c42067, k2, ← c42068] at e_669_7
  have hr := e_669_7
  simp only [e_669_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4212)) := by
  have c42069 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42070 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42071 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42072 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42073 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42074 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_8
  simp only [← c42069, ← c42070, ← c42071] at e_669_8
  have e_669_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_9
  simp only [← c42072, ← c42073, k2, ← c42074] at e_669_9
  have hr := e_669_9
  simp only [e_669_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4213)) := by
  have c42075 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42076 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42077 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42078 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42079 := (publicBatchWrapper86_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42080 := (publicBatchWrapper86_copies1315 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_10
  simp only [← c42075, ← c42076, ← c42077] at e_669_10
  have e_669_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_11
  simp only [← c42078, ← c42079, k2, ← c42080] at e_669_11
  have hr := e_669_11
  simp only [e_669_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4214)) := by
  have c42081 := (publicBatchWrapper86_copies1315 a h).2.1
  have c42082 := (publicBatchWrapper86_copies1315 a h).2.2.1
  have c42083 := (publicBatchWrapper86_copies1315 a h).2.2.2.1
  have c42084 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.1
  have c42085 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.1
  have c42086 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_12
  simp only [← c42081, ← c42082, ← c42083] at e_669_12
  have e_669_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_13
  simp only [← c42084, ← c42085, k2, ← c42086] at e_669_13
  have hr := e_669_13
  simp only [e_669_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4215)) := by
  have c42087 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.1
  have c42088 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.1
  have c42089 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.1
  have c42090 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42091 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42092 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_14
  simp only [← c42087, ← c42088, ← c42089] at e_669_14
  have e_669_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_15
  simp only [← c42090, ← c42091, k2, ← c42092] at e_669_15
  have hr := e_669_15
  simp only [e_669_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4216)) := by
  have c42093 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42094 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42095 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42096 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42097 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42098 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_16
  simp only [← c42093, ← c42094, ← c42095] at e_669_16
  have e_669_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_17
  simp only [← c42096, ← c42097, k2, ← c42098] at e_669_17
  have hr := e_669_17
  simp only [e_669_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 669 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4217)) := by
  have c42099 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42100 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42101 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42102 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42103 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42104 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_669_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_18
  simp only [← c42099, ← c42100, ← c42101] at e_669_18
  have e_669_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_19
  simp only [← c42102, ← c42103, k2, ← c42104] at e_669_19
  have hr := e_669_19
  simp only [e_669_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4218)) := by
  have c42105 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42106 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42107 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42108 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42109 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42110 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_0
  simp only [← c42105, ← c42106, ← c42107] at e_670_0
  have e_670_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_1
  simp only [← c42108, ← c42109, k2, ← c42110] at e_670_1
  have hr := e_670_1
  simp only [e_670_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4219)) := by
  have c42111 := (publicBatchWrapper86_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42112 := (publicBatchWrapper86_copies1316 a h).1
  have c42113 := (publicBatchWrapper86_copies1316 a h).2.1
  have c42114 := (publicBatchWrapper86_copies1316 a h).2.2.1
  have c42115 := (publicBatchWrapper86_copies1316 a h).2.2.2.1
  have c42116 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_2
  simp only [← c42111, ← c42112, ← c42113] at e_670_2
  have e_670_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_3
  simp only [← c42114, ← c42115, k2, ← c42116] at e_670_3
  have hr := e_670_3
  simp only [e_670_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4220)) := by
  have c42117 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.1
  have c42118 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.1
  have c42119 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.1
  have c42120 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.1
  have c42121 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.1
  have c42122 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_4
  simp only [← c42117, ← c42118, ← c42119] at e_670_4
  have e_670_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_5
  simp only [← c42120, ← c42121, k2, ← c42122] at e_670_5
  have hr := e_670_5
  simp only [e_670_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4221)) := by
  have c42123 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42124 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42125 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42126 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42127 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42128 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_6
  simp only [← c42123, ← c42124, ← c42125] at e_670_6
  have e_670_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_7
  simp only [← c42126, ← c42127, k2, ← c42128] at e_670_7
  have hr := e_670_7
  simp only [e_670_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4222)) := by
  have c42129 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42130 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42131 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42132 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42133 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42134 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_8
  simp only [← c42129, ← c42130, ← c42131] at e_670_8
  have e_670_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_9
  simp only [← c42132, ← c42133, k2, ← c42134] at e_670_9
  have hr := e_670_9
  simp only [e_670_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4223)) := by
  have c42135 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42136 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42137 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42138 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42139 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42140 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_10
  simp only [← c42135, ← c42136, ← c42137] at e_670_10
  have e_670_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_11
  simp only [← c42138, ← c42139, k2, ← c42140] at e_670_11
  have hr := e_670_11
  simp only [e_670_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4224)) := by
  have c42141 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42142 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42143 := (publicBatchWrapper86_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42144 := (publicBatchWrapper86_copies1317 a h).1
  have c42145 := (publicBatchWrapper86_copies1317 a h).2.1
  have c42146 := (publicBatchWrapper86_copies1317 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_12
  simp only [← c42141, ← c42142, ← c42143] at e_670_12
  have e_670_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_13
  simp only [← c42144, ← c42145, k2, ← c42146] at e_670_13
  have hr := e_670_13
  simp only [e_670_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4225)) := by
  have c42147 := (publicBatchWrapper86_copies1317 a h).2.2.2.1
  have c42148 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.1
  have c42149 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.1
  have c42150 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.1
  have c42151 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.1
  have c42152 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_14
  simp only [← c42147, ← c42148, ← c42149] at e_670_14
  have e_670_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_15
  simp only [← c42150, ← c42151, k2, ← c42152] at e_670_15
  have hr := e_670_15
  simp only [e_670_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4226)) := by
  have c42153 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.1
  have c42154 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42155 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42156 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42157 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42158 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_16
  simp only [← c42153, ← c42154, ← c42155] at e_670_16
  have e_670_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_17
  simp only [← c42156, ← c42157, k2, ← c42158] at e_670_17
  have hr := e_670_17
  simp only [e_670_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 670 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4227)) := by
  have c42159 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42160 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42161 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42162 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42163 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42164 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_670_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_18
  simp only [← c42159, ← c42160, ← c42161] at e_670_18
  have e_670_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_19
  simp only [← c42162, ← c42163, k2, ← c42164] at e_670_19
  have hr := e_670_19
  simp only [e_670_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4228)) := by
  have c42165 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42166 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42167 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42168 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42169 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42170 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_0
  simp only [← c42165, ← c42166, ← c42167] at e_671_0
  have e_671_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_1
  simp only [← c42168, ← c42169, k2, ← c42170] at e_671_1
  have hr := e_671_1
  simp only [e_671_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4229)) := by
  have c42171 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42172 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42173 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42174 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42175 := (publicBatchWrapper86_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42176 := (publicBatchWrapper86_copies1318 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_2
  simp only [← c42171, ← c42172, ← c42173] at e_671_2
  have e_671_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_3
  simp only [← c42174, ← c42175, k2, ← c42176] at e_671_3
  have hr := e_671_3
  simp only [e_671_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4230)) := by
  have c42177 := (publicBatchWrapper86_copies1318 a h).2.1
  have c42178 := (publicBatchWrapper86_copies1318 a h).2.2.1
  have c42179 := (publicBatchWrapper86_copies1318 a h).2.2.2.1
  have c42180 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.1
  have c42181 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.1
  have c42182 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_4
  simp only [← c42177, ← c42178, ← c42179] at e_671_4
  have e_671_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_5
  simp only [← c42180, ← c42181, k2, ← c42182] at e_671_5
  have hr := e_671_5
  simp only [e_671_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4231)) := by
  have c42183 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.1
  have c42184 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.1
  have c42185 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.1
  have c42186 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42187 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42188 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_6
  simp only [← c42183, ← c42184, ← c42185] at e_671_6
  have e_671_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_7
  simp only [← c42186, ← c42187, k2, ← c42188] at e_671_7
  have hr := e_671_7
  simp only [e_671_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4232)) := by
  have c42189 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42190 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42191 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42192 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42193 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42194 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_8
  simp only [← c42189, ← c42190, ← c42191] at e_671_8
  have e_671_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_9
  simp only [← c42192, ← c42193, k2, ← c42194] at e_671_9
  have hr := e_671_9
  simp only [e_671_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4233)) := by
  have c42195 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42196 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42197 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42198 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42199 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42200 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_10
  simp only [← c42195, ← c42196, ← c42197] at e_671_10
  have e_671_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_11
  simp only [← c42198, ← c42199, k2, ← c42200] at e_671_11
  have hr := e_671_11
  simp only [e_671_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4234)) := by
  have c42201 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42202 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42203 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42204 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42205 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42206 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_12
  simp only [← c42201, ← c42202, ← c42203] at e_671_12
  have e_671_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_13
  simp only [← c42204, ← c42205, k2, ← c42206] at e_671_13
  have hr := e_671_13
  simp only [e_671_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4235)) := by
  have c42207 := (publicBatchWrapper86_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42208 := (publicBatchWrapper86_copies1319 a h).1
  have c42209 := (publicBatchWrapper86_copies1319 a h).2.1
  have c42210 := (publicBatchWrapper86_copies1319 a h).2.2.1
  have c42211 := (publicBatchWrapper86_copies1319 a h).2.2.2.1
  have c42212 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_14
  simp only [← c42207, ← c42208, ← c42209] at e_671_14
  have e_671_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_15
  simp only [← c42210, ← c42211, k2, ← c42212] at e_671_15
  have hr := e_671_15
  simp only [e_671_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4236)) := by
  have c42213 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.1
  have c42214 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.1
  have c42215 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.1
  have c42216 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.1
  have c42217 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.1
  have c42218 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_16
  simp only [← c42213, ← c42214, ← c42215] at e_671_16
  have e_671_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_17
  simp only [← c42216, ← c42217, k2, ← c42218] at e_671_17
  have hr := e_671_17
  simp only [e_671_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 671 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4237)) := by
  have c42219 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42220 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42221 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42222 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42223 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42224 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_671_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_18
  simp only [← c42219, ← c42220, ← c42221] at e_671_18
  have e_671_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_19
  simp only [← c42222, ← c42223, k2, ← c42224] at e_671_19
  have hr := e_671_19
  simp only [e_671_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4238)) := by
  have c42225 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42226 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42227 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42228 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42229 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42230 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_0
  simp only [← c42225, ← c42226, ← c42227] at e_672_0
  have e_672_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_1
  simp only [← c42228, ← c42229, k2, ← c42230] at e_672_1
  have hr := e_672_1
  simp only [e_672_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4239)) := by
  have c42231 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42232 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42233 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42234 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42235 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42236 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_2
  simp only [← c42231, ← c42232, ← c42233] at e_672_2
  have e_672_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_3
  simp only [← c42234, ← c42235, k2, ← c42236] at e_672_3
  have hr := e_672_3
  simp only [e_672_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4240)) := by
  have c42237 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42238 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42239 := (publicBatchWrapper86_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42240 := (publicBatchWrapper86_copies1320 a h).1
  have c42241 := (publicBatchWrapper86_copies1320 a h).2.1
  have c42242 := (publicBatchWrapper86_copies1320 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_4
  simp only [← c42237, ← c42238, ← c42239] at e_672_4
  have e_672_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_5
  simp only [← c42240, ← c42241, k2, ← c42242] at e_672_5
  have hr := e_672_5
  simp only [e_672_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4241)) := by
  have c42243 := (publicBatchWrapper86_copies1320 a h).2.2.2.1
  have c42244 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.1
  have c42245 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.1
  have c42246 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.1
  have c42247 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.1
  have c42248 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_6
  simp only [← c42243, ← c42244, ← c42245] at e_672_6
  have e_672_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_7
  simp only [← c42246, ← c42247, k2, ← c42248] at e_672_7
  have hr := e_672_7
  simp only [e_672_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4242)) := by
  have c42249 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.1
  have c42250 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42251 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42252 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42253 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42254 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_8
  simp only [← c42249, ← c42250, ← c42251] at e_672_8
  have e_672_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_9
  simp only [← c42252, ← c42253, k2, ← c42254] at e_672_9
  have hr := e_672_9
  simp only [e_672_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4243)) := by
  have c42255 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42256 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42257 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42258 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42259 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42260 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_10
  simp only [← c42255, ← c42256, ← c42257] at e_672_10
  have e_672_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_11
  simp only [← c42258, ← c42259, k2, ← c42260] at e_672_11
  have hr := e_672_11
  simp only [e_672_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4244)) := by
  have c42261 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42262 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42263 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42264 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42265 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42266 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_12
  simp only [← c42261, ← c42262, ← c42263] at e_672_12
  have e_672_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_13
  simp only [← c42264, ← c42265, k2, ← c42266] at e_672_13
  have hr := e_672_13
  simp only [e_672_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4245)) := by
  have c42267 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42268 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42269 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42270 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42271 := (publicBatchWrapper86_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42272 := (publicBatchWrapper86_copies1321 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_14
  simp only [← c42267, ← c42268, ← c42269] at e_672_14
  have e_672_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_15
  simp only [← c42270, ← c42271, k2, ← c42272] at e_672_15
  have hr := e_672_15
  simp only [e_672_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4246)) := by
  have c42273 := (publicBatchWrapper86_copies1321 a h).2.1
  have c42274 := (publicBatchWrapper86_copies1321 a h).2.2.1
  have c42275 := (publicBatchWrapper86_copies1321 a h).2.2.2.1
  have c42276 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.1
  have c42277 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.1
  have c42278 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_16
  simp only [← c42273, ← c42274, ← c42275] at e_672_16
  have e_672_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_17
  simp only [← c42276, ← c42277, k2, ← c42278] at e_672_17
  have hr := e_672_17
  simp only [e_672_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 672 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4247)) := by
  have c42279 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.1
  have c42280 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.1
  have c42281 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.1
  have c42282 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42283 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42284 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_672_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_18
  simp only [← c42279, ← c42280, ← c42281] at e_672_18
  have e_672_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_19
  simp only [← c42282, ← c42283, k2, ← c42284] at e_672_19
  have hr := e_672_19
  simp only [e_672_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4248)) := by
  have c42285 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42286 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42287 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42288 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42289 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42290 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_0
  simp only [← c42285, ← c42286, ← c42287] at e_673_0
  have e_673_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_1
  simp only [← c42288, ← c42289, k2, ← c42290] at e_673_1
  have hr := e_673_1
  simp only [e_673_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4249)) := by
  have c42291 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42292 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42293 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42294 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42295 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42296 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_2
  simp only [← c42291, ← c42292, ← c42293] at e_673_2
  have e_673_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_3
  simp only [← c42294, ← c42295, k2, ← c42296] at e_673_3
  have hr := e_673_3
  simp only [e_673_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4250)) := by
  have c42297 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42298 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42299 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42300 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42301 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42302 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_4
  simp only [← c42297, ← c42298, ← c42299] at e_673_4
  have e_673_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_5
  simp only [← c42300, ← c42301, k2, ← c42302] at e_673_5
  have hr := e_673_5
  simp only [e_673_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4251)) := by
  have c42303 := (publicBatchWrapper86_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42304 := (publicBatchWrapper86_copies1322 a h).1
  have c42305 := (publicBatchWrapper86_copies1322 a h).2.1
  have c42306 := (publicBatchWrapper86_copies1322 a h).2.2.1
  have c42307 := (publicBatchWrapper86_copies1322 a h).2.2.2.1
  have c42308 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_6
  simp only [← c42303, ← c42304, ← c42305] at e_673_6
  have e_673_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_7
  simp only [← c42306, ← c42307, k2, ← c42308] at e_673_7
  have hr := e_673_7
  simp only [e_673_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4252)) := by
  have c42309 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.1
  have c42310 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.1
  have c42311 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.1
  have c42312 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.1
  have c42313 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.1
  have c42314 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_8
  simp only [← c42309, ← c42310, ← c42311] at e_673_8
  have e_673_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_9
  simp only [← c42312, ← c42313, k2, ← c42314] at e_673_9
  have hr := e_673_9
  simp only [e_673_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4253)) := by
  have c42315 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42316 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42317 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42318 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42319 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42320 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_10
  simp only [← c42315, ← c42316, ← c42317] at e_673_10
  have e_673_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_11
  simp only [← c42318, ← c42319, k2, ← c42320] at e_673_11
  have hr := e_673_11
  simp only [e_673_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4254)) := by
  have c42321 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42322 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42323 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42324 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42325 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42326 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_12
  simp only [← c42321, ← c42322, ← c42323] at e_673_12
  have e_673_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_13
  simp only [← c42324, ← c42325, k2, ← c42326] at e_673_13
  have hr := e_673_13
  simp only [e_673_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4255)) := by
  have c42327 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42328 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42329 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42330 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42331 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42332 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_14
  simp only [← c42327, ← c42328, ← c42329] at e_673_14
  have e_673_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_15
  simp only [← c42330, ← c42331, k2, ← c42332] at e_673_15
  have hr := e_673_15
  simp only [e_673_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4256)) := by
  have c42333 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42334 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42335 := (publicBatchWrapper86_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42336 := (publicBatchWrapper86_copies1323 a h).1
  have c42337 := (publicBatchWrapper86_copies1323 a h).2.1
  have c42338 := (publicBatchWrapper86_copies1323 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_16
  simp only [← c42333, ← c42334, ← c42335] at e_673_16
  have e_673_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_17
  simp only [← c42336, ← c42337, k2, ← c42338] at e_673_17
  have hr := e_673_17
  simp only [e_673_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 673 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4257)) := by
  have c42339 := (publicBatchWrapper86_copies1323 a h).2.2.2.1
  have c42340 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.1
  have c42341 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.1
  have c42342 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.1
  have c42343 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.1
  have c42344 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_673_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_18
  simp only [← c42339, ← c42340, ← c42341] at e_673_18
  have e_673_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_19
  simp only [← c42342, ← c42343, k2, ← c42344] at e_673_19
  have hr := e_673_19
  simp only [e_673_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4258)) := by
  have c42345 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.1
  have c42346 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42347 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42348 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42349 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42350 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_0
  simp only [← c42345, ← c42346, ← c42347] at e_674_0
  have e_674_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_1
  simp only [← c42348, ← c42349, k2, ← c42350] at e_674_1
  have hr := e_674_1
  simp only [e_674_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4259)) := by
  have c42351 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42352 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42353 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42354 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42355 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42356 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_2
  simp only [← c42351, ← c42352, ← c42353] at e_674_2
  have e_674_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_3
  simp only [← c42354, ← c42355, k2, ← c42356] at e_674_3
  have hr := e_674_3
  simp only [e_674_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4260)) := by
  have c42357 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42358 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42359 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42360 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42361 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42362 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_4
  simp only [← c42357, ← c42358, ← c42359] at e_674_4
  have e_674_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_5
  simp only [← c42360, ← c42361, k2, ← c42362] at e_674_5
  have hr := e_674_5
  simp only [e_674_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4261)) := by
  have c42363 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42364 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42365 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42366 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42367 := (publicBatchWrapper86_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42368 := (publicBatchWrapper86_copies1324 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_6
  simp only [← c42363, ← c42364, ← c42365] at e_674_6
  have e_674_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_7
  simp only [← c42366, ← c42367, k2, ← c42368] at e_674_7
  have hr := e_674_7
  simp only [e_674_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4262)) := by
  have c42369 := (publicBatchWrapper86_copies1324 a h).2.1
  have c42370 := (publicBatchWrapper86_copies1324 a h).2.2.1
  have c42371 := (publicBatchWrapper86_copies1324 a h).2.2.2.1
  have c42372 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.1
  have c42373 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.1
  have c42374 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_8
  simp only [← c42369, ← c42370, ← c42371] at e_674_8
  have e_674_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_9
  simp only [← c42372, ← c42373, k2, ← c42374] at e_674_9
  have hr := e_674_9
  simp only [e_674_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4263)) := by
  have c42375 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.1
  have c42376 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.1
  have c42377 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.1
  have c42378 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42379 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42380 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_10
  simp only [← c42375, ← c42376, ← c42377] at e_674_10
  have e_674_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_11
  simp only [← c42378, ← c42379, k2, ← c42380] at e_674_11
  have hr := e_674_11
  simp only [e_674_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4264)) := by
  have c42381 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42382 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42383 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42384 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42385 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42386 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_12
  simp only [← c42381, ← c42382, ← c42383] at e_674_12
  have e_674_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_13
  simp only [← c42384, ← c42385, k2, ← c42386] at e_674_13
  have hr := e_674_13
  simp only [e_674_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4265)) := by
  have c42387 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42388 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42389 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42390 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42391 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42392 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_14
  simp only [← c42387, ← c42388, ← c42389] at e_674_14
  have e_674_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_15
  simp only [← c42390, ← c42391, k2, ← c42392] at e_674_15
  have hr := e_674_15
  simp only [e_674_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4266)) := by
  have c42393 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42394 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42395 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42396 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42397 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42398 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_16
  simp only [← c42393, ← c42394, ← c42395] at e_674_16
  have e_674_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_17
  simp only [← c42396, ← c42397, k2, ← c42398] at e_674_17
  have hr := e_674_17
  simp only [e_674_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 674 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4267)) := by
  have c42399 := (publicBatchWrapper86_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42400 := (publicBatchWrapper86_copies1325 a h).1
  have c42401 := (publicBatchWrapper86_copies1325 a h).2.1
  have c42402 := (publicBatchWrapper86_copies1325 a h).2.2.1
  have c42403 := (publicBatchWrapper86_copies1325 a h).2.2.2.1
  have c42404 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_674_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_18
  simp only [← c42399, ← c42400, ← c42401] at e_674_18
  have e_674_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_19
  simp only [← c42402, ← c42403, k2, ← c42404] at e_674_19
  have hr := e_674_19
  simp only [e_674_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4268)) := by
  have c42405 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.1
  have c42406 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.1
  have c42407 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.1
  have c42408 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.1
  have c42409 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.1
  have c42410 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_0
  simp only [← c42405, ← c42406, ← c42407] at e_675_0
  have e_675_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_1
  simp only [← c42408, ← c42409, k2, ← c42410] at e_675_1
  have hr := e_675_1
  simp only [e_675_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4269)) := by
  have c42411 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42412 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42413 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42414 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42415 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42416 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_2
  simp only [← c42411, ← c42412, ← c42413] at e_675_2
  have e_675_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_3
  simp only [← c42414, ← c42415, k2, ← c42416] at e_675_3
  have hr := e_675_3
  simp only [e_675_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4270)) := by
  have c42417 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42418 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42419 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42420 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42421 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42422 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_4
  simp only [← c42417, ← c42418, ← c42419] at e_675_4
  have e_675_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_5
  simp only [← c42420, ← c42421, k2, ← c42422] at e_675_5
  have hr := e_675_5
  simp only [e_675_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4271)) := by
  have c42423 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42424 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42425 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42426 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42427 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42428 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_6
  simp only [← c42423, ← c42424, ← c42425] at e_675_6
  have e_675_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_7
  simp only [← c42426, ← c42427, k2, ← c42428] at e_675_7
  have hr := e_675_7
  simp only [e_675_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4272)) := by
  have c42429 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42430 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42431 := (publicBatchWrapper86_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42432 := (publicBatchWrapper86_copies1326 a h).1
  have c42433 := (publicBatchWrapper86_copies1326 a h).2.1
  have c42434 := (publicBatchWrapper86_copies1326 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_8
  simp only [← c42429, ← c42430, ← c42431] at e_675_8
  have e_675_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_9
  simp only [← c42432, ← c42433, k2, ← c42434] at e_675_9
  have hr := e_675_9
  simp only [e_675_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4273)) := by
  have c42435 := (publicBatchWrapper86_copies1326 a h).2.2.2.1
  have c42436 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.1
  have c42437 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.1
  have c42438 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.1
  have c42439 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.1
  have c42440 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_10
  simp only [← c42435, ← c42436, ← c42437] at e_675_10
  have e_675_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_11
  simp only [← c42438, ← c42439, k2, ← c42440] at e_675_11
  have hr := e_675_11
  simp only [e_675_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4274)) := by
  have c42441 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.1
  have c42442 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42443 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42444 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42445 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42446 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_12
  simp only [← c42441, ← c42442, ← c42443] at e_675_12
  have e_675_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_13
  simp only [← c42444, ← c42445, k2, ← c42446] at e_675_13
  have hr := e_675_13
  simp only [e_675_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4275)) := by
  have c42447 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42448 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42449 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42450 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42451 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42452 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_14
  simp only [← c42447, ← c42448, ← c42449] at e_675_14
  have e_675_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_15
  simp only [← c42450, ← c42451, k2, ← c42452] at e_675_15
  have hr := e_675_15
  simp only [e_675_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4276)) := by
  have c42453 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42454 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42455 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42456 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42457 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42458 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_16
  simp only [← c42453, ← c42454, ← c42455] at e_675_16
  have e_675_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_17
  simp only [← c42456, ← c42457, k2, ← c42458] at e_675_17
  have hr := e_675_17
  simp only [e_675_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 675 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4277)) := by
  have c42459 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42460 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42461 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42462 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42463 := (publicBatchWrapper86_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42464 := (publicBatchWrapper86_copies1327 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_675_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_18
  simp only [← c42459, ← c42460, ← c42461] at e_675_18
  have e_675_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_19
  simp only [← c42462, ← c42463, k2, ← c42464] at e_675_19
  have hr := e_675_19
  simp only [e_675_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4278)) := by
  have c42465 := (publicBatchWrapper86_copies1327 a h).2.1
  have c42466 := (publicBatchWrapper86_copies1327 a h).2.2.1
  have c42467 := (publicBatchWrapper86_copies1327 a h).2.2.2.1
  have c42468 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.1
  have c42469 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.1
  have c42470 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_0
  simp only [← c42465, ← c42466, ← c42467] at e_676_0
  have e_676_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_1
  simp only [← c42468, ← c42469, k2, ← c42470] at e_676_1
  have hr := e_676_1
  simp only [e_676_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4279)) := by
  have c42471 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.1
  have c42472 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.1
  have c42473 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.1
  have c42474 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42475 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42476 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_2
  simp only [← c42471, ← c42472, ← c42473] at e_676_2
  have e_676_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_3
  simp only [← c42474, ← c42475, k2, ← c42476] at e_676_3
  have hr := e_676_3
  simp only [e_676_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4280)) := by
  have c42477 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42478 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42479 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42480 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42481 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42482 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_4
  simp only [← c42477, ← c42478, ← c42479] at e_676_4
  have e_676_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_5
  simp only [← c42480, ← c42481, k2, ← c42482] at e_676_5
  have hr := e_676_5
  simp only [e_676_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4281)) := by
  have c42483 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42484 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42485 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42486 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42487 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42488 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_6
  simp only [← c42483, ← c42484, ← c42485] at e_676_6
  have e_676_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_7
  simp only [← c42486, ← c42487, k2, ← c42488] at e_676_7
  have hr := e_676_7
  simp only [e_676_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4282)) := by
  have c42489 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42490 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42491 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42492 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42493 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42494 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_8
  simp only [← c42489, ← c42490, ← c42491] at e_676_8
  have e_676_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_9
  simp only [← c42492, ← c42493, k2, ← c42494] at e_676_9
  have hr := e_676_9
  simp only [e_676_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4283)) := by
  have c42495 := (publicBatchWrapper86_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42496 := (publicBatchWrapper86_copies1328 a h).1
  have c42497 := (publicBatchWrapper86_copies1328 a h).2.1
  have c42498 := (publicBatchWrapper86_copies1328 a h).2.2.1
  have c42499 := (publicBatchWrapper86_copies1328 a h).2.2.2.1
  have c42500 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_10
  simp only [← c42495, ← c42496, ← c42497] at e_676_10
  have e_676_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_11
  simp only [← c42498, ← c42499, k2, ← c42500] at e_676_11
  have hr := e_676_11
  simp only [e_676_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4284)) := by
  have c42501 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.1
  have c42502 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.1
  have c42503 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.1
  have c42504 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.1
  have c42505 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.1
  have c42506 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_12
  simp only [← c42501, ← c42502, ← c42503] at e_676_12
  have e_676_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_13
  simp only [← c42504, ← c42505, k2, ← c42506] at e_676_13
  have hr := e_676_13
  simp only [e_676_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4285)) := by
  have c42507 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42508 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42509 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42510 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42511 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42512 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_14
  simp only [← c42507, ← c42508, ← c42509] at e_676_14
  have e_676_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_15
  simp only [← c42510, ← c42511, k2, ← c42512] at e_676_15
  have hr := e_676_15
  simp only [e_676_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4286)) := by
  have c42513 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42514 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42515 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42516 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42517 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42518 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_16
  simp only [← c42513, ← c42514, ← c42515] at e_676_16
  have e_676_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_17
  simp only [← c42516, ← c42517, k2, ← c42518] at e_676_17
  have hr := e_676_17
  simp only [e_676_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 676 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4287)) := by
  have c42519 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42520 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42521 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42522 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42523 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42524 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_676_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_18
  simp only [← c42519, ← c42520, ← c42521] at e_676_18
  have e_676_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_19
  simp only [← c42522, ← c42523, k2, ← c42524] at e_676_19
  have hr := e_676_19
  simp only [e_676_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4288)) := by
  have c42525 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42526 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42527 := (publicBatchWrapper86_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42528 := (publicBatchWrapper86_copies1329 a h).1
  have c42529 := (publicBatchWrapper86_copies1329 a h).2.1
  have c42530 := (publicBatchWrapper86_copies1329 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_0
  simp only [← c42525, ← c42526, ← c42527] at e_677_0
  have e_677_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_1
  simp only [← c42528, ← c42529, k2, ← c42530] at e_677_1
  have hr := e_677_1
  simp only [e_677_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4289)) := by
  have c42531 := (publicBatchWrapper86_copies1329 a h).2.2.2.1
  have c42532 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.1
  have c42533 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.1
  have c42534 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.1
  have c42535 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.1
  have c42536 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_2
  simp only [← c42531, ← c42532, ← c42533] at e_677_2
  have e_677_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_3
  simp only [← c42534, ← c42535, k2, ← c42536] at e_677_3
  have hr := e_677_3
  simp only [e_677_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4290)) := by
  have c42537 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.1
  have c42538 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42539 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42540 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42541 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42542 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_4
  simp only [← c42537, ← c42538, ← c42539] at e_677_4
  have e_677_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_5
  simp only [← c42540, ← c42541, k2, ← c42542] at e_677_5
  have hr := e_677_5
  simp only [e_677_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4291)) := by
  have c42543 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42544 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42545 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42546 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42547 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42548 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_6
  simp only [← c42543, ← c42544, ← c42545] at e_677_6
  have e_677_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_7
  simp only [← c42546, ← c42547, k2, ← c42548] at e_677_7
  have hr := e_677_7
  simp only [e_677_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4292)) := by
  have c42549 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42550 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42551 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42552 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42553 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42554 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_8
  simp only [← c42549, ← c42550, ← c42551] at e_677_8
  have e_677_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_9
  simp only [← c42552, ← c42553, k2, ← c42554] at e_677_9
  have hr := e_677_9
  simp only [e_677_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4293)) := by
  have c42555 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42556 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42557 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42558 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42559 := (publicBatchWrapper86_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42560 := (publicBatchWrapper86_copies1330 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_10
  simp only [← c42555, ← c42556, ← c42557] at e_677_10
  have e_677_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_11
  simp only [← c42558, ← c42559, k2, ← c42560] at e_677_11
  have hr := e_677_11
  simp only [e_677_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4294)) := by
  have c42561 := (publicBatchWrapper86_copies1330 a h).2.1
  have c42562 := (publicBatchWrapper86_copies1330 a h).2.2.1
  have c42563 := (publicBatchWrapper86_copies1330 a h).2.2.2.1
  have c42564 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.1
  have c42565 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.1
  have c42566 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_12
  simp only [← c42561, ← c42562, ← c42563] at e_677_12
  have e_677_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_13
  simp only [← c42564, ← c42565, k2, ← c42566] at e_677_13
  have hr := e_677_13
  simp only [e_677_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4295)) := by
  have c42567 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.1
  have c42568 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.1
  have c42569 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.1
  have c42570 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42571 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42572 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_14
  simp only [← c42567, ← c42568, ← c42569] at e_677_14
  have e_677_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_15
  simp only [← c42570, ← c42571, k2, ← c42572] at e_677_15
  have hr := e_677_15
  simp only [e_677_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4296)) := by
  have c42573 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42574 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42575 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42576 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42577 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42578 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_16
  simp only [← c42573, ← c42574, ← c42575] at e_677_16
  have e_677_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_17
  simp only [← c42576, ← c42577, k2, ← c42578] at e_677_17
  have hr := e_677_17
  simp only [e_677_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 677 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4297)) := by
  have c42579 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42580 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42581 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42582 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42583 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42584 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_677_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_18
  simp only [← c42579, ← c42580, ← c42581] at e_677_18
  have e_677_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_19
  simp only [← c42582, ← c42583, k2, ← c42584] at e_677_19
  have hr := e_677_19
  simp only [e_677_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4298)) := by
  have c42585 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42586 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42587 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42588 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42589 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42590 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_0
  simp only [← c42585, ← c42586, ← c42587] at e_678_0
  have e_678_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_1
  simp only [← c42588, ← c42589, k2, ← c42590] at e_678_1
  have hr := e_678_1
  simp only [e_678_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4299)) := by
  have c42591 := (publicBatchWrapper86_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42592 := (publicBatchWrapper86_copies1331 a h).1
  have c42593 := (publicBatchWrapper86_copies1331 a h).2.1
  have c42594 := (publicBatchWrapper86_copies1331 a h).2.2.1
  have c42595 := (publicBatchWrapper86_copies1331 a h).2.2.2.1
  have c42596 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_2
  simp only [← c42591, ← c42592, ← c42593] at e_678_2
  have e_678_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_3
  simp only [← c42594, ← c42595, k2, ← c42596] at e_678_3
  have hr := e_678_3
  simp only [e_678_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4300)) := by
  have c42597 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.1
  have c42598 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.1
  have c42599 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.1
  have c42600 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.1
  have c42601 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.1
  have c42602 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_4
  simp only [← c42597, ← c42598, ← c42599] at e_678_4
  have e_678_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_5
  simp only [← c42600, ← c42601, k2, ← c42602] at e_678_5
  have hr := e_678_5
  simp only [e_678_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4301)) := by
  have c42603 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42604 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42605 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42606 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42607 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42608 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_6
  simp only [← c42603, ← c42604, ← c42605] at e_678_6
  have e_678_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_7
  simp only [← c42606, ← c42607, k2, ← c42608] at e_678_7
  have hr := e_678_7
  simp only [e_678_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4302)) := by
  have c42609 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42610 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42611 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42612 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42613 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42614 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_8
  simp only [← c42609, ← c42610, ← c42611] at e_678_8
  have e_678_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_9
  simp only [← c42612, ← c42613, k2, ← c42614] at e_678_9
  have hr := e_678_9
  simp only [e_678_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4303)) := by
  have c42615 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42616 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42617 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42618 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42619 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42620 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_10
  simp only [← c42615, ← c42616, ← c42617] at e_678_10
  have e_678_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_11
  simp only [← c42618, ← c42619, k2, ← c42620] at e_678_11
  have hr := e_678_11
  simp only [e_678_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4304)) := by
  have c42621 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42622 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42623 := (publicBatchWrapper86_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42624 := (publicBatchWrapper86_copies1332 a h).1
  have c42625 := (publicBatchWrapper86_copies1332 a h).2.1
  have c42626 := (publicBatchWrapper86_copies1332 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_12
  simp only [← c42621, ← c42622, ← c42623] at e_678_12
  have e_678_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_13
  simp only [← c42624, ← c42625, k2, ← c42626] at e_678_13
  have hr := e_678_13
  simp only [e_678_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4305)) := by
  have c42627 := (publicBatchWrapper86_copies1332 a h).2.2.2.1
  have c42628 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.1
  have c42629 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.1
  have c42630 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.1
  have c42631 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.1
  have c42632 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_14
  simp only [← c42627, ← c42628, ← c42629] at e_678_14
  have e_678_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_15
  simp only [← c42630, ← c42631, k2, ← c42632] at e_678_15
  have hr := e_678_15
  simp only [e_678_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4306)) := by
  have c42633 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.1
  have c42634 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42635 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42636 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42637 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42638 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_16
  simp only [← c42633, ← c42634, ← c42635] at e_678_16
  have e_678_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_17
  simp only [← c42636, ← c42637, k2, ← c42638] at e_678_17
  have hr := e_678_17
  simp only [e_678_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 678 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4307)) := by
  have c42639 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42640 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42641 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42642 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42643 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42644 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_678_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_18
  simp only [← c42639, ← c42640, ← c42641] at e_678_18
  have e_678_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_19
  simp only [← c42642, ← c42643, k2, ← c42644] at e_678_19
  have hr := e_678_19
  simp only [e_678_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4308)) := by
  have c42645 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42646 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42647 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42648 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42649 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42650 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_0
  simp only [← c42645, ← c42646, ← c42647] at e_679_0
  have e_679_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_1
  simp only [← c42648, ← c42649, k2, ← c42650] at e_679_1
  have hr := e_679_1
  simp only [e_679_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4309)) := by
  have c42651 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42652 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42653 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42654 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42655 := (publicBatchWrapper86_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42656 := (publicBatchWrapper86_copies1333 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_2
  simp only [← c42651, ← c42652, ← c42653] at e_679_2
  have e_679_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_3
  simp only [← c42654, ← c42655, k2, ← c42656] at e_679_3
  have hr := e_679_3
  simp only [e_679_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4310)) := by
  have c42657 := (publicBatchWrapper86_copies1333 a h).2.1
  have c42658 := (publicBatchWrapper86_copies1333 a h).2.2.1
  have c42659 := (publicBatchWrapper86_copies1333 a h).2.2.2.1
  have c42660 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.1
  have c42661 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.1
  have c42662 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_4
  simp only [← c42657, ← c42658, ← c42659] at e_679_4
  have e_679_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_5
  simp only [← c42660, ← c42661, k2, ← c42662] at e_679_5
  have hr := e_679_5
  simp only [e_679_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4311)) := by
  have c42663 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.1
  have c42664 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.1
  have c42665 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.1
  have c42666 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42667 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42668 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_6
  simp only [← c42663, ← c42664, ← c42665] at e_679_6
  have e_679_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_7
  simp only [← c42666, ← c42667, k2, ← c42668] at e_679_7
  have hr := e_679_7
  simp only [e_679_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4312)) := by
  have c42669 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42670 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42671 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42672 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42673 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42674 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_8
  simp only [← c42669, ← c42670, ← c42671] at e_679_8
  have e_679_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_9
  simp only [← c42672, ← c42673, k2, ← c42674] at e_679_9
  have hr := e_679_9
  simp only [e_679_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4313)) := by
  have c42675 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42676 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42677 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42678 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42679 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42680 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_10
  simp only [← c42675, ← c42676, ← c42677] at e_679_10
  have e_679_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_11
  simp only [← c42678, ← c42679, k2, ← c42680] at e_679_11
  have hr := e_679_11
  simp only [e_679_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4314)) := by
  have c42681 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42682 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42683 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42684 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42685 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42686 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_12
  simp only [← c42681, ← c42682, ← c42683] at e_679_12
  have e_679_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_13
  simp only [← c42684, ← c42685, k2, ← c42686] at e_679_13
  have hr := e_679_13
  simp only [e_679_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4315)) := by
  have c42687 := (publicBatchWrapper86_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42688 := (publicBatchWrapper86_copies1334 a h).1
  have c42689 := (publicBatchWrapper86_copies1334 a h).2.1
  have c42690 := (publicBatchWrapper86_copies1334 a h).2.2.1
  have c42691 := (publicBatchWrapper86_copies1334 a h).2.2.2.1
  have c42692 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_14
  simp only [← c42687, ← c42688, ← c42689] at e_679_14
  have e_679_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_15
  simp only [← c42690, ← c42691, k2, ← c42692] at e_679_15
  have hr := e_679_15
  simp only [e_679_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4316)) := by
  have c42693 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.1
  have c42694 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.1
  have c42695 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.1
  have c42696 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.1
  have c42697 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.1
  have c42698 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_16
  simp only [← c42693, ← c42694, ← c42695] at e_679_16
  have e_679_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_17
  simp only [← c42696, ← c42697, k2, ← c42698] at e_679_17
  have hr := e_679_17
  simp only [e_679_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 679 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4317)) := by
  have c42699 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42700 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42701 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42702 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42703 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42704 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_679_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_18
  simp only [← c42699, ← c42700, ← c42701] at e_679_18
  have e_679_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_19
  simp only [← c42702, ← c42703, k2, ← c42704] at e_679_19
  have hr := e_679_19
  simp only [e_679_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4318)) := by
  have c42705 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42706 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42707 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42708 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42709 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42710 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_0
  simp only [← c42705, ← c42706, ← c42707] at e_680_0
  have e_680_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_1
  simp only [← c42708, ← c42709, k2, ← c42710] at e_680_1
  have hr := e_680_1
  simp only [e_680_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4319)) := by
  have c42711 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42712 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42713 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42714 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42715 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42716 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_2
  simp only [← c42711, ← c42712, ← c42713] at e_680_2
  have e_680_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_3
  simp only [← c42714, ← c42715, k2, ← c42716] at e_680_3
  have hr := e_680_3
  simp only [e_680_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4320)) := by
  have c42717 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42718 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42719 := (publicBatchWrapper86_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42720 := (publicBatchWrapper86_copies1335 a h).1
  have c42721 := (publicBatchWrapper86_copies1335 a h).2.1
  have c42722 := (publicBatchWrapper86_copies1335 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_4
  simp only [← c42717, ← c42718, ← c42719] at e_680_4
  have e_680_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_5
  simp only [← c42720, ← c42721, k2, ← c42722] at e_680_5
  have hr := e_680_5
  simp only [e_680_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4321)) := by
  have c42723 := (publicBatchWrapper86_copies1335 a h).2.2.2.1
  have c42724 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.1
  have c42725 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.1
  have c42726 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.1
  have c42727 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.1
  have c42728 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_6
  simp only [← c42723, ← c42724, ← c42725] at e_680_6
  have e_680_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_7
  simp only [← c42726, ← c42727, k2, ← c42728] at e_680_7
  have hr := e_680_7
  simp only [e_680_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4322)) := by
  have c42729 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.1
  have c42730 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42731 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42732 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42733 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42734 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_8
  simp only [← c42729, ← c42730, ← c42731] at e_680_8
  have e_680_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_9
  simp only [← c42732, ← c42733, k2, ← c42734] at e_680_9
  have hr := e_680_9
  simp only [e_680_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4323)) := by
  have c42735 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42736 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42737 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42738 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42739 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42740 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_10
  simp only [← c42735, ← c42736, ← c42737] at e_680_10
  have e_680_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_11
  simp only [← c42738, ← c42739, k2, ← c42740] at e_680_11
  have hr := e_680_11
  simp only [e_680_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4324)) := by
  have c42741 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42742 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42743 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42744 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42745 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42746 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_12
  simp only [← c42741, ← c42742, ← c42743] at e_680_12
  have e_680_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_13
  simp only [← c42744, ← c42745, k2, ← c42746] at e_680_13
  have hr := e_680_13
  simp only [e_680_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4325)) := by
  have c42747 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42748 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42749 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42750 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42751 := (publicBatchWrapper86_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42752 := (publicBatchWrapper86_copies1336 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_14
  simp only [← c42747, ← c42748, ← c42749] at e_680_14
  have e_680_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_15
  simp only [← c42750, ← c42751, k2, ← c42752] at e_680_15
  have hr := e_680_15
  simp only [e_680_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4326)) := by
  have c42753 := (publicBatchWrapper86_copies1336 a h).2.1
  have c42754 := (publicBatchWrapper86_copies1336 a h).2.2.1
  have c42755 := (publicBatchWrapper86_copies1336 a h).2.2.2.1
  have c42756 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.1
  have c42757 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.1
  have c42758 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_16
  simp only [← c42753, ← c42754, ← c42755] at e_680_16
  have e_680_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_17
  simp only [← c42756, ← c42757, k2, ← c42758] at e_680_17
  have hr := e_680_17
  simp only [e_680_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 680 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4327)) := by
  have c42759 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.1
  have c42760 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.1
  have c42761 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.1
  have c42762 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42763 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42764 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_680_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_18
  simp only [← c42759, ← c42760, ← c42761] at e_680_18
  have e_680_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_19
  simp only [← c42762, ← c42763, k2, ← c42764] at e_680_19
  have hr := e_680_19
  simp only [e_680_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4328)) := by
  have c42765 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42766 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42767 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42768 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42769 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42770 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_0
  simp only [← c42765, ← c42766, ← c42767] at e_681_0
  have e_681_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_1
  simp only [← c42768, ← c42769, k2, ← c42770] at e_681_1
  have hr := e_681_1
  simp only [e_681_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4329)) := by
  have c42771 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42772 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42773 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42774 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42775 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42776 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_2
  simp only [← c42771, ← c42772, ← c42773] at e_681_2
  have e_681_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_3
  simp only [← c42774, ← c42775, k2, ← c42776] at e_681_3
  have hr := e_681_3
  simp only [e_681_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4330)) := by
  have c42777 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42778 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42779 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42780 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42781 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42782 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_4
  simp only [← c42777, ← c42778, ← c42779] at e_681_4
  have e_681_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_5
  simp only [← c42780, ← c42781, k2, ← c42782] at e_681_5
  have hr := e_681_5
  simp only [e_681_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4331)) := by
  have c42783 := (publicBatchWrapper86_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42784 := (publicBatchWrapper86_copies1337 a h).1
  have c42785 := (publicBatchWrapper86_copies1337 a h).2.1
  have c42786 := (publicBatchWrapper86_copies1337 a h).2.2.1
  have c42787 := (publicBatchWrapper86_copies1337 a h).2.2.2.1
  have c42788 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_6
  simp only [← c42783, ← c42784, ← c42785] at e_681_6
  have e_681_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_7
  simp only [← c42786, ← c42787, k2, ← c42788] at e_681_7
  have hr := e_681_7
  simp only [e_681_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4332)) := by
  have c42789 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.1
  have c42790 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.1
  have c42791 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.1
  have c42792 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.1
  have c42793 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.1
  have c42794 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_8
  simp only [← c42789, ← c42790, ← c42791] at e_681_8
  have e_681_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_9
  simp only [← c42792, ← c42793, k2, ← c42794] at e_681_9
  have hr := e_681_9
  simp only [e_681_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4333)) := by
  have c42795 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42796 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42797 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42798 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42799 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42800 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_10
  simp only [← c42795, ← c42796, ← c42797] at e_681_10
  have e_681_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_11
  simp only [← c42798, ← c42799, k2, ← c42800] at e_681_11
  have hr := e_681_11
  simp only [e_681_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4334)) := by
  have c42801 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42802 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42803 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42804 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42805 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42806 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_12
  simp only [← c42801, ← c42802, ← c42803] at e_681_12
  have e_681_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_13
  simp only [← c42804, ← c42805, k2, ← c42806] at e_681_13
  have hr := e_681_13
  simp only [e_681_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4335)) := by
  have c42807 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42808 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42809 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42810 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42811 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42812 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_14
  simp only [← c42807, ← c42808, ← c42809] at e_681_14
  have e_681_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_15
  simp only [← c42810, ← c42811, k2, ← c42812] at e_681_15
  have hr := e_681_15
  simp only [e_681_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4336)) := by
  have c42813 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42814 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42815 := (publicBatchWrapper86_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42816 := (publicBatchWrapper86_copies1338 a h).1
  have c42817 := (publicBatchWrapper86_copies1338 a h).2.1
  have c42818 := (publicBatchWrapper86_copies1338 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_16
  simp only [← c42813, ← c42814, ← c42815] at e_681_16
  have e_681_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_17
  simp only [← c42816, ← c42817, k2, ← c42818] at e_681_17
  have hr := e_681_17
  simp only [e_681_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 681 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4337)) := by
  have c42819 := (publicBatchWrapper86_copies1338 a h).2.2.2.1
  have c42820 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.1
  have c42821 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.1
  have c42822 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.1
  have c42823 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.1
  have c42824 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_681_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_18
  simp only [← c42819, ← c42820, ← c42821] at e_681_18
  have e_681_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_19
  simp only [← c42822, ← c42823, k2, ← c42824] at e_681_19
  have hr := e_681_19
  simp only [e_681_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4338)) := by
  have c42825 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.1
  have c42826 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42827 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42828 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42829 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42830 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_0
  simp only [← c42825, ← c42826, ← c42827] at e_682_0
  have e_682_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_1
  simp only [← c42828, ← c42829, k2, ← c42830] at e_682_1
  have hr := e_682_1
  simp only [e_682_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4339)) := by
  have c42831 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42832 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42833 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42834 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42835 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42836 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_2
  simp only [← c42831, ← c42832, ← c42833] at e_682_2
  have e_682_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_3
  simp only [← c42834, ← c42835, k2, ← c42836] at e_682_3
  have hr := e_682_3
  simp only [e_682_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4340)) := by
  have c42837 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42838 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42839 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42840 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42841 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42842 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_4
  simp only [← c42837, ← c42838, ← c42839] at e_682_4
  have e_682_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_5
  simp only [← c42840, ← c42841, k2, ← c42842] at e_682_5
  have hr := e_682_5
  simp only [e_682_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4341)) := by
  have c42843 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42844 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42845 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42846 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42847 := (publicBatchWrapper86_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42848 := (publicBatchWrapper86_copies1339 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_6
  simp only [← c42843, ← c42844, ← c42845] at e_682_6
  have e_682_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_7
  simp only [← c42846, ← c42847, k2, ← c42848] at e_682_7
  have hr := e_682_7
  simp only [e_682_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4342)) := by
  have c42849 := (publicBatchWrapper86_copies1339 a h).2.1
  have c42850 := (publicBatchWrapper86_copies1339 a h).2.2.1
  have c42851 := (publicBatchWrapper86_copies1339 a h).2.2.2.1
  have c42852 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.1
  have c42853 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.1
  have c42854 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_8
  simp only [← c42849, ← c42850, ← c42851] at e_682_8
  have e_682_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_9
  simp only [← c42852, ← c42853, k2, ← c42854] at e_682_9
  have hr := e_682_9
  simp only [e_682_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4343)) := by
  have c42855 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.1
  have c42856 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.1
  have c42857 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.1
  have c42858 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42859 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42860 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_10
  simp only [← c42855, ← c42856, ← c42857] at e_682_10
  have e_682_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_11
  simp only [← c42858, ← c42859, k2, ← c42860] at e_682_11
  have hr := e_682_11
  simp only [e_682_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4344)) := by
  have c42861 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42862 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42863 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42864 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42865 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42866 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_12
  simp only [← c42861, ← c42862, ← c42863] at e_682_12
  have e_682_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_13
  simp only [← c42864, ← c42865, k2, ← c42866] at e_682_13
  have hr := e_682_13
  simp only [e_682_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4345)) := by
  have c42867 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42868 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42869 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42870 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42871 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42872 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_14
  simp only [← c42867, ← c42868, ← c42869] at e_682_14
  have e_682_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_15
  simp only [← c42870, ← c42871, k2, ← c42872] at e_682_15
  have hr := e_682_15
  simp only [e_682_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4346)) := by
  have c42873 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42874 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42875 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42876 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42877 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42878 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_16
  simp only [← c42873, ← c42874, ← c42875] at e_682_16
  have e_682_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_17
  simp only [← c42876, ← c42877, k2, ← c42878] at e_682_17
  have hr := e_682_17
  simp only [e_682_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 682 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4347)) := by
  have c42879 := (publicBatchWrapper86_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42880 := (publicBatchWrapper86_copies1340 a h).1
  have c42881 := (publicBatchWrapper86_copies1340 a h).2.1
  have c42882 := (publicBatchWrapper86_copies1340 a h).2.2.1
  have c42883 := (publicBatchWrapper86_copies1340 a h).2.2.2.1
  have c42884 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_682_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_18
  simp only [← c42879, ← c42880, ← c42881] at e_682_18
  have e_682_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_19
  simp only [← c42882, ← c42883, k2, ← c42884] at e_682_19
  have hr := e_682_19
  simp only [e_682_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4348)) := by
  have c42885 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.1
  have c42886 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.1
  have c42887 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.1
  have c42888 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.1
  have c42889 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.1
  have c42890 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_0
  simp only [← c42885, ← c42886, ← c42887] at e_683_0
  have e_683_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_1
  simp only [← c42888, ← c42889, k2, ← c42890] at e_683_1
  have hr := e_683_1
  simp only [e_683_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4349)) := by
  have c42891 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42892 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42893 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42894 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42895 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42896 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_2
  simp only [← c42891, ← c42892, ← c42893] at e_683_2
  have e_683_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_3
  simp only [← c42894, ← c42895, k2, ← c42896] at e_683_3
  have hr := e_683_3
  simp only [e_683_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4350)) := by
  have c42897 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42898 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42899 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42900 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42901 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42902 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_4
  simp only [← c42897, ← c42898, ← c42899] at e_683_4
  have e_683_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_5
  simp only [← c42900, ← c42901, k2, ← c42902] at e_683_5
  have hr := e_683_5
  simp only [e_683_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4351)) := by
  have c42903 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42904 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42905 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42906 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42907 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42908 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_6
  simp only [← c42903, ← c42904, ← c42905] at e_683_6
  have e_683_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_7
  simp only [← c42906, ← c42907, k2, ← c42908] at e_683_7
  have hr := e_683_7
  simp only [e_683_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4352)) := by
  have c42909 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42910 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42911 := (publicBatchWrapper86_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42912 := (publicBatchWrapper86_copies1341 a h).1
  have c42913 := (publicBatchWrapper86_copies1341 a h).2.1
  have c42914 := (publicBatchWrapper86_copies1341 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_8
  simp only [← c42909, ← c42910, ← c42911] at e_683_8
  have e_683_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_9
  simp only [← c42912, ← c42913, k2, ← c42914] at e_683_9
  have hr := e_683_9
  simp only [e_683_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4353)) := by
  have c42915 := (publicBatchWrapper86_copies1341 a h).2.2.2.1
  have c42916 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.1
  have c42917 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.1
  have c42918 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.1
  have c42919 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.1
  have c42920 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_10
  simp only [← c42915, ← c42916, ← c42917] at e_683_10
  have e_683_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_11
  simp only [← c42918, ← c42919, k2, ← c42920] at e_683_11
  have hr := e_683_11
  simp only [e_683_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4354)) := by
  have c42921 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.1
  have c42922 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42923 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42924 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42925 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42926 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_12
  simp only [← c42921, ← c42922, ← c42923] at e_683_12
  have e_683_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_13
  simp only [← c42924, ← c42925, k2, ← c42926] at e_683_13
  have hr := e_683_13
  simp only [e_683_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4355)) := by
  have c42927 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42928 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42929 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42930 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42931 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42932 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_14
  simp only [← c42927, ← c42928, ← c42929] at e_683_14
  have e_683_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_15
  simp only [← c42930, ← c42931, k2, ← c42932] at e_683_15
  have hr := e_683_15
  simp only [e_683_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4356)) := by
  have c42933 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42934 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42935 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42936 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42937 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42938 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_16
  simp only [← c42933, ← c42934, ← c42935] at e_683_16
  have e_683_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_17
  simp only [← c42936, ← c42937, k2, ← c42938] at e_683_17
  have hr := e_683_17
  simp only [e_683_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 683 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4357)) := by
  have c42939 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42940 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42941 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42942 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42943 := (publicBatchWrapper86_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42944 := (publicBatchWrapper86_copies1342 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_683_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_18
  simp only [← c42939, ← c42940, ← c42941] at e_683_18
  have e_683_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_19
  simp only [← c42942, ← c42943, k2, ← c42944] at e_683_19
  have hr := e_683_19
  simp only [e_683_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4358)) := by
  have c42945 := (publicBatchWrapper86_copies1342 a h).2.1
  have c42946 := (publicBatchWrapper86_copies1342 a h).2.2.1
  have c42947 := (publicBatchWrapper86_copies1342 a h).2.2.2.1
  have c42948 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.1
  have c42949 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.1
  have c42950 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_0
  simp only [← c42945, ← c42946, ← c42947] at e_684_0
  have e_684_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_1
  simp only [← c42948, ← c42949, k2, ← c42950] at e_684_1
  have hr := e_684_1
  simp only [e_684_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4359)) := by
  have c42951 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.1
  have c42952 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.1
  have c42953 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.1
  have c42954 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42955 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42956 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_2
  simp only [← c42951, ← c42952, ← c42953] at e_684_2
  have e_684_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_3
  simp only [← c42954, ← c42955, k2, ← c42956] at e_684_3
  have hr := e_684_3
  simp only [e_684_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 23) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4360)) := by
  have c42957 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42958 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42959 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42960 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42961 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42962 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_4
  simp only [← c42957, ← c42958, ← c42959] at e_684_4
  have e_684_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_5
  simp only [← c42960, ← c42961, k2, ← c42962] at e_684_5
  have hr := e_684_5
  simp only [e_684_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 31) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4361)) := by
  have c42963 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42964 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42965 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42966 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42967 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42968 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_6
  simp only [← c42963, ← c42964, ← c42965] at e_684_6
  have e_684_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_7
  simp only [← c42966, ← c42967, k2, ← c42968] at e_684_7
  have hr := e_684_7
  simp only [e_684_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 39) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4362)) := by
  have c42969 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42970 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42971 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42972 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42973 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42974 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_8
  simp only [← c42969, ← c42970, ← c42971] at e_684_8
  have e_684_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_9
  simp only [← c42972, ← c42973, k2, ← c42974] at e_684_9
  have hr := e_684_9
  simp only [e_684_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 47) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4363)) := by
  have c42975 := (publicBatchWrapper86_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42976 := (publicBatchWrapper86_copies1343 a h).1
  have c42977 := (publicBatchWrapper86_copies1343 a h).2.1
  have c42978 := (publicBatchWrapper86_copies1343 a h).2.2.1
  have c42979 := (publicBatchWrapper86_copies1343 a h).2.2.2.1
  have c42980 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_10
  simp only [← c42975, ← c42976, ← c42977] at e_684_10
  have e_684_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_11
  simp only [← c42978, ← c42979, k2, ← c42980] at e_684_11
  have hr := e_684_11
  simp only [e_684_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 55) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4364)) := by
  have c42981 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.1
  have c42982 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.1
  have c42983 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.1
  have c42984 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.1
  have c42985 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.1
  have c42986 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_12
  simp only [← c42981, ← c42982, ← c42983] at e_684_12
  have e_684_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_13
  simp only [← c42984, ← c42985, k2, ← c42986] at e_684_13
  have hr := e_684_13
  simp only [e_684_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 63) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4365)) := by
  have c42987 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42988 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42989 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42990 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42991 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42992 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_14
  simp only [← c42987, ← c42988, ← c42989] at e_684_14
  have e_684_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_15
  simp only [← c42990, ← c42991, k2, ← c42992] at e_684_15
  have hr := e_684_15
  simp only [e_684_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 71) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4366)) := by
  have c42993 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42994 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42995 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42996 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42997 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42998 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_16
  simp only [← c42993, ← c42994, ← c42995] at e_684_16
  have e_684_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_17
  simp only [← c42996, ← c42997, k2, ← c42998] at e_684_17
  have hr := e_684_17
  simp only [e_684_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 684 79) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4367)) := by
  have c42999 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43000 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43001 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43002 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43003 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43004 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_684_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_18
  simp only [← c42999, ← c43000, ← c43001] at e_684_18
  have e_684_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_19
  simp only [← c43002, ← c43003, k2, ← c43004] at e_684_19
  have hr := e_684_19
  simp only [e_684_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 7) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4368)) := by
  have c43005 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43006 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43007 := (publicBatchWrapper86_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43008 := (publicBatchWrapper86_copies1344 a h).1
  have c43009 := (publicBatchWrapper86_copies1344 a h).2.1
  have c43010 := (publicBatchWrapper86_copies1344 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_0
  simp only [← c43005, ← c43006, ← c43007] at e_685_0
  have e_685_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_1
  simp only [← c43008, ← c43009, k2, ← c43010] at e_685_1
  have hr := e_685_1
  simp only [e_685_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 15) = bselect (a (.wire 22 71)) (a (.virt 200)) (a (.virt 4369)) := by
  have c43011 := (publicBatchWrapper86_copies1344 a h).2.2.2.1
  have c43012 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.1
  have c43013 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.1
  have c43014 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.1
  have c43015 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.1
  have c43016 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_2
  simp only [← c43011, ← c43012, ← c43013] at e_685_2
  have e_685_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_3
  simp only [← c43014, ← c43015, k2, ← c43016] at e_685_3
  have hr := e_685_3
  simp only [e_685_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4410)) := by
  have c43017 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.1
  have c43018 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43019 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43020 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43021 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43022 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_4
  simp only [← c43017, ← c43018, ← c43019] at e_685_4
  have e_685_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_5
  simp only [← c43020, ← c43021, k2, ← c43022] at e_685_5
  have hr := e_685_5
  simp only [e_685_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4411)) := by
  have c43023 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43024 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43025 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43026 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43027 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43028 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_6
  simp only [← c43023, ← c43024, ← c43025] at e_685_6
  have e_685_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_7
  simp only [← c43026, ← c43027, k2, ← c43028] at e_685_7
  have hr := e_685_7
  simp only [e_685_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4412)) := by
  have c43029 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43030 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43031 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43032 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43033 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43034 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_8
  simp only [← c43029, ← c43030, ← c43031] at e_685_8
  have e_685_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_9
  simp only [← c43032, ← c43033, k2, ← c43034] at e_685_9
  have hr := e_685_9
  simp only [e_685_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4413)) := by
  have c43035 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43036 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43037 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43038 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43039 := (publicBatchWrapper86_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43040 := (publicBatchWrapper86_copies1345 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_10
  simp only [← c43035, ← c43036, ← c43037] at e_685_10
  have e_685_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_11
  simp only [← c43038, ← c43039, k2, ← c43040] at e_685_11
  have hr := e_685_11
  simp only [e_685_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4414)) := by
  have c43041 := (publicBatchWrapper86_copies1345 a h).2.1
  have c43042 := (publicBatchWrapper86_copies1345 a h).2.2.1
  have c43043 := (publicBatchWrapper86_copies1345 a h).2.2.2.1
  have c43044 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.1
  have c43045 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.1
  have c43046 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_12
  simp only [← c43041, ← c43042, ← c43043] at e_685_12
  have e_685_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_13
  simp only [← c43044, ← c43045, k2, ← c43046] at e_685_13
  have hr := e_685_13
  simp only [e_685_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4415)) := by
  have c43047 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.1
  have c43048 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.1
  have c43049 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.1
  have c43050 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43051 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43052 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_14
  simp only [← c43047, ← c43048, ← c43049] at e_685_14
  have e_685_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_15
  simp only [← c43050, ← c43051, k2, ← c43052] at e_685_15
  have hr := e_685_15
  simp only [e_685_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4416)) := by
  have c43053 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43054 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43055 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43056 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43057 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43058 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_16
  simp only [← c43053, ← c43054, ← c43055] at e_685_16
  have e_685_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_17
  simp only [← c43056, ← c43057, k2, ← c43058] at e_685_17
  have hr := e_685_17
  simp only [e_685_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 685 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4417)) := by
  have c43059 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43060 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43061 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43062 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43063 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43064 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_685_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_18
  simp only [← c43059, ← c43060, ← c43061] at e_685_18
  have e_685_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_19
  simp only [← c43062, ← c43063, k2, ← c43064] at e_685_19
  have hr := e_685_19
  simp only [e_685_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4418)) := by
  have c43065 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43066 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43067 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43068 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43069 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43070 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_0
  simp only [← c43065, ← c43066, ← c43067] at e_686_0
  have e_686_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_1
  simp only [← c43068, ← c43069, k2, ← c43070] at e_686_1
  have hr := e_686_1
  simp only [e_686_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4419)) := by
  have c43071 := (publicBatchWrapper86_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43072 := (publicBatchWrapper86_copies1346 a h).1
  have c43073 := (publicBatchWrapper86_copies1346 a h).2.1
  have c43074 := (publicBatchWrapper86_copies1346 a h).2.2.1
  have c43075 := (publicBatchWrapper86_copies1346 a h).2.2.2.1
  have c43076 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_2
  simp only [← c43071, ← c43072, ← c43073] at e_686_2
  have e_686_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_3
  simp only [← c43074, ← c43075, k2, ← c43076] at e_686_3
  have hr := e_686_3
  simp only [e_686_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4420)) := by
  have c43077 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.1
  have c43078 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.1
  have c43079 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.1
  have c43080 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.1
  have c43081 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.1
  have c43082 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_4
  simp only [← c43077, ← c43078, ← c43079] at e_686_4
  have e_686_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_5
  simp only [← c43080, ← c43081, k2, ← c43082] at e_686_5
  have hr := e_686_5
  simp only [e_686_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4421)) := by
  have c43083 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43084 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43085 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43086 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43087 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43088 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_6
  simp only [← c43083, ← c43084, ← c43085] at e_686_6
  have e_686_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_7
  simp only [← c43086, ← c43087, k2, ← c43088] at e_686_7
  have hr := e_686_7
  simp only [e_686_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4422)) := by
  have c43089 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43090 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43091 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43092 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43093 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43094 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_8
  simp only [← c43089, ← c43090, ← c43091] at e_686_8
  have e_686_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_9
  simp only [← c43092, ← c43093, k2, ← c43094] at e_686_9
  have hr := e_686_9
  simp only [e_686_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4423)) := by
  have c43095 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43096 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43097 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43098 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43099 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43100 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_10
  simp only [← c43095, ← c43096, ← c43097] at e_686_10
  have e_686_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_11
  simp only [← c43098, ← c43099, k2, ← c43100] at e_686_11
  have hr := e_686_11
  simp only [e_686_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4424)) := by
  have c43101 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43102 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43103 := (publicBatchWrapper86_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43104 := (publicBatchWrapper86_copies1347 a h).1
  have c43105 := (publicBatchWrapper86_copies1347 a h).2.1
  have c43106 := (publicBatchWrapper86_copies1347 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_12
  simp only [← c43101, ← c43102, ← c43103] at e_686_12
  have e_686_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_13
  simp only [← c43104, ← c43105, k2, ← c43106] at e_686_13
  have hr := e_686_13
  simp only [e_686_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4425)) := by
  have c43107 := (publicBatchWrapper86_copies1347 a h).2.2.2.1
  have c43108 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.1
  have c43109 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.1
  have c43110 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.1
  have c43111 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.1
  have c43112 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_14
  simp only [← c43107, ← c43108, ← c43109] at e_686_14
  have e_686_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_15
  simp only [← c43110, ← c43111, k2, ← c43112] at e_686_15
  have hr := e_686_15
  simp only [e_686_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4426)) := by
  have c43113 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.1
  have c43114 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43115 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43116 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43117 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43118 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_16
  simp only [← c43113, ← c43114, ← c43115] at e_686_16
  have e_686_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_17
  simp only [← c43116, ← c43117, k2, ← c43118] at e_686_17
  have hr := e_686_17
  simp only [e_686_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 686 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4427)) := by
  have c43119 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43120 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43121 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43122 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43123 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43124 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_686_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_18
  simp only [← c43119, ← c43120, ← c43121] at e_686_18
  have e_686_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_19
  simp only [← c43122, ← c43123, k2, ← c43124] at e_686_19
  have hr := e_686_19
  simp only [e_686_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4428)) := by
  have c43125 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43126 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43127 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43128 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43129 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43130 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_0
  simp only [← c43125, ← c43126, ← c43127] at e_687_0
  have e_687_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_1
  simp only [← c43128, ← c43129, k2, ← c43130] at e_687_1
  have hr := e_687_1
  simp only [e_687_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4429)) := by
  have c43131 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43132 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43133 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43134 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43135 := (publicBatchWrapper86_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43136 := (publicBatchWrapper86_copies1348 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_2
  simp only [← c43131, ← c43132, ← c43133] at e_687_2
  have e_687_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_3
  simp only [← c43134, ← c43135, k2, ← c43136] at e_687_3
  have hr := e_687_3
  simp only [e_687_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4430)) := by
  have c43137 := (publicBatchWrapper86_copies1348 a h).2.1
  have c43138 := (publicBatchWrapper86_copies1348 a h).2.2.1
  have c43139 := (publicBatchWrapper86_copies1348 a h).2.2.2.1
  have c43140 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.1
  have c43141 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.1
  have c43142 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_4
  simp only [← c43137, ← c43138, ← c43139] at e_687_4
  have e_687_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_5
  simp only [← c43140, ← c43141, k2, ← c43142] at e_687_5
  have hr := e_687_5
  simp only [e_687_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4431)) := by
  have c43143 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.1
  have c43144 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.1
  have c43145 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.1
  have c43146 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43147 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43148 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_6
  simp only [← c43143, ← c43144, ← c43145] at e_687_6
  have e_687_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_7
  simp only [← c43146, ← c43147, k2, ← c43148] at e_687_7
  have hr := e_687_7
  simp only [e_687_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4432)) := by
  have c43149 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43150 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43151 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43152 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43153 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43154 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_8
  simp only [← c43149, ← c43150, ← c43151] at e_687_8
  have e_687_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_9
  simp only [← c43152, ← c43153, k2, ← c43154] at e_687_9
  have hr := e_687_9
  simp only [e_687_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4433)) := by
  have c43155 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43156 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43157 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43158 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43159 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43160 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_10
  simp only [← c43155, ← c43156, ← c43157] at e_687_10
  have e_687_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_11
  simp only [← c43158, ← c43159, k2, ← c43160] at e_687_11
  have hr := e_687_11
  simp only [e_687_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4434)) := by
  have c43161 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43162 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43163 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43164 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43165 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43166 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_12
  simp only [← c43161, ← c43162, ← c43163] at e_687_12
  have e_687_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_13
  simp only [← c43164, ← c43165, k2, ← c43166] at e_687_13
  have hr := e_687_13
  simp only [e_687_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4435)) := by
  have c43167 := (publicBatchWrapper86_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43168 := (publicBatchWrapper86_copies1349 a h).1
  have c43169 := (publicBatchWrapper86_copies1349 a h).2.1
  have c43170 := (publicBatchWrapper86_copies1349 a h).2.2.1
  have c43171 := (publicBatchWrapper86_copies1349 a h).2.2.2.1
  have c43172 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_14
  simp only [← c43167, ← c43168, ← c43169] at e_687_14
  have e_687_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_15
  simp only [← c43170, ← c43171, k2, ← c43172] at e_687_15
  have hr := e_687_15
  simp only [e_687_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4436)) := by
  have c43173 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.1
  have c43174 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.1
  have c43175 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.1
  have c43176 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.1
  have c43177 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.1
  have c43178 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_16
  simp only [← c43173, ← c43174, ← c43175] at e_687_16
  have e_687_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_17
  simp only [← c43176, ← c43177, k2, ← c43178] at e_687_17
  have hr := e_687_17
  simp only [e_687_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 687 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4437)) := by
  have c43179 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43180 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43181 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43182 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43183 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43184 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_687_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_18
  simp only [← c43179, ← c43180, ← c43181] at e_687_18
  have e_687_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_19
  simp only [← c43182, ← c43183, k2, ← c43184] at e_687_19
  have hr := e_687_19
  simp only [e_687_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4438)) := by
  have c43185 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43186 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43187 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43188 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43189 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43190 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_0
  simp only [← c43185, ← c43186, ← c43187] at e_688_0
  have e_688_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_1
  simp only [← c43188, ← c43189, k2, ← c43190] at e_688_1
  have hr := e_688_1
  simp only [e_688_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4439)) := by
  have c43191 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43192 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43193 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43194 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43195 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43196 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_2
  simp only [← c43191, ← c43192, ← c43193] at e_688_2
  have e_688_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_3
  simp only [← c43194, ← c43195, k2, ← c43196] at e_688_3
  have hr := e_688_3
  simp only [e_688_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4440)) := by
  have c43197 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43198 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43199 := (publicBatchWrapper86_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43200 := (publicBatchWrapper86_copies1350 a h).1
  have c43201 := (publicBatchWrapper86_copies1350 a h).2.1
  have c43202 := (publicBatchWrapper86_copies1350 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_4
  simp only [← c43197, ← c43198, ← c43199] at e_688_4
  have e_688_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_5
  simp only [← c43200, ← c43201, k2, ← c43202] at e_688_5
  have hr := e_688_5
  simp only [e_688_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4441)) := by
  have c43203 := (publicBatchWrapper86_copies1350 a h).2.2.2.1
  have c43204 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.1
  have c43205 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.1
  have c43206 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.1
  have c43207 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.1
  have c43208 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_6
  simp only [← c43203, ← c43204, ← c43205] at e_688_6
  have e_688_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_7
  simp only [← c43206, ← c43207, k2, ← c43208] at e_688_7
  have hr := e_688_7
  simp only [e_688_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4442)) := by
  have c43209 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.1
  have c43210 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43211 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43212 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43213 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43214 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_8
  simp only [← c43209, ← c43210, ← c43211] at e_688_8
  have e_688_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_9
  simp only [← c43212, ← c43213, k2, ← c43214] at e_688_9
  have hr := e_688_9
  simp only [e_688_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4443)) := by
  have c43215 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43216 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43217 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43218 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43219 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43220 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_10
  simp only [← c43215, ← c43216, ← c43217] at e_688_10
  have e_688_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_11
  simp only [← c43218, ← c43219, k2, ← c43220] at e_688_11
  have hr := e_688_11
  simp only [e_688_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4444)) := by
  have c43221 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43222 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43223 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43224 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43225 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43226 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_12
  simp only [← c43221, ← c43222, ← c43223] at e_688_12
  have e_688_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_13
  simp only [← c43224, ← c43225, k2, ← c43226] at e_688_13
  have hr := e_688_13
  simp only [e_688_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4445)) := by
  have c43227 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43228 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43229 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43230 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43231 := (publicBatchWrapper86_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43232 := (publicBatchWrapper86_copies1351 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_14
  simp only [← c43227, ← c43228, ← c43229] at e_688_14
  have e_688_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_15
  simp only [← c43230, ← c43231, k2, ← c43232] at e_688_15
  have hr := e_688_15
  simp only [e_688_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4446)) := by
  have c43233 := (publicBatchWrapper86_copies1351 a h).2.1
  have c43234 := (publicBatchWrapper86_copies1351 a h).2.2.1
  have c43235 := (publicBatchWrapper86_copies1351 a h).2.2.2.1
  have c43236 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.1
  have c43237 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.1
  have c43238 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_16
  simp only [← c43233, ← c43234, ← c43235] at e_688_16
  have e_688_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_17
  simp only [← c43236, ← c43237, k2, ← c43238] at e_688_17
  have hr := e_688_17
  simp only [e_688_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 688 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4447)) := by
  have c43239 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.1
  have c43240 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.1
  have c43241 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.1
  have c43242 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43243 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43244 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_688_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_18
  simp only [← c43239, ← c43240, ← c43241] at e_688_18
  have e_688_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_19
  simp only [← c43242, ← c43243, k2, ← c43244] at e_688_19
  have hr := e_688_19
  simp only [e_688_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4448)) := by
  have c43245 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43246 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43247 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43248 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43249 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43250 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_0
  simp only [← c43245, ← c43246, ← c43247] at e_689_0
  have e_689_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_1
  simp only [← c43248, ← c43249, k2, ← c43250] at e_689_1
  have hr := e_689_1
  simp only [e_689_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4449)) := by
  have c43251 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43252 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43253 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43254 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43255 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43256 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_2
  simp only [← c43251, ← c43252, ← c43253] at e_689_2
  have e_689_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_3
  simp only [← c43254, ← c43255, k2, ← c43256] at e_689_3
  have hr := e_689_3
  simp only [e_689_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4450)) := by
  have c43257 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43258 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43259 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43260 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43261 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43262 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_4
  simp only [← c43257, ← c43258, ← c43259] at e_689_4
  have e_689_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_5
  simp only [← c43260, ← c43261, k2, ← c43262] at e_689_5
  have hr := e_689_5
  simp only [e_689_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4451)) := by
  have c43263 := (publicBatchWrapper86_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43264 := (publicBatchWrapper86_copies1352 a h).1
  have c43265 := (publicBatchWrapper86_copies1352 a h).2.1
  have c43266 := (publicBatchWrapper86_copies1352 a h).2.2.1
  have c43267 := (publicBatchWrapper86_copies1352 a h).2.2.2.1
  have c43268 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_6
  simp only [← c43263, ← c43264, ← c43265] at e_689_6
  have e_689_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_7
  simp only [← c43266, ← c43267, k2, ← c43268] at e_689_7
  have hr := e_689_7
  simp only [e_689_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4452)) := by
  have c43269 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.1
  have c43270 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.1
  have c43271 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.1
  have c43272 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.1
  have c43273 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.1
  have c43274 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_8
  simp only [← c43269, ← c43270, ← c43271] at e_689_8
  have e_689_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_9
  simp only [← c43272, ← c43273, k2, ← c43274] at e_689_9
  have hr := e_689_9
  simp only [e_689_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4453)) := by
  have c43275 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43276 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43277 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43278 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43279 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43280 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_10
  simp only [← c43275, ← c43276, ← c43277] at e_689_10
  have e_689_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_11
  simp only [← c43278, ← c43279, k2, ← c43280] at e_689_11
  have hr := e_689_11
  simp only [e_689_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4454)) := by
  have c43281 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43282 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43283 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43284 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43285 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43286 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_12
  simp only [← c43281, ← c43282, ← c43283] at e_689_12
  have e_689_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_13
  simp only [← c43284, ← c43285, k2, ← c43286] at e_689_13
  have hr := e_689_13
  simp only [e_689_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4455)) := by
  have c43287 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43288 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43289 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43290 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43291 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43292 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_14
  simp only [← c43287, ← c43288, ← c43289] at e_689_14
  have e_689_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_15
  simp only [← c43290, ← c43291, k2, ← c43292] at e_689_15
  have hr := e_689_15
  simp only [e_689_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4456)) := by
  have c43293 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43294 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43295 := (publicBatchWrapper86_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43296 := (publicBatchWrapper86_copies1353 a h).1
  have c43297 := (publicBatchWrapper86_copies1353 a h).2.1
  have c43298 := (publicBatchWrapper86_copies1353 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_16
  simp only [← c43293, ← c43294, ← c43295] at e_689_16
  have e_689_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_17
  simp only [← c43296, ← c43297, k2, ← c43298] at e_689_17
  have hr := e_689_17
  simp only [e_689_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 689 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4457)) := by
  have c43299 := (publicBatchWrapper86_copies1353 a h).2.2.2.1
  have c43300 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.1
  have c43301 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.1
  have c43302 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.1
  have c43303 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.1
  have c43304 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_689_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_18
  simp only [← c43299, ← c43300, ← c43301] at e_689_18
  have e_689_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_19
  simp only [← c43302, ← c43303, k2, ← c43304] at e_689_19
  have hr := e_689_19
  simp only [e_689_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4458)) := by
  have c43305 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.1
  have c43306 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43307 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43308 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43309 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43310 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_0
  simp only [← c43305, ← c43306, ← c43307] at e_690_0
  have e_690_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_1
  simp only [← c43308, ← c43309, k2, ← c43310] at e_690_1
  have hr := e_690_1
  simp only [e_690_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4459)) := by
  have c43311 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43312 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43313 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43314 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43315 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43316 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_2
  simp only [← c43311, ← c43312, ← c43313] at e_690_2
  have e_690_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_3
  simp only [← c43314, ← c43315, k2, ← c43316] at e_690_3
  have hr := e_690_3
  simp only [e_690_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4460)) := by
  have c43317 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43318 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43319 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43320 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43321 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43322 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_4
  simp only [← c43317, ← c43318, ← c43319] at e_690_4
  have e_690_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_5
  simp only [← c43320, ← c43321, k2, ← c43322] at e_690_5
  have hr := e_690_5
  simp only [e_690_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4461)) := by
  have c43323 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43324 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43325 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43326 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43327 := (publicBatchWrapper86_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43328 := (publicBatchWrapper86_copies1354 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_6
  simp only [← c43323, ← c43324, ← c43325] at e_690_6
  have e_690_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_7
  simp only [← c43326, ← c43327, k2, ← c43328] at e_690_7
  have hr := e_690_7
  simp only [e_690_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4462)) := by
  have c43329 := (publicBatchWrapper86_copies1354 a h).2.1
  have c43330 := (publicBatchWrapper86_copies1354 a h).2.2.1
  have c43331 := (publicBatchWrapper86_copies1354 a h).2.2.2.1
  have c43332 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.1
  have c43333 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.1
  have c43334 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_8
  simp only [← c43329, ← c43330, ← c43331] at e_690_8
  have e_690_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_9
  simp only [← c43332, ← c43333, k2, ← c43334] at e_690_9
  have hr := e_690_9
  simp only [e_690_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4463)) := by
  have c43335 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.1
  have c43336 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.1
  have c43337 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.1
  have c43338 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43339 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43340 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_10
  simp only [← c43335, ← c43336, ← c43337] at e_690_10
  have e_690_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_11
  simp only [← c43338, ← c43339, k2, ← c43340] at e_690_11
  have hr := e_690_11
  simp only [e_690_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4464)) := by
  have c43341 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43342 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43343 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43344 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43345 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43346 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_12
  simp only [← c43341, ← c43342, ← c43343] at e_690_12
  have e_690_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_13
  simp only [← c43344, ← c43345, k2, ← c43346] at e_690_13
  have hr := e_690_13
  simp only [e_690_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4465)) := by
  have c43347 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43348 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43349 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43350 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43351 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43352 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_14
  simp only [← c43347, ← c43348, ← c43349] at e_690_14
  have e_690_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_15
  simp only [← c43350, ← c43351, k2, ← c43352] at e_690_15
  have hr := e_690_15
  simp only [e_690_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4466)) := by
  have c43353 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43354 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43355 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43356 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43357 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43358 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_16
  simp only [← c43353, ← c43354, ← c43355] at e_690_16
  have e_690_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_17
  simp only [← c43356, ← c43357, k2, ← c43358] at e_690_17
  have hr := e_690_17
  simp only [e_690_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 690 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4467)) := by
  have c43359 := (publicBatchWrapper86_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43360 := (publicBatchWrapper86_copies1355 a h).1
  have c43361 := (publicBatchWrapper86_copies1355 a h).2.1
  have c43362 := (publicBatchWrapper86_copies1355 a h).2.2.1
  have c43363 := (publicBatchWrapper86_copies1355 a h).2.2.2.1
  have c43364 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_690_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_18
  simp only [← c43359, ← c43360, ← c43361] at e_690_18
  have e_690_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_19
  simp only [← c43362, ← c43363, k2, ← c43364] at e_690_19
  have hr := e_690_19
  simp only [e_690_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4468)) := by
  have c43365 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.1
  have c43366 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.1
  have c43367 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.1
  have c43368 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.1
  have c43369 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.1
  have c43370 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_0
  simp only [← c43365, ← c43366, ← c43367] at e_691_0
  have e_691_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_1
  simp only [← c43368, ← c43369, k2, ← c43370] at e_691_1
  have hr := e_691_1
  simp only [e_691_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4469)) := by
  have c43371 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43372 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43373 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43374 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43375 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43376 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_2
  simp only [← c43371, ← c43372, ← c43373] at e_691_2
  have e_691_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_3
  simp only [← c43374, ← c43375, k2, ← c43376] at e_691_3
  have hr := e_691_3
  simp only [e_691_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4470)) := by
  have c43377 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43378 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43379 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43380 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43381 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43382 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_4
  simp only [← c43377, ← c43378, ← c43379] at e_691_4
  have e_691_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_5
  simp only [← c43380, ← c43381, k2, ← c43382] at e_691_5
  have hr := e_691_5
  simp only [e_691_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4471)) := by
  have c43383 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43384 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43385 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43386 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43387 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43388 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_6
  simp only [← c43383, ← c43384, ← c43385] at e_691_6
  have e_691_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_7
  simp only [← c43386, ← c43387, k2, ← c43388] at e_691_7
  have hr := e_691_7
  simp only [e_691_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4472)) := by
  have c43389 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43390 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43391 := (publicBatchWrapper86_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43392 := (publicBatchWrapper86_copies1356 a h).1
  have c43393 := (publicBatchWrapper86_copies1356 a h).2.1
  have c43394 := (publicBatchWrapper86_copies1356 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_8
  simp only [← c43389, ← c43390, ← c43391] at e_691_8
  have e_691_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_9
  simp only [← c43392, ← c43393, k2, ← c43394] at e_691_9
  have hr := e_691_9
  simp only [e_691_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4473)) := by
  have c43395 := (publicBatchWrapper86_copies1356 a h).2.2.2.1
  have c43396 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.1
  have c43397 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.1
  have c43398 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.1
  have c43399 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.1
  have c43400 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_10
  simp only [← c43395, ← c43396, ← c43397] at e_691_10
  have e_691_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_11
  simp only [← c43398, ← c43399, k2, ← c43400] at e_691_11
  have hr := e_691_11
  simp only [e_691_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4474)) := by
  have c43401 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.1
  have c43402 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43403 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43404 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43405 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43406 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_12
  simp only [← c43401, ← c43402, ← c43403] at e_691_12
  have e_691_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_13
  simp only [← c43404, ← c43405, k2, ← c43406] at e_691_13
  have hr := e_691_13
  simp only [e_691_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4475)) := by
  have c43407 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43408 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43409 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43410 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43411 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43412 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_14
  simp only [← c43407, ← c43408, ← c43409] at e_691_14
  have e_691_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_15
  simp only [← c43410, ← c43411, k2, ← c43412] at e_691_15
  have hr := e_691_15
  simp only [e_691_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4476)) := by
  have c43413 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43414 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43415 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43416 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43417 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43418 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_16
  simp only [← c43413, ← c43414, ← c43415] at e_691_16
  have e_691_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_17
  simp only [← c43416, ← c43417, k2, ← c43418] at e_691_17
  have hr := e_691_17
  simp only [e_691_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 691 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4477)) := by
  have c43419 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43420 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43421 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43422 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43423 := (publicBatchWrapper86_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43424 := (publicBatchWrapper86_copies1357 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_691_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_18
  simp only [← c43419, ← c43420, ← c43421] at e_691_18
  have e_691_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_19
  simp only [← c43422, ← c43423, k2, ← c43424] at e_691_19
  have hr := e_691_19
  simp only [e_691_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4478)) := by
  have c43425 := (publicBatchWrapper86_copies1357 a h).2.1
  have c43426 := (publicBatchWrapper86_copies1357 a h).2.2.1
  have c43427 := (publicBatchWrapper86_copies1357 a h).2.2.2.1
  have c43428 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.1
  have c43429 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.1
  have c43430 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_0
  simp only [← c43425, ← c43426, ← c43427] at e_692_0
  have e_692_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_1
  simp only [← c43428, ← c43429, k2, ← c43430] at e_692_1
  have hr := e_692_1
  simp only [e_692_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4479)) := by
  have c43431 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.1
  have c43432 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.1
  have c43433 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.1
  have c43434 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43435 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43436 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_2
  simp only [← c43431, ← c43432, ← c43433] at e_692_2
  have e_692_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_3
  simp only [← c43434, ← c43435, k2, ← c43436] at e_692_3
  have hr := e_692_3
  simp only [e_692_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4480)) := by
  have c43437 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43438 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43439 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43440 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43441 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43442 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_4
  simp only [← c43437, ← c43438, ← c43439] at e_692_4
  have e_692_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_5
  simp only [← c43440, ← c43441, k2, ← c43442] at e_692_5
  have hr := e_692_5
  simp only [e_692_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4481)) := by
  have c43443 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43444 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43445 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43446 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43447 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43448 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_6
  simp only [← c43443, ← c43444, ← c43445] at e_692_6
  have e_692_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_7
  simp only [← c43446, ← c43447, k2, ← c43448] at e_692_7
  have hr := e_692_7
  simp only [e_692_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4482)) := by
  have c43449 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43450 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43451 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43452 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43453 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43454 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_8
  simp only [← c43449, ← c43450, ← c43451] at e_692_8
  have e_692_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_9
  simp only [← c43452, ← c43453, k2, ← c43454] at e_692_9
  have hr := e_692_9
  simp only [e_692_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4483)) := by
  have c43455 := (publicBatchWrapper86_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43456 := (publicBatchWrapper86_copies1358 a h).1
  have c43457 := (publicBatchWrapper86_copies1358 a h).2.1
  have c43458 := (publicBatchWrapper86_copies1358 a h).2.2.1
  have c43459 := (publicBatchWrapper86_copies1358 a h).2.2.2.1
  have c43460 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_10
  simp only [← c43455, ← c43456, ← c43457] at e_692_10
  have e_692_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_11
  simp only [← c43458, ← c43459, k2, ← c43460] at e_692_11
  have hr := e_692_11
  simp only [e_692_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4484)) := by
  have c43461 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.1
  have c43462 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.1
  have c43463 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.1
  have c43464 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.1
  have c43465 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.1
  have c43466 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_12
  simp only [← c43461, ← c43462, ← c43463] at e_692_12
  have e_692_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_13
  simp only [← c43464, ← c43465, k2, ← c43466] at e_692_13
  have hr := e_692_13
  simp only [e_692_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4485)) := by
  have c43467 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43468 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43469 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43470 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43471 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43472 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_14
  simp only [← c43467, ← c43468, ← c43469] at e_692_14
  have e_692_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_15
  simp only [← c43470, ← c43471, k2, ← c43472] at e_692_15
  have hr := e_692_15
  simp only [e_692_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4486)) := by
  have c43473 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43474 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43475 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43476 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43477 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43478 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_16
  simp only [← c43473, ← c43474, ← c43475] at e_692_16
  have e_692_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_17
  simp only [← c43476, ← c43477, k2, ← c43478] at e_692_17
  have hr := e_692_17
  simp only [e_692_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 692 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4487)) := by
  have c43479 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43480 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43481 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43482 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43483 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43484 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_692_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_18
  simp only [← c43479, ← c43480, ← c43481] at e_692_18
  have e_692_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_19
  simp only [← c43482, ← c43483, k2, ← c43484] at e_692_19
  have hr := e_692_19
  simp only [e_692_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4488)) := by
  have c43485 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43486 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43487 := (publicBatchWrapper86_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43488 := (publicBatchWrapper86_copies1359 a h).1
  have c43489 := (publicBatchWrapper86_copies1359 a h).2.1
  have c43490 := (publicBatchWrapper86_copies1359 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_0
  simp only [← c43485, ← c43486, ← c43487] at e_693_0
  have e_693_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_1
  simp only [← c43488, ← c43489, k2, ← c43490] at e_693_1
  have hr := e_693_1
  simp only [e_693_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4489)) := by
  have c43491 := (publicBatchWrapper86_copies1359 a h).2.2.2.1
  have c43492 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.1
  have c43493 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.1
  have c43494 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.1
  have c43495 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.1
  have c43496 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_2
  simp only [← c43491, ← c43492, ← c43493] at e_693_2
  have e_693_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_3
  simp only [← c43494, ← c43495, k2, ← c43496] at e_693_3
  have hr := e_693_3
  simp only [e_693_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4490)) := by
  have c43497 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.1
  have c43498 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43499 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43500 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43501 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43502 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_4
  simp only [← c43497, ← c43498, ← c43499] at e_693_4
  have e_693_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_5
  simp only [← c43500, ← c43501, k2, ← c43502] at e_693_5
  have hr := e_693_5
  simp only [e_693_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4491)) := by
  have c43503 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43504 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43505 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43506 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43507 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43508 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_6
  simp only [← c43503, ← c43504, ← c43505] at e_693_6
  have e_693_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_7
  simp only [← c43506, ← c43507, k2, ← c43508] at e_693_7
  have hr := e_693_7
  simp only [e_693_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4492)) := by
  have c43509 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43510 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43511 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43512 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43513 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43514 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_8
  simp only [← c43509, ← c43510, ← c43511] at e_693_8
  have e_693_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_9
  simp only [← c43512, ← c43513, k2, ← c43514] at e_693_9
  have hr := e_693_9
  simp only [e_693_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4493)) := by
  have c43515 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43516 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43517 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43518 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43519 := (publicBatchWrapper86_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43520 := (publicBatchWrapper86_copies1360 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_10
  simp only [← c43515, ← c43516, ← c43517] at e_693_10
  have e_693_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_11
  simp only [← c43518, ← c43519, k2, ← c43520] at e_693_11
  have hr := e_693_11
  simp only [e_693_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4494)) := by
  have c43521 := (publicBatchWrapper86_copies1360 a h).2.1
  have c43522 := (publicBatchWrapper86_copies1360 a h).2.2.1
  have c43523 := (publicBatchWrapper86_copies1360 a h).2.2.2.1
  have c43524 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.1
  have c43525 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.1
  have c43526 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_12
  simp only [← c43521, ← c43522, ← c43523] at e_693_12
  have e_693_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_13
  simp only [← c43524, ← c43525, k2, ← c43526] at e_693_13
  have hr := e_693_13
  simp only [e_693_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4495)) := by
  have c43527 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.1
  have c43528 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.1
  have c43529 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.1
  have c43530 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43531 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43532 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_14
  simp only [← c43527, ← c43528, ← c43529] at e_693_14
  have e_693_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_15
  simp only [← c43530, ← c43531, k2, ← c43532] at e_693_15
  have hr := e_693_15
  simp only [e_693_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4496)) := by
  have c43533 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43534 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43535 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43536 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43537 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43538 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_16
  simp only [← c43533, ← c43534, ← c43535] at e_693_16
  have e_693_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_17
  simp only [← c43536, ← c43537, k2, ← c43538] at e_693_17
  have hr := e_693_17
  simp only [e_693_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 693 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4497)) := by
  have c43539 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43540 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43541 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43542 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43543 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43544 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_693_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_18
  simp only [← c43539, ← c43540, ← c43541] at e_693_18
  have e_693_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_19
  simp only [← c43542, ← c43543, k2, ← c43544] at e_693_19
  have hr := e_693_19
  simp only [e_693_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4498)) := by
  have c43545 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43546 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43547 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43548 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43549 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43550 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_0
  simp only [← c43545, ← c43546, ← c43547] at e_694_0
  have e_694_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_1
  simp only [← c43548, ← c43549, k2, ← c43550] at e_694_1
  have hr := e_694_1
  simp only [e_694_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4499)) := by
  have c43551 := (publicBatchWrapper86_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43552 := (publicBatchWrapper86_copies1361 a h).1
  have c43553 := (publicBatchWrapper86_copies1361 a h).2.1
  have c43554 := (publicBatchWrapper86_copies1361 a h).2.2.1
  have c43555 := (publicBatchWrapper86_copies1361 a h).2.2.2.1
  have c43556 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_2
  simp only [← c43551, ← c43552, ← c43553] at e_694_2
  have e_694_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_3
  simp only [← c43554, ← c43555, k2, ← c43556] at e_694_3
  have hr := e_694_3
  simp only [e_694_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4500)) := by
  have c43557 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.1
  have c43558 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.1
  have c43559 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.1
  have c43560 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.1
  have c43561 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.1
  have c43562 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_4
  simp only [← c43557, ← c43558, ← c43559] at e_694_4
  have e_694_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_5
  simp only [← c43560, ← c43561, k2, ← c43562] at e_694_5
  have hr := e_694_5
  simp only [e_694_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4501)) := by
  have c43563 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43564 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43565 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43566 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43567 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43568 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_6
  simp only [← c43563, ← c43564, ← c43565] at e_694_6
  have e_694_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_7
  simp only [← c43566, ← c43567, k2, ← c43568] at e_694_7
  have hr := e_694_7
  simp only [e_694_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4502)) := by
  have c43569 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43570 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43571 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43572 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43573 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43574 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_8
  simp only [← c43569, ← c43570, ← c43571] at e_694_8
  have e_694_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_9
  simp only [← c43572, ← c43573, k2, ← c43574] at e_694_9
  have hr := e_694_9
  simp only [e_694_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4503)) := by
  have c43575 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43576 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43577 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43578 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43579 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43580 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_10
  simp only [← c43575, ← c43576, ← c43577] at e_694_10
  have e_694_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_11
  simp only [← c43578, ← c43579, k2, ← c43580] at e_694_11
  have hr := e_694_11
  simp only [e_694_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4504)) := by
  have c43581 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43582 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43583 := (publicBatchWrapper86_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43584 := (publicBatchWrapper86_copies1362 a h).1
  have c43585 := (publicBatchWrapper86_copies1362 a h).2.1
  have c43586 := (publicBatchWrapper86_copies1362 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_12
  simp only [← c43581, ← c43582, ← c43583] at e_694_12
  have e_694_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_13
  simp only [← c43584, ← c43585, k2, ← c43586] at e_694_13
  have hr := e_694_13
  simp only [e_694_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4505)) := by
  have c43587 := (publicBatchWrapper86_copies1362 a h).2.2.2.1
  have c43588 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.1
  have c43589 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.1
  have c43590 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.1
  have c43591 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.1
  have c43592 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_14
  simp only [← c43587, ← c43588, ← c43589] at e_694_14
  have e_694_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_15
  simp only [← c43590, ← c43591, k2, ← c43592] at e_694_15
  have hr := e_694_15
  simp only [e_694_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4506)) := by
  have c43593 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.1
  have c43594 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43595 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43596 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43597 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43598 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_16
  simp only [← c43593, ← c43594, ← c43595] at e_694_16
  have e_694_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_17
  simp only [← c43596, ← c43597, k2, ← c43598] at e_694_17
  have hr := e_694_17
  simp only [e_694_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 694 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4507)) := by
  have c43599 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43600 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43601 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43602 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43603 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43604 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_694_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_18
  simp only [← c43599, ← c43600, ← c43601] at e_694_18
  have e_694_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_19
  simp only [← c43602, ← c43603, k2, ← c43604] at e_694_19
  have hr := e_694_19
  simp only [e_694_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4508)) := by
  have c43605 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43606 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43607 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43608 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43609 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43610 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_0
  simp only [← c43605, ← c43606, ← c43607] at e_695_0
  have e_695_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_1
  simp only [← c43608, ← c43609, k2, ← c43610] at e_695_1
  have hr := e_695_1
  simp only [e_695_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4509)) := by
  have c43611 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43612 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43613 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43614 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43615 := (publicBatchWrapper86_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43616 := (publicBatchWrapper86_copies1363 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_2
  simp only [← c43611, ← c43612, ← c43613] at e_695_2
  have e_695_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_3
  simp only [← c43614, ← c43615, k2, ← c43616] at e_695_3
  have hr := e_695_3
  simp only [e_695_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4510)) := by
  have c43617 := (publicBatchWrapper86_copies1363 a h).2.1
  have c43618 := (publicBatchWrapper86_copies1363 a h).2.2.1
  have c43619 := (publicBatchWrapper86_copies1363 a h).2.2.2.1
  have c43620 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.1
  have c43621 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.1
  have c43622 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_4
  simp only [← c43617, ← c43618, ← c43619] at e_695_4
  have e_695_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_5
  simp only [← c43620, ← c43621, k2, ← c43622] at e_695_5
  have hr := e_695_5
  simp only [e_695_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4511)) := by
  have c43623 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.1
  have c43624 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.1
  have c43625 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.1
  have c43626 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43627 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43628 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_6
  simp only [← c43623, ← c43624, ← c43625] at e_695_6
  have e_695_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_7
  simp only [← c43626, ← c43627, k2, ← c43628] at e_695_7
  have hr := e_695_7
  simp only [e_695_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4512)) := by
  have c43629 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43630 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43631 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43632 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43633 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43634 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_8
  simp only [← c43629, ← c43630, ← c43631] at e_695_8
  have e_695_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_9
  simp only [← c43632, ← c43633, k2, ← c43634] at e_695_9
  have hr := e_695_9
  simp only [e_695_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4513)) := by
  have c43635 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43636 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43637 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43638 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43639 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43640 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_10
  simp only [← c43635, ← c43636, ← c43637] at e_695_10
  have e_695_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_11
  simp only [← c43638, ← c43639, k2, ← c43640] at e_695_11
  have hr := e_695_11
  simp only [e_695_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4514)) := by
  have c43641 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43642 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43643 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43644 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43645 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43646 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_12
  simp only [← c43641, ← c43642, ← c43643] at e_695_12
  have e_695_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_13
  simp only [← c43644, ← c43645, k2, ← c43646] at e_695_13
  have hr := e_695_13
  simp only [e_695_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4515)) := by
  have c43647 := (publicBatchWrapper86_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43648 := (publicBatchWrapper86_copies1364 a h).1
  have c43649 := (publicBatchWrapper86_copies1364 a h).2.1
  have c43650 := (publicBatchWrapper86_copies1364 a h).2.2.1
  have c43651 := (publicBatchWrapper86_copies1364 a h).2.2.2.1
  have c43652 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_14
  simp only [← c43647, ← c43648, ← c43649] at e_695_14
  have e_695_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_15
  simp only [← c43650, ← c43651, k2, ← c43652] at e_695_15
  have hr := e_695_15
  simp only [e_695_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4516)) := by
  have c43653 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.1
  have c43654 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.1
  have c43655 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.1
  have c43656 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.1
  have c43657 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.1
  have c43658 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_16
  simp only [← c43653, ← c43654, ← c43655] at e_695_16
  have e_695_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_17
  simp only [← c43656, ← c43657, k2, ← c43658] at e_695_17
  have hr := e_695_17
  simp only [e_695_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 695 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4517)) := by
  have c43659 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43660 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43661 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43662 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43663 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43664 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_695_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_18
  simp only [← c43659, ← c43660, ← c43661] at e_695_18
  have e_695_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_19
  simp only [← c43662, ← c43663, k2, ← c43664] at e_695_19
  have hr := e_695_19
  simp only [e_695_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4518)) := by
  have c43665 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43666 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43667 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43668 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43669 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43670 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_0
  simp only [← c43665, ← c43666, ← c43667] at e_696_0
  have e_696_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_1
  simp only [← c43668, ← c43669, k2, ← c43670] at e_696_1
  have hr := e_696_1
  simp only [e_696_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4519)) := by
  have c43671 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43672 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43673 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43674 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43675 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43676 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_2
  simp only [← c43671, ← c43672, ← c43673] at e_696_2
  have e_696_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_3
  simp only [← c43674, ← c43675, k2, ← c43676] at e_696_3
  have hr := e_696_3
  simp only [e_696_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4520)) := by
  have c43677 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43678 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43679 := (publicBatchWrapper86_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43680 := (publicBatchWrapper86_copies1365 a h).1
  have c43681 := (publicBatchWrapper86_copies1365 a h).2.1
  have c43682 := (publicBatchWrapper86_copies1365 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_4
  simp only [← c43677, ← c43678, ← c43679] at e_696_4
  have e_696_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_5
  simp only [← c43680, ← c43681, k2, ← c43682] at e_696_5
  have hr := e_696_5
  simp only [e_696_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4521)) := by
  have c43683 := (publicBatchWrapper86_copies1365 a h).2.2.2.1
  have c43684 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.1
  have c43685 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.1
  have c43686 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.1
  have c43687 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.1
  have c43688 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_6
  simp only [← c43683, ← c43684, ← c43685] at e_696_6
  have e_696_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_7
  simp only [← c43686, ← c43687, k2, ← c43688] at e_696_7
  have hr := e_696_7
  simp only [e_696_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4522)) := by
  have c43689 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.1
  have c43690 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43691 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43692 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43693 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43694 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_8
  simp only [← c43689, ← c43690, ← c43691] at e_696_8
  have e_696_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_9
  simp only [← c43692, ← c43693, k2, ← c43694] at e_696_9
  have hr := e_696_9
  simp only [e_696_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4523)) := by
  have c43695 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43696 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43697 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43698 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43699 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43700 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_10
  simp only [← c43695, ← c43696, ← c43697] at e_696_10
  have e_696_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_11
  simp only [← c43698, ← c43699, k2, ← c43700] at e_696_11
  have hr := e_696_11
  simp only [e_696_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4524)) := by
  have c43701 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43702 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43703 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43704 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43705 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43706 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_12
  simp only [← c43701, ← c43702, ← c43703] at e_696_12
  have e_696_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_13
  simp only [← c43704, ← c43705, k2, ← c43706] at e_696_13
  have hr := e_696_13
  simp only [e_696_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4525)) := by
  have c43707 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43708 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43709 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43710 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43711 := (publicBatchWrapper86_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43712 := (publicBatchWrapper86_copies1366 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_14
  simp only [← c43707, ← c43708, ← c43709] at e_696_14
  have e_696_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_15
  simp only [← c43710, ← c43711, k2, ← c43712] at e_696_15
  have hr := e_696_15
  simp only [e_696_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4526)) := by
  have c43713 := (publicBatchWrapper86_copies1366 a h).2.1
  have c43714 := (publicBatchWrapper86_copies1366 a h).2.2.1
  have c43715 := (publicBatchWrapper86_copies1366 a h).2.2.2.1
  have c43716 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.1
  have c43717 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.1
  have c43718 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_16
  simp only [← c43713, ← c43714, ← c43715] at e_696_16
  have e_696_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_17
  simp only [← c43716, ← c43717, k2, ← c43718] at e_696_17
  have hr := e_696_17
  simp only [e_696_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 696 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4527)) := by
  have c43719 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.1
  have c43720 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.1
  have c43721 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.1
  have c43722 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43723 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43724 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_696_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_18
  simp only [← c43719, ← c43720, ← c43721] at e_696_18
  have e_696_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_19
  simp only [← c43722, ← c43723, k2, ← c43724] at e_696_19
  have hr := e_696_19
  simp only [e_696_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4528)) := by
  have c43725 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43726 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43727 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43728 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43729 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43730 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_0
  simp only [← c43725, ← c43726, ← c43727] at e_697_0
  have e_697_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_1
  simp only [← c43728, ← c43729, k2, ← c43730] at e_697_1
  have hr := e_697_1
  simp only [e_697_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4529)) := by
  have c43731 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43732 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43733 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43734 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43735 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43736 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_2
  simp only [← c43731, ← c43732, ← c43733] at e_697_2
  have e_697_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_3
  simp only [← c43734, ← c43735, k2, ← c43736] at e_697_3
  have hr := e_697_3
  simp only [e_697_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4530)) := by
  have c43737 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43738 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43739 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43740 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43741 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43742 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_4
  simp only [← c43737, ← c43738, ← c43739] at e_697_4
  have e_697_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_5
  simp only [← c43740, ← c43741, k2, ← c43742] at e_697_5
  have hr := e_697_5
  simp only [e_697_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4531)) := by
  have c43743 := (publicBatchWrapper86_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43744 := (publicBatchWrapper86_copies1367 a h).1
  have c43745 := (publicBatchWrapper86_copies1367 a h).2.1
  have c43746 := (publicBatchWrapper86_copies1367 a h).2.2.1
  have c43747 := (publicBatchWrapper86_copies1367 a h).2.2.2.1
  have c43748 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_6
  simp only [← c43743, ← c43744, ← c43745] at e_697_6
  have e_697_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_7
  simp only [← c43746, ← c43747, k2, ← c43748] at e_697_7
  have hr := e_697_7
  simp only [e_697_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4532)) := by
  have c43749 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.1
  have c43750 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.1
  have c43751 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.1
  have c43752 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.1
  have c43753 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.1
  have c43754 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_8
  simp only [← c43749, ← c43750, ← c43751] at e_697_8
  have e_697_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_9
  simp only [← c43752, ← c43753, k2, ← c43754] at e_697_9
  have hr := e_697_9
  simp only [e_697_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4533)) := by
  have c43755 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43756 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43757 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43758 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43759 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43760 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_10
  simp only [← c43755, ← c43756, ← c43757] at e_697_10
  have e_697_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_11
  simp only [← c43758, ← c43759, k2, ← c43760] at e_697_11
  have hr := e_697_11
  simp only [e_697_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4534)) := by
  have c43761 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43762 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43763 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43764 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43765 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43766 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_12
  simp only [← c43761, ← c43762, ← c43763] at e_697_12
  have e_697_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_13
  simp only [← c43764, ← c43765, k2, ← c43766] at e_697_13
  have hr := e_697_13
  simp only [e_697_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4535)) := by
  have c43767 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43768 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43769 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43770 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43771 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43772 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_14
  simp only [← c43767, ← c43768, ← c43769] at e_697_14
  have e_697_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_15
  simp only [← c43770, ← c43771, k2, ← c43772] at e_697_15
  have hr := e_697_15
  simp only [e_697_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4536)) := by
  have c43773 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43774 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43775 := (publicBatchWrapper86_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43776 := (publicBatchWrapper86_copies1368 a h).1
  have c43777 := (publicBatchWrapper86_copies1368 a h).2.1
  have c43778 := (publicBatchWrapper86_copies1368 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_16
  simp only [← c43773, ← c43774, ← c43775] at e_697_16
  have e_697_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_17
  simp only [← c43776, ← c43777, k2, ← c43778] at e_697_17
  have hr := e_697_17
  simp only [e_697_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 697 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4537)) := by
  have c43779 := (publicBatchWrapper86_copies1368 a h).2.2.2.1
  have c43780 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.1
  have c43781 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.1
  have c43782 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.1
  have c43783 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.1
  have c43784 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_697_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_18
  simp only [← c43779, ← c43780, ← c43781] at e_697_18
  have e_697_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_19
  simp only [← c43782, ← c43783, k2, ← c43784] at e_697_19
  have hr := e_697_19
  simp only [e_697_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4538)) := by
  have c43785 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.1
  have c43786 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43787 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43788 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43789 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43790 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_0
  simp only [← c43785, ← c43786, ← c43787] at e_698_0
  have e_698_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_1
  simp only [← c43788, ← c43789, k2, ← c43790] at e_698_1
  have hr := e_698_1
  simp only [e_698_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4539)) := by
  have c43791 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43792 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43793 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43794 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43795 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43796 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_2
  simp only [← c43791, ← c43792, ← c43793] at e_698_2
  have e_698_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_3
  simp only [← c43794, ← c43795, k2, ← c43796] at e_698_3
  have hr := e_698_3
  simp only [e_698_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4540)) := by
  have c43797 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43798 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43799 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43800 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43801 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43802 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_4
  simp only [← c43797, ← c43798, ← c43799] at e_698_4
  have e_698_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_5
  simp only [← c43800, ← c43801, k2, ← c43802] at e_698_5
  have hr := e_698_5
  simp only [e_698_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4541)) := by
  have c43803 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43804 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43805 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43806 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43807 := (publicBatchWrapper86_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43808 := (publicBatchWrapper86_copies1369 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_6
  simp only [← c43803, ← c43804, ← c43805] at e_698_6
  have e_698_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_7
  simp only [← c43806, ← c43807, k2, ← c43808] at e_698_7
  have hr := e_698_7
  simp only [e_698_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4542)) := by
  have c43809 := (publicBatchWrapper86_copies1369 a h).2.1
  have c43810 := (publicBatchWrapper86_copies1369 a h).2.2.1
  have c43811 := (publicBatchWrapper86_copies1369 a h).2.2.2.1
  have c43812 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.1
  have c43813 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.1
  have c43814 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_8
  simp only [← c43809, ← c43810, ← c43811] at e_698_8
  have e_698_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_9
  simp only [← c43812, ← c43813, k2, ← c43814] at e_698_9
  have hr := e_698_9
  simp only [e_698_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4543)) := by
  have c43815 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.1
  have c43816 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.1
  have c43817 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.1
  have c43818 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43819 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43820 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_10
  simp only [← c43815, ← c43816, ← c43817] at e_698_10
  have e_698_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_11
  simp only [← c43818, ← c43819, k2, ← c43820] at e_698_11
  have hr := e_698_11
  simp only [e_698_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4544)) := by
  have c43821 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43822 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43823 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43824 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43825 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43826 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_12
  simp only [← c43821, ← c43822, ← c43823] at e_698_12
  have e_698_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_13
  simp only [← c43824, ← c43825, k2, ← c43826] at e_698_13
  have hr := e_698_13
  simp only [e_698_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4545)) := by
  have c43827 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43828 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43829 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43830 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43831 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43832 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_14
  simp only [← c43827, ← c43828, ← c43829] at e_698_14
  have e_698_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_15
  simp only [← c43830, ← c43831, k2, ← c43832] at e_698_15
  have hr := e_698_15
  simp only [e_698_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4546)) := by
  have c43833 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43834 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43835 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43836 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43837 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43838 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_16
  simp only [← c43833, ← c43834, ← c43835] at e_698_16
  have e_698_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_17
  simp only [← c43836, ← c43837, k2, ← c43838] at e_698_17
  have hr := e_698_17
  simp only [e_698_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 698 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4547)) := by
  have c43839 := (publicBatchWrapper86_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43840 := (publicBatchWrapper86_copies1370 a h).1
  have c43841 := (publicBatchWrapper86_copies1370 a h).2.1
  have c43842 := (publicBatchWrapper86_copies1370 a h).2.2.1
  have c43843 := (publicBatchWrapper86_copies1370 a h).2.2.2.1
  have c43844 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_698_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_18
  simp only [← c43839, ← c43840, ← c43841] at e_698_18
  have e_698_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_19
  simp only [← c43842, ← c43843, k2, ← c43844] at e_698_19
  have hr := e_698_19
  simp only [e_698_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4548)) := by
  have c43845 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.1
  have c43846 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.1
  have c43847 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.1
  have c43848 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.1
  have c43849 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.1
  have c43850 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_0
  simp only [← c43845, ← c43846, ← c43847] at e_699_0
  have e_699_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_1
  simp only [← c43848, ← c43849, k2, ← c43850] at e_699_1
  have hr := e_699_1
  simp only [e_699_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4549)) := by
  have c43851 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43852 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43853 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43854 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43855 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43856 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_2
  simp only [← c43851, ← c43852, ← c43853] at e_699_2
  have e_699_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_3
  simp only [← c43854, ← c43855, k2, ← c43856] at e_699_3
  have hr := e_699_3
  simp only [e_699_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4550)) := by
  have c43857 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43858 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43859 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43860 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43861 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43862 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_4
  simp only [← c43857, ← c43858, ← c43859] at e_699_4
  have e_699_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_5
  simp only [← c43860, ← c43861, k2, ← c43862] at e_699_5
  have hr := e_699_5
  simp only [e_699_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4551)) := by
  have c43863 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43864 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43865 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43866 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43867 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43868 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_6
  simp only [← c43863, ← c43864, ← c43865] at e_699_6
  have e_699_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_7
  simp only [← c43866, ← c43867, k2, ← c43868] at e_699_7
  have hr := e_699_7
  simp only [e_699_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4552)) := by
  have c43869 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43870 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43871 := (publicBatchWrapper86_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43872 := (publicBatchWrapper86_copies1371 a h).1
  have c43873 := (publicBatchWrapper86_copies1371 a h).2.1
  have c43874 := (publicBatchWrapper86_copies1371 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_8
  simp only [← c43869, ← c43870, ← c43871] at e_699_8
  have e_699_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_9
  simp only [← c43872, ← c43873, k2, ← c43874] at e_699_9
  have hr := e_699_9
  simp only [e_699_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4553)) := by
  have c43875 := (publicBatchWrapper86_copies1371 a h).2.2.2.1
  have c43876 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.1
  have c43877 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.1
  have c43878 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.1
  have c43879 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.1
  have c43880 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_10
  simp only [← c43875, ← c43876, ← c43877] at e_699_10
  have e_699_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_11
  simp only [← c43878, ← c43879, k2, ← c43880] at e_699_11
  have hr := e_699_11
  simp only [e_699_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4554)) := by
  have c43881 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.1
  have c43882 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43883 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43884 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43885 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43886 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_12
  simp only [← c43881, ← c43882, ← c43883] at e_699_12
  have e_699_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_13
  simp only [← c43884, ← c43885, k2, ← c43886] at e_699_13
  have hr := e_699_13
  simp only [e_699_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4555)) := by
  have c43887 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43888 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43889 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43890 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43891 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43892 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_14
  simp only [← c43887, ← c43888, ← c43889] at e_699_14
  have e_699_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_15
  simp only [← c43890, ← c43891, k2, ← c43892] at e_699_15
  have hr := e_699_15
  simp only [e_699_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4556)) := by
  have c43893 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43894 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43895 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43896 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43897 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43898 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_16
  simp only [← c43893, ← c43894, ← c43895] at e_699_16
  have e_699_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_17
  simp only [← c43896, ← c43897, k2, ← c43898] at e_699_17
  have hr := e_699_17
  simp only [e_699_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 699 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4557)) := by
  have c43899 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43900 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43901 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43902 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43903 := (publicBatchWrapper86_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43904 := (publicBatchWrapper86_copies1372 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_699_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_18
  simp only [← c43899, ← c43900, ← c43901] at e_699_18
  have e_699_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_19
  simp only [← c43902, ← c43903, k2, ← c43904] at e_699_19
  have hr := e_699_19
  simp only [e_699_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4558)) := by
  have c43905 := (publicBatchWrapper86_copies1372 a h).2.1
  have c43906 := (publicBatchWrapper86_copies1372 a h).2.2.1
  have c43907 := (publicBatchWrapper86_copies1372 a h).2.2.2.1
  have c43908 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.1
  have c43909 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.1
  have c43910 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_0
  simp only [← c43905, ← c43906, ← c43907] at e_700_0
  have e_700_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_1
  simp only [← c43908, ← c43909, k2, ← c43910] at e_700_1
  have hr := e_700_1
  simp only [e_700_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4559)) := by
  have c43911 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.1
  have c43912 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.1
  have c43913 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.1
  have c43914 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43915 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43916 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_2
  simp only [← c43911, ← c43912, ← c43913] at e_700_2
  have e_700_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_3
  simp only [← c43914, ← c43915, k2, ← c43916] at e_700_3
  have hr := e_700_3
  simp only [e_700_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 23) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4560)) := by
  have c43917 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43918 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43919 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43920 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43921 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43922 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_4
  simp only [← c43917, ← c43918, ← c43919] at e_700_4
  have e_700_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_5
  simp only [← c43920, ← c43921, k2, ← c43922] at e_700_5
  have hr := e_700_5
  simp only [e_700_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 31) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4561)) := by
  have c43923 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43924 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43925 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43926 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43927 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43928 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_6
  simp only [← c43923, ← c43924, ← c43925] at e_700_6
  have e_700_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_7
  simp only [← c43926, ← c43927, k2, ← c43928] at e_700_7
  have hr := e_700_7
  simp only [e_700_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 39) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4562)) := by
  have c43929 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43930 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43931 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43932 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43933 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43934 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_8
  simp only [← c43929, ← c43930, ← c43931] at e_700_8
  have e_700_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_9
  simp only [← c43932, ← c43933, k2, ← c43934] at e_700_9
  have hr := e_700_9
  simp only [e_700_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 47) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4563)) := by
  have c43935 := (publicBatchWrapper86_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43936 := (publicBatchWrapper86_copies1373 a h).1
  have c43937 := (publicBatchWrapper86_copies1373 a h).2.1
  have c43938 := (publicBatchWrapper86_copies1373 a h).2.2.1
  have c43939 := (publicBatchWrapper86_copies1373 a h).2.2.2.1
  have c43940 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_10
  simp only [← c43935, ← c43936, ← c43937] at e_700_10
  have e_700_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_11
  simp only [← c43938, ← c43939, k2, ← c43940] at e_700_11
  have hr := e_700_11
  simp only [e_700_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 55) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4564)) := by
  have c43941 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.1
  have c43942 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.1
  have c43943 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.1
  have c43944 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.1
  have c43945 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.1
  have c43946 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_12
  simp only [← c43941, ← c43942, ← c43943] at e_700_12
  have e_700_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_13
  simp only [← c43944, ← c43945, k2, ← c43946] at e_700_13
  have hr := e_700_13
  simp only [e_700_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 63) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4565)) := by
  have c43947 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43948 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43949 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43950 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43951 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43952 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_14
  simp only [← c43947, ← c43948, ← c43949] at e_700_14
  have e_700_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_15
  simp only [← c43950, ← c43951, k2, ← c43952] at e_700_15
  have hr := e_700_15
  simp only [e_700_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 71) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4566)) := by
  have c43953 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43954 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43955 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43956 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43957 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43958 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_16
  simp only [← c43953, ← c43954, ← c43955] at e_700_16
  have e_700_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_17
  simp only [← c43956, ← c43957, k2, ← c43958] at e_700_17
  have hr := e_700_17
  simp only [e_700_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 700 79) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4567)) := by
  have c43959 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43960 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43961 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43962 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43963 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43964 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_700_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_18
  simp only [← c43959, ← c43960, ← c43961] at e_700_18
  have e_700_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_19
  simp only [← c43962, ← c43963, k2, ← c43964] at e_700_19
  have hr := e_700_19
  simp only [e_700_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 7) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4568)) := by
  have c43965 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43966 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43967 := (publicBatchWrapper86_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43968 := (publicBatchWrapper86_copies1374 a h).1
  have c43969 := (publicBatchWrapper86_copies1374 a h).2.1
  have c43970 := (publicBatchWrapper86_copies1374 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_0
  simp only [← c43965, ← c43966, ← c43967] at e_701_0
  have e_701_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_1
  simp only [← c43968, ← c43969, k2, ← c43970] at e_701_1
  have hr := e_701_1
  simp only [e_701_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 15) = bselect (a (.wire 23 35)) (a (.virt 200)) (a (.virt 4569)) := by
  have c43971 := (publicBatchWrapper86_copies1374 a h).2.2.2.1
  have c43972 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.1
  have c43973 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.1
  have c43974 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.1
  have c43975 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.1
  have c43976 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_2
  simp only [← c43971, ← c43972, ← c43973] at e_701_2
  have e_701_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_3
  simp only [← c43974, ← c43975, k2, ← c43976] at e_701_3
  have hr := e_701_3
  simp only [e_701_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4610)) := by
  have c43977 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.1
  have c43978 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43979 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43980 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43981 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43982 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_4
  simp only [← c43977, ← c43978, ← c43979] at e_701_4
  have e_701_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_5
  simp only [← c43980, ← c43981, k2, ← c43982] at e_701_5
  have hr := e_701_5
  simp only [e_701_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4611)) := by
  have c43983 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43984 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43985 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43986 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43987 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43988 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_6
  simp only [← c43983, ← c43984, ← c43985] at e_701_6
  have e_701_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_7
  simp only [← c43986, ← c43987, k2, ← c43988] at e_701_7
  have hr := e_701_7
  simp only [e_701_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4612)) := by
  have c43989 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43990 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43991 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43992 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43993 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43994 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_8
  simp only [← c43989, ← c43990, ← c43991] at e_701_8
  have e_701_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_9
  simp only [← c43992, ← c43993, k2, ← c43994] at e_701_9
  have hr := e_701_9
  simp only [e_701_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4613)) := by
  have c43995 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43996 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43997 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43998 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43999 := (publicBatchWrapper86_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44000 := (publicBatchWrapper86_copies1375 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_10
  simp only [← c43995, ← c43996, ← c43997] at e_701_10
  have e_701_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_11
  simp only [← c43998, ← c43999, k2, ← c44000] at e_701_11
  have hr := e_701_11
  simp only [e_701_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4614)) := by
  have c44001 := (publicBatchWrapper86_copies1375 a h).2.1
  have c44002 := (publicBatchWrapper86_copies1375 a h).2.2.1
  have c44003 := (publicBatchWrapper86_copies1375 a h).2.2.2.1
  have c44004 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.1
  have c44005 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.1
  have c44006 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_12
  simp only [← c44001, ← c44002, ← c44003] at e_701_12
  have e_701_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_13
  simp only [← c44004, ← c44005, k2, ← c44006] at e_701_13
  have hr := e_701_13
  simp only [e_701_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4615)) := by
  have c44007 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.1
  have c44008 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.1
  have c44009 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.1
  have c44010 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44011 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44012 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_14
  simp only [← c44007, ← c44008, ← c44009] at e_701_14
  have e_701_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_15
  simp only [← c44010, ← c44011, k2, ← c44012] at e_701_15
  have hr := e_701_15
  simp only [e_701_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4616)) := by
  have c44013 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44014 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44015 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44016 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44017 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44018 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_16
  simp only [← c44013, ← c44014, ← c44015] at e_701_16
  have e_701_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_17
  simp only [← c44016, ← c44017, k2, ← c44018] at e_701_17
  have hr := e_701_17
  simp only [e_701_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 701 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4617)) := by
  have c44019 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44020 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44021 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44022 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44023 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44024 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_701_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_18
  simp only [← c44019, ← c44020, ← c44021] at e_701_18
  have e_701_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_19
  simp only [← c44022, ← c44023, k2, ← c44024] at e_701_19
  have hr := e_701_19
  simp only [e_701_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4618)) := by
  have c44025 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44026 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44027 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44028 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44029 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44030 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_0
  simp only [← c44025, ← c44026, ← c44027] at e_702_0
  have e_702_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_1
  simp only [← c44028, ← c44029, k2, ← c44030] at e_702_1
  have hr := e_702_1
  simp only [e_702_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4619)) := by
  have c44031 := (publicBatchWrapper86_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44032 := (publicBatchWrapper86_copies1376 a h).1
  have c44033 := (publicBatchWrapper86_copies1376 a h).2.1
  have c44034 := (publicBatchWrapper86_copies1376 a h).2.2.1
  have c44035 := (publicBatchWrapper86_copies1376 a h).2.2.2.1
  have c44036 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_2
  simp only [← c44031, ← c44032, ← c44033] at e_702_2
  have e_702_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_3
  simp only [← c44034, ← c44035, k2, ← c44036] at e_702_3
  have hr := e_702_3
  simp only [e_702_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4620)) := by
  have c44037 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.1
  have c44038 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.1
  have c44039 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.1
  have c44040 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.1
  have c44041 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.1
  have c44042 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_4
  simp only [← c44037, ← c44038, ← c44039] at e_702_4
  have e_702_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_5
  simp only [← c44040, ← c44041, k2, ← c44042] at e_702_5
  have hr := e_702_5
  simp only [e_702_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4621)) := by
  have c44043 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44044 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44045 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44046 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44047 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44048 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_6
  simp only [← c44043, ← c44044, ← c44045] at e_702_6
  have e_702_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_7
  simp only [← c44046, ← c44047, k2, ← c44048] at e_702_7
  have hr := e_702_7
  simp only [e_702_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4622)) := by
  have c44049 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44050 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44051 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44052 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44053 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44054 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_8
  simp only [← c44049, ← c44050, ← c44051] at e_702_8
  have e_702_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_9
  simp only [← c44052, ← c44053, k2, ← c44054] at e_702_9
  have hr := e_702_9
  simp only [e_702_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4623)) := by
  have c44055 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44056 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44057 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44058 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44059 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44060 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_10
  simp only [← c44055, ← c44056, ← c44057] at e_702_10
  have e_702_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_11
  simp only [← c44058, ← c44059, k2, ← c44060] at e_702_11
  have hr := e_702_11
  simp only [e_702_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4624)) := by
  have c44061 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44062 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44063 := (publicBatchWrapper86_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44064 := (publicBatchWrapper86_copies1377 a h).1
  have c44065 := (publicBatchWrapper86_copies1377 a h).2.1
  have c44066 := (publicBatchWrapper86_copies1377 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_12
  simp only [← c44061, ← c44062, ← c44063] at e_702_12
  have e_702_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_13
  simp only [← c44064, ← c44065, k2, ← c44066] at e_702_13
  have hr := e_702_13
  simp only [e_702_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4625)) := by
  have c44067 := (publicBatchWrapper86_copies1377 a h).2.2.2.1
  have c44068 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.1
  have c44069 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.1
  have c44070 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.1
  have c44071 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.1
  have c44072 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_14
  simp only [← c44067, ← c44068, ← c44069] at e_702_14
  have e_702_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_15
  simp only [← c44070, ← c44071, k2, ← c44072] at e_702_15
  have hr := e_702_15
  simp only [e_702_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4626)) := by
  have c44073 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.1
  have c44074 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44075 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44076 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44077 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44078 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_16
  simp only [← c44073, ← c44074, ← c44075] at e_702_16
  have e_702_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_17
  simp only [← c44076, ← c44077, k2, ← c44078] at e_702_17
  have hr := e_702_17
  simp only [e_702_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 702 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4627)) := by
  have c44079 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44080 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44081 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44082 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44083 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44084 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_702_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_18
  simp only [← c44079, ← c44080, ← c44081] at e_702_18
  have e_702_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_19
  simp only [← c44082, ← c44083, k2, ← c44084] at e_702_19
  have hr := e_702_19
  simp only [e_702_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4628)) := by
  have c44085 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44086 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44087 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44088 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44089 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44090 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_0
  simp only [← c44085, ← c44086, ← c44087] at e_703_0
  have e_703_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_1
  simp only [← c44088, ← c44089, k2, ← c44090] at e_703_1
  have hr := e_703_1
  simp only [e_703_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4629)) := by
  have c44091 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44092 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44093 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44094 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44095 := (publicBatchWrapper86_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44096 := (publicBatchWrapper86_copies1378 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_2
  simp only [← c44091, ← c44092, ← c44093] at e_703_2
  have e_703_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_3
  simp only [← c44094, ← c44095, k2, ← c44096] at e_703_3
  have hr := e_703_3
  simp only [e_703_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4630)) := by
  have c44097 := (publicBatchWrapper86_copies1378 a h).2.1
  have c44098 := (publicBatchWrapper86_copies1378 a h).2.2.1
  have c44099 := (publicBatchWrapper86_copies1378 a h).2.2.2.1
  have c44100 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.1
  have c44101 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.1
  have c44102 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_4
  simp only [← c44097, ← c44098, ← c44099] at e_703_4
  have e_703_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_5
  simp only [← c44100, ← c44101, k2, ← c44102] at e_703_5
  have hr := e_703_5
  simp only [e_703_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4631)) := by
  have c44103 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.1
  have c44104 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.1
  have c44105 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.1
  have c44106 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44107 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44108 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_6
  simp only [← c44103, ← c44104, ← c44105] at e_703_6
  have e_703_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_7
  simp only [← c44106, ← c44107, k2, ← c44108] at e_703_7
  have hr := e_703_7
  simp only [e_703_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4632)) := by
  have c44109 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44110 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44111 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44112 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44113 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44114 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_8
  simp only [← c44109, ← c44110, ← c44111] at e_703_8
  have e_703_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_9
  simp only [← c44112, ← c44113, k2, ← c44114] at e_703_9
  have hr := e_703_9
  simp only [e_703_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4633)) := by
  have c44115 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44116 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44117 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44118 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44119 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44120 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_10
  simp only [← c44115, ← c44116, ← c44117] at e_703_10
  have e_703_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_11
  simp only [← c44118, ← c44119, k2, ← c44120] at e_703_11
  have hr := e_703_11
  simp only [e_703_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4634)) := by
  have c44121 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44122 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44123 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44124 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44125 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44126 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_12
  simp only [← c44121, ← c44122, ← c44123] at e_703_12
  have e_703_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_13
  simp only [← c44124, ← c44125, k2, ← c44126] at e_703_13
  have hr := e_703_13
  simp only [e_703_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4635)) := by
  have c44127 := (publicBatchWrapper86_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44128 := (publicBatchWrapper86_copies1379 a h).1
  have c44129 := (publicBatchWrapper86_copies1379 a h).2.1
  have c44130 := (publicBatchWrapper86_copies1379 a h).2.2.1
  have c44131 := (publicBatchWrapper86_copies1379 a h).2.2.2.1
  have c44132 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_14
  simp only [← c44127, ← c44128, ← c44129] at e_703_14
  have e_703_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_15
  simp only [← c44130, ← c44131, k2, ← c44132] at e_703_15
  have hr := e_703_15
  simp only [e_703_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4636)) := by
  have c44133 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.1
  have c44134 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.1
  have c44135 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.1
  have c44136 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.1
  have c44137 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.1
  have c44138 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_16
  simp only [← c44133, ← c44134, ← c44135] at e_703_16
  have e_703_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_17
  simp only [← c44136, ← c44137, k2, ← c44138] at e_703_17
  have hr := e_703_17
  simp only [e_703_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 703 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4637)) := by
  have c44139 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44140 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44141 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44142 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44143 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44144 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_703_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_18
  simp only [← c44139, ← c44140, ← c44141] at e_703_18
  have e_703_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_19
  simp only [← c44142, ← c44143, k2, ← c44144] at e_703_19
  have hr := e_703_19
  simp only [e_703_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4638)) := by
  have c44145 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44146 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44147 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44148 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44149 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44150 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_0
  simp only [← c44145, ← c44146, ← c44147] at e_704_0
  have e_704_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_1
  simp only [← c44148, ← c44149, k2, ← c44150] at e_704_1
  have hr := e_704_1
  simp only [e_704_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4639)) := by
  have c44151 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44152 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44153 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44154 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44155 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44156 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_2
  simp only [← c44151, ← c44152, ← c44153] at e_704_2
  have e_704_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_3
  simp only [← c44154, ← c44155, k2, ← c44156] at e_704_3
  have hr := e_704_3
  simp only [e_704_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4640)) := by
  have c44157 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44158 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44159 := (publicBatchWrapper86_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44160 := (publicBatchWrapper86_copies1380 a h).1
  have c44161 := (publicBatchWrapper86_copies1380 a h).2.1
  have c44162 := (publicBatchWrapper86_copies1380 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_4
  simp only [← c44157, ← c44158, ← c44159] at e_704_4
  have e_704_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_5
  simp only [← c44160, ← c44161, k2, ← c44162] at e_704_5
  have hr := e_704_5
  simp only [e_704_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4641)) := by
  have c44163 := (publicBatchWrapper86_copies1380 a h).2.2.2.1
  have c44164 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.1
  have c44165 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.1
  have c44166 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.1
  have c44167 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.1
  have c44168 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_6
  simp only [← c44163, ← c44164, ← c44165] at e_704_6
  have e_704_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_7
  simp only [← c44166, ← c44167, k2, ← c44168] at e_704_7
  have hr := e_704_7
  simp only [e_704_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4642)) := by
  have c44169 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.1
  have c44170 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44171 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44172 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44173 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44174 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_8
  simp only [← c44169, ← c44170, ← c44171] at e_704_8
  have e_704_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_9
  simp only [← c44172, ← c44173, k2, ← c44174] at e_704_9
  have hr := e_704_9
  simp only [e_704_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4643)) := by
  have c44175 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44176 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44177 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44178 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44179 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44180 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_10
  simp only [← c44175, ← c44176, ← c44177] at e_704_10
  have e_704_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_11
  simp only [← c44178, ← c44179, k2, ← c44180] at e_704_11
  have hr := e_704_11
  simp only [e_704_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4644)) := by
  have c44181 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44182 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44183 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44184 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44185 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44186 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_12
  simp only [← c44181, ← c44182, ← c44183] at e_704_12
  have e_704_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_13
  simp only [← c44184, ← c44185, k2, ← c44186] at e_704_13
  have hr := e_704_13
  simp only [e_704_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4645)) := by
  have c44187 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44188 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44189 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44190 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44191 := (publicBatchWrapper86_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44192 := (publicBatchWrapper86_copies1381 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_14
  simp only [← c44187, ← c44188, ← c44189] at e_704_14
  have e_704_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_15
  simp only [← c44190, ← c44191, k2, ← c44192] at e_704_15
  have hr := e_704_15
  simp only [e_704_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4646)) := by
  have c44193 := (publicBatchWrapper86_copies1381 a h).2.1
  have c44194 := (publicBatchWrapper86_copies1381 a h).2.2.1
  have c44195 := (publicBatchWrapper86_copies1381 a h).2.2.2.1
  have c44196 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.1
  have c44197 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.1
  have c44198 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_16
  simp only [← c44193, ← c44194, ← c44195] at e_704_16
  have e_704_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_17
  simp only [← c44196, ← c44197, k2, ← c44198] at e_704_17
  have hr := e_704_17
  simp only [e_704_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 704 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4647)) := by
  have c44199 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.1
  have c44200 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.1
  have c44201 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.1
  have c44202 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44203 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44204 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_704_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_18
  simp only [← c44199, ← c44200, ← c44201] at e_704_18
  have e_704_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_19
  simp only [← c44202, ← c44203, k2, ← c44204] at e_704_19
  have hr := e_704_19
  simp only [e_704_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4648)) := by
  have c44205 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44206 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44207 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44208 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44209 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44210 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_0
  simp only [← c44205, ← c44206, ← c44207] at e_705_0
  have e_705_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_1
  simp only [← c44208, ← c44209, k2, ← c44210] at e_705_1
  have hr := e_705_1
  simp only [e_705_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4649)) := by
  have c44211 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44212 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44213 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44214 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44215 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44216 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_2
  simp only [← c44211, ← c44212, ← c44213] at e_705_2
  have e_705_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_3
  simp only [← c44214, ← c44215, k2, ← c44216] at e_705_3
  have hr := e_705_3
  simp only [e_705_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4650)) := by
  have c44217 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44218 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44219 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44220 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44221 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44222 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_4
  simp only [← c44217, ← c44218, ← c44219] at e_705_4
  have e_705_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_5
  simp only [← c44220, ← c44221, k2, ← c44222] at e_705_5
  have hr := e_705_5
  simp only [e_705_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4651)) := by
  have c44223 := (publicBatchWrapper86_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44224 := (publicBatchWrapper86_copies1382 a h).1
  have c44225 := (publicBatchWrapper86_copies1382 a h).2.1
  have c44226 := (publicBatchWrapper86_copies1382 a h).2.2.1
  have c44227 := (publicBatchWrapper86_copies1382 a h).2.2.2.1
  have c44228 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_6
  simp only [← c44223, ← c44224, ← c44225] at e_705_6
  have e_705_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_7
  simp only [← c44226, ← c44227, k2, ← c44228] at e_705_7
  have hr := e_705_7
  simp only [e_705_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4652)) := by
  have c44229 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.1
  have c44230 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.1
  have c44231 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.1
  have c44232 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.1
  have c44233 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.1
  have c44234 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_8
  simp only [← c44229, ← c44230, ← c44231] at e_705_8
  have e_705_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_9
  simp only [← c44232, ← c44233, k2, ← c44234] at e_705_9
  have hr := e_705_9
  simp only [e_705_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4653)) := by
  have c44235 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44236 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44237 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44238 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44239 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44240 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_10
  simp only [← c44235, ← c44236, ← c44237] at e_705_10
  have e_705_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_11
  simp only [← c44238, ← c44239, k2, ← c44240] at e_705_11
  have hr := e_705_11
  simp only [e_705_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4654)) := by
  have c44241 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44242 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44243 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44244 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44245 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44246 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_12
  simp only [← c44241, ← c44242, ← c44243] at e_705_12
  have e_705_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_13
  simp only [← c44244, ← c44245, k2, ← c44246] at e_705_13
  have hr := e_705_13
  simp only [e_705_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4655)) := by
  have c44247 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44248 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44249 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44250 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44251 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44252 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_14
  simp only [← c44247, ← c44248, ← c44249] at e_705_14
  have e_705_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_15
  simp only [← c44250, ← c44251, k2, ← c44252] at e_705_15
  have hr := e_705_15
  simp only [e_705_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4656)) := by
  have c44253 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44254 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44255 := (publicBatchWrapper86_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44256 := (publicBatchWrapper86_copies1383 a h).1
  have c44257 := (publicBatchWrapper86_copies1383 a h).2.1
  have c44258 := (publicBatchWrapper86_copies1383 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_16
  simp only [← c44253, ← c44254, ← c44255] at e_705_16
  have e_705_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_17
  simp only [← c44256, ← c44257, k2, ← c44258] at e_705_17
  have hr := e_705_17
  simp only [e_705_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 705 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4657)) := by
  have c44259 := (publicBatchWrapper86_copies1383 a h).2.2.2.1
  have c44260 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.1
  have c44261 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.1
  have c44262 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.1
  have c44263 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.1
  have c44264 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_705_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_18
  simp only [← c44259, ← c44260, ← c44261] at e_705_18
  have e_705_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_19
  simp only [← c44262, ← c44263, k2, ← c44264] at e_705_19
  have hr := e_705_19
  simp only [e_705_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4658)) := by
  have c44265 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.1
  have c44266 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44267 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44268 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44269 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44270 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_0
  simp only [← c44265, ← c44266, ← c44267] at e_706_0
  have e_706_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_1
  simp only [← c44268, ← c44269, k2, ← c44270] at e_706_1
  have hr := e_706_1
  simp only [e_706_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4659)) := by
  have c44271 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44272 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44273 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44274 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44275 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44276 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_2
  simp only [← c44271, ← c44272, ← c44273] at e_706_2
  have e_706_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_3
  simp only [← c44274, ← c44275, k2, ← c44276] at e_706_3
  have hr := e_706_3
  simp only [e_706_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4660)) := by
  have c44277 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44278 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44279 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44280 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44281 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44282 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_4
  simp only [← c44277, ← c44278, ← c44279] at e_706_4
  have e_706_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_5
  simp only [← c44280, ← c44281, k2, ← c44282] at e_706_5
  have hr := e_706_5
  simp only [e_706_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4661)) := by
  have c44283 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44284 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44285 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44286 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44287 := (publicBatchWrapper86_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44288 := (publicBatchWrapper86_copies1384 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_6
  simp only [← c44283, ← c44284, ← c44285] at e_706_6
  have e_706_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_7
  simp only [← c44286, ← c44287, k2, ← c44288] at e_706_7
  have hr := e_706_7
  simp only [e_706_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4662)) := by
  have c44289 := (publicBatchWrapper86_copies1384 a h).2.1
  have c44290 := (publicBatchWrapper86_copies1384 a h).2.2.1
  have c44291 := (publicBatchWrapper86_copies1384 a h).2.2.2.1
  have c44292 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.1
  have c44293 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.1
  have c44294 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_8
  simp only [← c44289, ← c44290, ← c44291] at e_706_8
  have e_706_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_9
  simp only [← c44292, ← c44293, k2, ← c44294] at e_706_9
  have hr := e_706_9
  simp only [e_706_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4663)) := by
  have c44295 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.1
  have c44296 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.1
  have c44297 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.1
  have c44298 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44299 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44300 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_10
  simp only [← c44295, ← c44296, ← c44297] at e_706_10
  have e_706_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_11
  simp only [← c44298, ← c44299, k2, ← c44300] at e_706_11
  have hr := e_706_11
  simp only [e_706_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4664)) := by
  have c44301 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44302 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44303 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44304 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44305 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44306 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_12
  simp only [← c44301, ← c44302, ← c44303] at e_706_12
  have e_706_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_13
  simp only [← c44304, ← c44305, k2, ← c44306] at e_706_13
  have hr := e_706_13
  simp only [e_706_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4665)) := by
  have c44307 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44308 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44309 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44310 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44311 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44312 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_14
  simp only [← c44307, ← c44308, ← c44309] at e_706_14
  have e_706_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_15
  simp only [← c44310, ← c44311, k2, ← c44312] at e_706_15
  have hr := e_706_15
  simp only [e_706_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4666)) := by
  have c44313 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44314 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44315 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44316 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44317 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44318 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_16
  simp only [← c44313, ← c44314, ← c44315] at e_706_16
  have e_706_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_17
  simp only [← c44316, ← c44317, k2, ← c44318] at e_706_17
  have hr := e_706_17
  simp only [e_706_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 706 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4667)) := by
  have c44319 := (publicBatchWrapper86_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44320 := (publicBatchWrapper86_copies1385 a h).1
  have c44321 := (publicBatchWrapper86_copies1385 a h).2.1
  have c44322 := (publicBatchWrapper86_copies1385 a h).2.2.1
  have c44323 := (publicBatchWrapper86_copies1385 a h).2.2.2.1
  have c44324 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_706_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_18
  simp only [← c44319, ← c44320, ← c44321] at e_706_18
  have e_706_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_19
  simp only [← c44322, ← c44323, k2, ← c44324] at e_706_19
  have hr := e_706_19
  simp only [e_706_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4668)) := by
  have c44325 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.1
  have c44326 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.1
  have c44327 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.1
  have c44328 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.1
  have c44329 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.1
  have c44330 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_0
  simp only [← c44325, ← c44326, ← c44327] at e_707_0
  have e_707_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_1
  simp only [← c44328, ← c44329, k2, ← c44330] at e_707_1
  have hr := e_707_1
  simp only [e_707_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4669)) := by
  have c44331 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44332 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44333 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44334 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44335 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44336 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_2
  simp only [← c44331, ← c44332, ← c44333] at e_707_2
  have e_707_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_3
  simp only [← c44334, ← c44335, k2, ← c44336] at e_707_3
  have hr := e_707_3
  simp only [e_707_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4670)) := by
  have c44337 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44338 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44339 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44340 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44341 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44342 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_4
  simp only [← c44337, ← c44338, ← c44339] at e_707_4
  have e_707_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_5
  simp only [← c44340, ← c44341, k2, ← c44342] at e_707_5
  have hr := e_707_5
  simp only [e_707_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4671)) := by
  have c44343 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44344 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44345 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44346 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44347 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44348 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_6
  simp only [← c44343, ← c44344, ← c44345] at e_707_6
  have e_707_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_7
  simp only [← c44346, ← c44347, k2, ← c44348] at e_707_7
  have hr := e_707_7
  simp only [e_707_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4672)) := by
  have c44349 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44350 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44351 := (publicBatchWrapper86_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44352 := (publicBatchWrapper86_copies1386 a h).1
  have c44353 := (publicBatchWrapper86_copies1386 a h).2.1
  have c44354 := (publicBatchWrapper86_copies1386 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_8
  simp only [← c44349, ← c44350, ← c44351] at e_707_8
  have e_707_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_9
  simp only [← c44352, ← c44353, k2, ← c44354] at e_707_9
  have hr := e_707_9
  simp only [e_707_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4673)) := by
  have c44355 := (publicBatchWrapper86_copies1386 a h).2.2.2.1
  have c44356 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.1
  have c44357 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.1
  have c44358 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.1
  have c44359 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.1
  have c44360 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_10
  simp only [← c44355, ← c44356, ← c44357] at e_707_10
  have e_707_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_11
  simp only [← c44358, ← c44359, k2, ← c44360] at e_707_11
  have hr := e_707_11
  simp only [e_707_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4674)) := by
  have c44361 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.1
  have c44362 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44363 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44364 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44365 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44366 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_12
  simp only [← c44361, ← c44362, ← c44363] at e_707_12
  have e_707_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_13
  simp only [← c44364, ← c44365, k2, ← c44366] at e_707_13
  have hr := e_707_13
  simp only [e_707_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4675)) := by
  have c44367 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44368 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44369 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44370 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44371 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44372 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_14
  simp only [← c44367, ← c44368, ← c44369] at e_707_14
  have e_707_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_15
  simp only [← c44370, ← c44371, k2, ← c44372] at e_707_15
  have hr := e_707_15
  simp only [e_707_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4676)) := by
  have c44373 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44374 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44375 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44376 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44377 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44378 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_16
  simp only [← c44373, ← c44374, ← c44375] at e_707_16
  have e_707_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_17
  simp only [← c44376, ← c44377, k2, ← c44378] at e_707_17
  have hr := e_707_17
  simp only [e_707_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 707 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4677)) := by
  have c44379 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44380 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44381 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44382 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44383 := (publicBatchWrapper86_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44384 := (publicBatchWrapper86_copies1387 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_707_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_18
  simp only [← c44379, ← c44380, ← c44381] at e_707_18
  have e_707_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_19
  simp only [← c44382, ← c44383, k2, ← c44384] at e_707_19
  have hr := e_707_19
  simp only [e_707_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4678)) := by
  have c44385 := (publicBatchWrapper86_copies1387 a h).2.1
  have c44386 := (publicBatchWrapper86_copies1387 a h).2.2.1
  have c44387 := (publicBatchWrapper86_copies1387 a h).2.2.2.1
  have c44388 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.1
  have c44389 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.1
  have c44390 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_0
  simp only [← c44385, ← c44386, ← c44387] at e_708_0
  have e_708_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_1
  simp only [← c44388, ← c44389, k2, ← c44390] at e_708_1
  have hr := e_708_1
  simp only [e_708_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4679)) := by
  have c44391 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.1
  have c44392 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.1
  have c44393 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.1
  have c44394 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44395 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44396 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_2
  simp only [← c44391, ← c44392, ← c44393] at e_708_2
  have e_708_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_3
  simp only [← c44394, ← c44395, k2, ← c44396] at e_708_3
  have hr := e_708_3
  simp only [e_708_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4680)) := by
  have c44397 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44398 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44399 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44400 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44401 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44402 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_4
  simp only [← c44397, ← c44398, ← c44399] at e_708_4
  have e_708_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_5
  simp only [← c44400, ← c44401, k2, ← c44402] at e_708_5
  have hr := e_708_5
  simp only [e_708_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4681)) := by
  have c44403 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44404 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44405 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44406 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44407 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44408 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_6
  simp only [← c44403, ← c44404, ← c44405] at e_708_6
  have e_708_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_7
  simp only [← c44406, ← c44407, k2, ← c44408] at e_708_7
  have hr := e_708_7
  simp only [e_708_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4682)) := by
  have c44409 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44410 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44411 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44412 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44413 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44414 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_8
  simp only [← c44409, ← c44410, ← c44411] at e_708_8
  have e_708_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_9
  simp only [← c44412, ← c44413, k2, ← c44414] at e_708_9
  have hr := e_708_9
  simp only [e_708_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4683)) := by
  have c44415 := (publicBatchWrapper86_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44416 := (publicBatchWrapper86_copies1388 a h).1
  have c44417 := (publicBatchWrapper86_copies1388 a h).2.1
  have c44418 := (publicBatchWrapper86_copies1388 a h).2.2.1
  have c44419 := (publicBatchWrapper86_copies1388 a h).2.2.2.1
  have c44420 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_10
  simp only [← c44415, ← c44416, ← c44417] at e_708_10
  have e_708_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_11
  simp only [← c44418, ← c44419, k2, ← c44420] at e_708_11
  have hr := e_708_11
  simp only [e_708_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4684)) := by
  have c44421 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.1
  have c44422 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.1
  have c44423 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.1
  have c44424 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.1
  have c44425 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.1
  have c44426 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_12
  simp only [← c44421, ← c44422, ← c44423] at e_708_12
  have e_708_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_13
  simp only [← c44424, ← c44425, k2, ← c44426] at e_708_13
  have hr := e_708_13
  simp only [e_708_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4685)) := by
  have c44427 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44428 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44429 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44430 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44431 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44432 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_14
  simp only [← c44427, ← c44428, ← c44429] at e_708_14
  have e_708_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_15
  simp only [← c44430, ← c44431, k2, ← c44432] at e_708_15
  have hr := e_708_15
  simp only [e_708_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4686)) := by
  have c44433 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44434 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44435 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44436 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44437 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44438 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_16
  simp only [← c44433, ← c44434, ← c44435] at e_708_16
  have e_708_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_17
  simp only [← c44436, ← c44437, k2, ← c44438] at e_708_17
  have hr := e_708_17
  simp only [e_708_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 708 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4687)) := by
  have c44439 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44440 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44441 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44442 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44443 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44444 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_708_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_18
  simp only [← c44439, ← c44440, ← c44441] at e_708_18
  have e_708_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_19
  simp only [← c44442, ← c44443, k2, ← c44444] at e_708_19
  have hr := e_708_19
  simp only [e_708_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4688)) := by
  have c44445 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44446 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44447 := (publicBatchWrapper86_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44448 := (publicBatchWrapper86_copies1389 a h).1
  have c44449 := (publicBatchWrapper86_copies1389 a h).2.1
  have c44450 := (publicBatchWrapper86_copies1389 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_0
  simp only [← c44445, ← c44446, ← c44447] at e_709_0
  have e_709_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_1
  simp only [← c44448, ← c44449, k2, ← c44450] at e_709_1
  have hr := e_709_1
  simp only [e_709_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4689)) := by
  have c44451 := (publicBatchWrapper86_copies1389 a h).2.2.2.1
  have c44452 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.1
  have c44453 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.1
  have c44454 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.1
  have c44455 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.1
  have c44456 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_2
  simp only [← c44451, ← c44452, ← c44453] at e_709_2
  have e_709_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_3
  simp only [← c44454, ← c44455, k2, ← c44456] at e_709_3
  have hr := e_709_3
  simp only [e_709_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 23) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4690)) := by
  have c44457 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.1
  have c44458 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44459 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44460 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44461 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44462 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_4
  simp only [← c44457, ← c44458, ← c44459] at e_709_4
  have e_709_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_5
  simp only [← c44460, ← c44461, k2, ← c44462] at e_709_5
  have hr := e_709_5
  simp only [e_709_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 31) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4691)) := by
  have c44463 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44464 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44465 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44466 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44467 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44468 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_6
  simp only [← c44463, ← c44464, ← c44465] at e_709_6
  have e_709_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_7
  simp only [← c44466, ← c44467, k2, ← c44468] at e_709_7
  have hr := e_709_7
  simp only [e_709_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 39) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4692)) := by
  have c44469 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44470 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44471 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44472 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44473 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44474 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_8
  simp only [← c44469, ← c44470, ← c44471] at e_709_8
  have e_709_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_9
  simp only [← c44472, ← c44473, k2, ← c44474] at e_709_9
  have hr := e_709_9
  simp only [e_709_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 47) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4693)) := by
  have c44475 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44476 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44477 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44478 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44479 := (publicBatchWrapper86_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44480 := (publicBatchWrapper86_copies1390 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_10
  simp only [← c44475, ← c44476, ← c44477] at e_709_10
  have e_709_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_11
  simp only [← c44478, ← c44479, k2, ← c44480] at e_709_11
  have hr := e_709_11
  simp only [e_709_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 55) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4694)) := by
  have c44481 := (publicBatchWrapper86_copies1390 a h).2.1
  have c44482 := (publicBatchWrapper86_copies1390 a h).2.2.1
  have c44483 := (publicBatchWrapper86_copies1390 a h).2.2.2.1
  have c44484 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.1
  have c44485 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.1
  have c44486 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_12
  simp only [← c44481, ← c44482, ← c44483] at e_709_12
  have e_709_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_13
  simp only [← c44484, ← c44485, k2, ← c44486] at e_709_13
  have hr := e_709_13
  simp only [e_709_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 63) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4695)) := by
  have c44487 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.1
  have c44488 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.1
  have c44489 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.1
  have c44490 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44491 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44492 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_14
  simp only [← c44487, ← c44488, ← c44489] at e_709_14
  have e_709_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_15
  simp only [← c44490, ← c44491, k2, ← c44492] at e_709_15
  have hr := e_709_15
  simp only [e_709_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 71) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4696)) := by
  have c44493 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44494 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44495 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44496 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44497 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44498 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_16
  simp only [← c44493, ← c44494, ← c44495] at e_709_16
  have e_709_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_17
  simp only [← c44496, ← c44497, k2, ← c44498] at e_709_17
  have hr := e_709_17
  simp only [e_709_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 709 79) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4697)) := by
  have c44499 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44500 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44501 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44502 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44503 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44504 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_709_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_18
  simp only [← c44499, ← c44500, ← c44501] at e_709_18
  have e_709_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_19
  simp only [← c44502, ← c44503, k2, ← c44504] at e_709_19
  have hr := e_709_19
  simp only [e_709_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 710 7) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4698)) := by
  have c44505 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44506 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44507 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44508 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44509 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44510 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_710_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_0
  simp only [← c44505, ← c44506, ← c44507] at e_710_0
  have e_710_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_1
  simp only [← c44508, ← c44509, k2, ← c44510] at e_710_1
  have hr := e_710_1
  simp only [e_710_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f6693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 710 15) = bselect (a (.wire 23 79)) (a (.virt 200)) (a (.virt 4699)) := by
  have c44511 := (publicBatchWrapper86_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44512 := (publicBatchWrapper86_copies1391 a h).1
  have c44513 := (publicBatchWrapper86_copies1391 a h).2.1
  have c44514 := (publicBatchWrapper86_copies1391 a h).2.2.1
  have c44515 := (publicBatchWrapper86_copies1391 a h).2.2.2.1
  have c44516 := (publicBatchWrapper86_copies1391 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_710_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_2
  simp only [← c44511, ← c44512, ← c44513] at e_710_2
  have e_710_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_3
  simp only [← c44514, ← c44515, k2, ← c44516] at e_710_3
  have hr := e_710_3
  simp only [e_710_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
