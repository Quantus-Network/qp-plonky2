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

theorem publicBatchWrapper89_f18229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6173)) := by
  have c113880 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113881 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113882 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113883 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113884 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113885 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_16
  simp only [← c113880, ← c113881, ← c113882] at e_1864_16
  have e_1864_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_17
  simp only [← c113883, ← c113884, k2, ← c113885] at e_1864_17
  have hr := e_1864_17
  simp only [e_1864_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6174)) := by
  have c113886 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113887 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113888 := (publicBatchWrapper89_copies3559 a h).1
  have c113889 := (publicBatchWrapper89_copies3559 a h).2.1
  have c113890 := (publicBatchWrapper89_copies3559 a h).2.2.1
  have c113891 := (publicBatchWrapper89_copies3559 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_18
  simp only [← c113886, ← c113887, ← c113888] at e_1864_18
  have e_1864_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_19
  simp only [← c113889, ← c113890, k2, ← c113891] at e_1864_19
  have hr := e_1864_19
  simp only [e_1864_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6175)) := by
  have c113892 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.1
  have c113893 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.1
  have c113894 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.1
  have c113895 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.1
  have c113896 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.1
  have c113897 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_0
  simp only [← c113892, ← c113893, ← c113894] at e_1865_0
  have e_1865_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_1
  simp only [← c113895, ← c113896, k2, ← c113897] at e_1865_1
  have hr := e_1865_1
  simp only [e_1865_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6176)) := by
  have c113898 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113899 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113900 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113901 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113902 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113903 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_2
  simp only [← c113898, ← c113899, ← c113900] at e_1865_2
  have e_1865_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_3
  simp only [← c113901, ← c113902, k2, ← c113903] at e_1865_3
  have hr := e_1865_3
  simp only [e_1865_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6177)) := by
  have c113904 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113905 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113906 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113907 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113908 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113909 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_4
  simp only [← c113904, ← c113905, ← c113906] at e_1865_4
  have e_1865_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_5
  simp only [← c113907, ← c113908, k2, ← c113909] at e_1865_5
  have hr := e_1865_5
  simp only [e_1865_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6178)) := by
  have c113910 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113911 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113912 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113913 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113914 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113915 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_6
  simp only [← c113910, ← c113911, ← c113912] at e_1865_6
  have e_1865_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_7
  simp only [← c113913, ← c113914, k2, ← c113915] at e_1865_7
  have hr := e_1865_7
  simp only [e_1865_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6179)) := by
  have c113916 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113917 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113918 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113919 := (publicBatchWrapper89_copies3559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113920 := (publicBatchWrapper89_copies3560 a h).1
  have c113921 := (publicBatchWrapper89_copies3560 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_8
  simp only [← c113916, ← c113917, ← c113918] at e_1865_8
  have e_1865_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_9
  simp only [← c113919, ← c113920, k2, ← c113921] at e_1865_9
  have hr := e_1865_9
  simp only [e_1865_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6180)) := by
  have c113922 := (publicBatchWrapper89_copies3560 a h).2.2.1
  have c113923 := (publicBatchWrapper89_copies3560 a h).2.2.2.1
  have c113924 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.1
  have c113925 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.1
  have c113926 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.1
  have c113927 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_10
  simp only [← c113922, ← c113923, ← c113924] at e_1865_10
  have e_1865_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_11
  simp only [← c113925, ← c113926, k2, ← c113927] at e_1865_11
  have hr := e_1865_11
  simp only [e_1865_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6181)) := by
  have c113928 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.1
  have c113929 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.1
  have c113930 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113931 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113932 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113933 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_12
  simp only [← c113928, ← c113929, ← c113930] at e_1865_12
  have e_1865_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_13
  simp only [← c113931, ← c113932, k2, ← c113933] at e_1865_13
  have hr := e_1865_13
  simp only [e_1865_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6182)) := by
  have c113934 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113935 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113936 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113937 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113938 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113939 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_14
  simp only [← c113934, ← c113935, ← c113936] at e_1865_14
  have e_1865_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_15
  simp only [← c113937, ← c113938, k2, ← c113939] at e_1865_15
  have hr := e_1865_15
  simp only [e_1865_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6183)) := by
  have c113940 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113941 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113942 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113943 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113944 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113945 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_16
  simp only [← c113940, ← c113941, ← c113942] at e_1865_16
  have e_1865_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_17
  simp only [← c113943, ← c113944, k2, ← c113945] at e_1865_17
  have hr := e_1865_17
  simp only [e_1865_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1865 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6184)) := by
  have c113946 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113947 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113948 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113949 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113950 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113951 := (publicBatchWrapper89_copies3560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1865_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_18
  simp only [← c113946, ← c113947, ← c113948] at e_1865_18
  have e_1865_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1865 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_19
  simp only [← c113949, ← c113950, k2, ← c113951] at e_1865_19
  have hr := e_1865_19
  simp only [e_1865_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6185)) := by
  have c113952 := (publicBatchWrapper89_copies3561 a h).1
  have c113953 := (publicBatchWrapper89_copies3561 a h).2.1
  have c113954 := (publicBatchWrapper89_copies3561 a h).2.2.1
  have c113955 := (publicBatchWrapper89_copies3561 a h).2.2.2.1
  have c113956 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.1
  have c113957 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_0
  simp only [← c113952, ← c113953, ← c113954] at e_1866_0
  have e_1866_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_1
  simp only [← c113955, ← c113956, k2, ← c113957] at e_1866_1
  have hr := e_1866_1
  simp only [e_1866_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6186)) := by
  have c113958 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.1
  have c113959 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.1
  have c113960 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.1
  have c113961 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.1
  have c113962 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113963 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_2
  simp only [← c113958, ← c113959, ← c113960] at e_1866_2
  have e_1866_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_3
  simp only [← c113961, ← c113962, k2, ← c113963] at e_1866_3
  have hr := e_1866_3
  simp only [e_1866_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6187)) := by
  have c113964 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113965 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113966 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113967 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113968 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113969 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_4
  simp only [← c113964, ← c113965, ← c113966] at e_1866_4
  have e_1866_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_5
  simp only [← c113967, ← c113968, k2, ← c113969] at e_1866_5
  have hr := e_1866_5
  simp only [e_1866_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6188)) := by
  have c113970 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113971 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113972 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113973 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113974 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113975 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_6
  simp only [← c113970, ← c113971, ← c113972] at e_1866_6
  have e_1866_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_7
  simp only [← c113973, ← c113974, k2, ← c113975] at e_1866_7
  have hr := e_1866_7
  simp only [e_1866_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6189)) := by
  have c113976 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113977 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113978 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113979 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113980 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113981 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_8
  simp only [← c113976, ← c113977, ← c113978] at e_1866_8
  have e_1866_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_9
  simp only [← c113979, ← c113980, k2, ← c113981] at e_1866_9
  have hr := e_1866_9
  simp only [e_1866_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6190)) := by
  have c113982 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113983 := (publicBatchWrapper89_copies3561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113984 := (publicBatchWrapper89_copies3562 a h).1
  have c113985 := (publicBatchWrapper89_copies3562 a h).2.1
  have c113986 := (publicBatchWrapper89_copies3562 a h).2.2.1
  have c113987 := (publicBatchWrapper89_copies3562 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_10
  simp only [← c113982, ← c113983, ← c113984] at e_1866_10
  have e_1866_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_11
  simp only [← c113985, ← c113986, k2, ← c113987] at e_1866_11
  have hr := e_1866_11
  simp only [e_1866_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6191)) := by
  have c113988 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.1
  have c113989 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.1
  have c113990 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.1
  have c113991 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.1
  have c113992 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.1
  have c113993 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_12
  simp only [← c113988, ← c113989, ← c113990] at e_1866_12
  have e_1866_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_13
  simp only [← c113991, ← c113992, k2, ← c113993] at e_1866_13
  have hr := e_1866_13
  simp only [e_1866_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6192)) := by
  have c113994 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113995 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113996 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113997 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113998 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113999 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_14
  simp only [← c113994, ← c113995, ← c113996] at e_1866_14
  have e_1866_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_15
  simp only [← c113997, ← c113998, k2, ← c113999] at e_1866_15
  have hr := e_1866_15
  simp only [e_1866_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6193)) := by
  have c114000 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114001 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114002 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114003 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114004 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114005 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_16
  simp only [← c114000, ← c114001, ← c114002] at e_1866_16
  have e_1866_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_17
  simp only [← c114003, ← c114004, k2, ← c114005] at e_1866_17
  have hr := e_1866_17
  simp only [e_1866_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1866 79) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6194)) := by
  have c114006 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114007 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114008 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114009 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114010 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114011 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1866_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_18
  simp only [← c114006, ← c114007, ← c114008] at e_1866_18
  have e_1866_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1866 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_19
  simp only [← c114009, ← c114010, k2, ← c114011] at e_1866_19
  have hr := e_1866_19
  simp only [e_1866_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 7) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6195)) := by
  have c114012 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114013 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114014 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114015 := (publicBatchWrapper89_copies3562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114016 := (publicBatchWrapper89_copies3563 a h).1
  have c114017 := (publicBatchWrapper89_copies3563 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_0
  simp only [← c114012, ← c114013, ← c114014] at e_1867_0
  have e_1867_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_1
  simp only [← c114015, ← c114016, k2, ← c114017] at e_1867_1
  have hr := e_1867_1
  simp only [e_1867_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 15) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6196)) := by
  have c114018 := (publicBatchWrapper89_copies3563 a h).2.2.1
  have c114019 := (publicBatchWrapper89_copies3563 a h).2.2.2.1
  have c114020 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.1
  have c114021 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.1
  have c114022 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.1
  have c114023 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_2
  simp only [← c114018, ← c114019, ← c114020] at e_1867_2
  have e_1867_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_3
  simp only [← c114021, ← c114022, k2, ← c114023] at e_1867_3
  have hr := e_1867_3
  simp only [e_1867_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 23) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6197)) := by
  have c114024 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.1
  have c114025 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.1
  have c114026 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114027 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114028 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114029 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_4
  simp only [← c114024, ← c114025, ← c114026] at e_1867_4
  have e_1867_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_5
  simp only [← c114027, ← c114028, k2, ← c114029] at e_1867_5
  have hr := e_1867_5
  simp only [e_1867_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 31) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6198)) := by
  have c114030 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114031 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114032 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114033 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114034 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114035 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_6
  simp only [← c114030, ← c114031, ← c114032] at e_1867_6
  have e_1867_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_7
  simp only [← c114033, ← c114034, k2, ← c114035] at e_1867_7
  have hr := e_1867_7
  simp only [e_1867_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 39) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6199)) := by
  have c114036 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114037 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114038 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114039 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114040 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114041 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_8
  simp only [← c114036, ← c114037, ← c114038] at e_1867_8
  have e_1867_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_9
  simp only [← c114039, ← c114040, k2, ← c114041] at e_1867_9
  have hr := e_1867_9
  simp only [e_1867_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6200)) := by
  have c114042 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114043 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114044 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114045 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114046 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114047 := (publicBatchWrapper89_copies3563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_10
  simp only [← c114042, ← c114043, ← c114044] at e_1867_10
  have e_1867_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_11
  simp only [← c114045, ← c114046, k2, ← c114047] at e_1867_11
  have hr := e_1867_11
  simp only [e_1867_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6201)) := by
  have c114048 := (publicBatchWrapper89_copies3564 a h).1
  have c114049 := (publicBatchWrapper89_copies3564 a h).2.1
  have c114050 := (publicBatchWrapper89_copies3564 a h).2.2.1
  have c114051 := (publicBatchWrapper89_copies3564 a h).2.2.2.1
  have c114052 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.1
  have c114053 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_12
  simp only [← c114048, ← c114049, ← c114050] at e_1867_12
  have e_1867_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_13
  simp only [← c114051, ← c114052, k2, ← c114053] at e_1867_13
  have hr := e_1867_13
  simp only [e_1867_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6370)) := by
  have c114054 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.1
  have c114055 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.1
  have c114056 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.1
  have c114057 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.1
  have c114058 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114059 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_14
  simp only [← c114054, ← c114055, ← c114056] at e_1867_14
  have e_1867_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_15
  simp only [← c114057, ← c114058, k2, ← c114059] at e_1867_15
  have hr := e_1867_15
  simp only [e_1867_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6371)) := by
  have c114060 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114061 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114062 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114063 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114064 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114065 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_16
  simp only [← c114060, ← c114061, ← c114062] at e_1867_16
  have e_1867_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_17
  simp only [← c114063, ← c114064, k2, ← c114065] at e_1867_17
  have hr := e_1867_17
  simp only [e_1867_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1867 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6372)) := by
  have c114066 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114067 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114068 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114069 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114070 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114071 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1867_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_18
  simp only [← c114066, ← c114067, ← c114068] at e_1867_18
  have e_1867_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1867 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_19
  simp only [← c114069, ← c114070, k2, ← c114071] at e_1867_19
  have hr := e_1867_19
  simp only [e_1867_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6373)) := by
  have c114072 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114073 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114074 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114075 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114076 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114077 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_0
  simp only [← c114072, ← c114073, ← c114074] at e_1868_0
  have e_1868_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_1
  simp only [← c114075, ← c114076, k2, ← c114077] at e_1868_1
  have hr := e_1868_1
  simp only [e_1868_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6374)) := by
  have c114078 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114079 := (publicBatchWrapper89_copies3564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114080 := (publicBatchWrapper89_copies3565 a h).1
  have c114081 := (publicBatchWrapper89_copies3565 a h).2.1
  have c114082 := (publicBatchWrapper89_copies3565 a h).2.2.1
  have c114083 := (publicBatchWrapper89_copies3565 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_2
  simp only [← c114078, ← c114079, ← c114080] at e_1868_2
  have e_1868_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_3
  simp only [← c114081, ← c114082, k2, ← c114083] at e_1868_3
  have hr := e_1868_3
  simp only [e_1868_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6375)) := by
  have c114084 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.1
  have c114085 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.1
  have c114086 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.1
  have c114087 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.1
  have c114088 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.1
  have c114089 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_4
  simp only [← c114084, ← c114085, ← c114086] at e_1868_4
  have e_1868_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_5
  simp only [← c114087, ← c114088, k2, ← c114089] at e_1868_5
  have hr := e_1868_5
  simp only [e_1868_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6376)) := by
  have c114090 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114091 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114092 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114093 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114094 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114095 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_6
  simp only [← c114090, ← c114091, ← c114092] at e_1868_6
  have e_1868_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_7
  simp only [← c114093, ← c114094, k2, ← c114095] at e_1868_7
  have hr := e_1868_7
  simp only [e_1868_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6377)) := by
  have c114096 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114097 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114098 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114099 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114100 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114101 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_8
  simp only [← c114096, ← c114097, ← c114098] at e_1868_8
  have e_1868_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_9
  simp only [← c114099, ← c114100, k2, ← c114101] at e_1868_9
  have hr := e_1868_9
  simp only [e_1868_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6378)) := by
  have c114102 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114103 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114104 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114105 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114106 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114107 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_10
  simp only [← c114102, ← c114103, ← c114104] at e_1868_10
  have e_1868_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_11
  simp only [← c114105, ← c114106, k2, ← c114107] at e_1868_11
  have hr := e_1868_11
  simp only [e_1868_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6379)) := by
  have c114108 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114109 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114110 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114111 := (publicBatchWrapper89_copies3565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114112 := (publicBatchWrapper89_copies3566 a h).1
  have c114113 := (publicBatchWrapper89_copies3566 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_12
  simp only [← c114108, ← c114109, ← c114110] at e_1868_12
  have e_1868_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_13
  simp only [← c114111, ← c114112, k2, ← c114113] at e_1868_13
  have hr := e_1868_13
  simp only [e_1868_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6380)) := by
  have c114114 := (publicBatchWrapper89_copies3566 a h).2.2.1
  have c114115 := (publicBatchWrapper89_copies3566 a h).2.2.2.1
  have c114116 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.1
  have c114117 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.1
  have c114118 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.1
  have c114119 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_14
  simp only [← c114114, ← c114115, ← c114116] at e_1868_14
  have e_1868_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_15
  simp only [← c114117, ← c114118, k2, ← c114119] at e_1868_15
  have hr := e_1868_15
  simp only [e_1868_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6381)) := by
  have c114120 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.1
  have c114121 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.1
  have c114122 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114123 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114124 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114125 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_16
  simp only [← c114120, ← c114121, ← c114122] at e_1868_16
  have e_1868_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_17
  simp only [← c114123, ← c114124, k2, ← c114125] at e_1868_17
  have hr := e_1868_17
  simp only [e_1868_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1868 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6382)) := by
  have c114126 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114127 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114128 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114129 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114130 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114131 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1868_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_18
  simp only [← c114126, ← c114127, ← c114128] at e_1868_18
  have e_1868_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1868 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_19
  simp only [← c114129, ← c114130, k2, ← c114131] at e_1868_19
  have hr := e_1868_19
  simp only [e_1868_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6383)) := by
  have c114132 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114133 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114134 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114135 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114136 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114137 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_0
  simp only [← c114132, ← c114133, ← c114134] at e_1869_0
  have e_1869_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_1
  simp only [← c114135, ← c114136, k2, ← c114137] at e_1869_1
  have hr := e_1869_1
  simp only [e_1869_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6384)) := by
  have c114138 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114139 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114140 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114141 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114142 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114143 := (publicBatchWrapper89_copies3566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_2
  simp only [← c114138, ← c114139, ← c114140] at e_1869_2
  have e_1869_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_3
  simp only [← c114141, ← c114142, k2, ← c114143] at e_1869_3
  have hr := e_1869_3
  simp only [e_1869_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6385)) := by
  have c114144 := (publicBatchWrapper89_copies3567 a h).1
  have c114145 := (publicBatchWrapper89_copies3567 a h).2.1
  have c114146 := (publicBatchWrapper89_copies3567 a h).2.2.1
  have c114147 := (publicBatchWrapper89_copies3567 a h).2.2.2.1
  have c114148 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.1
  have c114149 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_4
  simp only [← c114144, ← c114145, ← c114146] at e_1869_4
  have e_1869_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_5
  simp only [← c114147, ← c114148, k2, ← c114149] at e_1869_5
  have hr := e_1869_5
  simp only [e_1869_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6386)) := by
  have c114150 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.1
  have c114151 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.1
  have c114152 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.1
  have c114153 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.1
  have c114154 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114155 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_6
  simp only [← c114150, ← c114151, ← c114152] at e_1869_6
  have e_1869_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_7
  simp only [← c114153, ← c114154, k2, ← c114155] at e_1869_7
  have hr := e_1869_7
  simp only [e_1869_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6387)) := by
  have c114156 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114157 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114158 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114159 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114160 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114161 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_8
  simp only [← c114156, ← c114157, ← c114158] at e_1869_8
  have e_1869_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_9
  simp only [← c114159, ← c114160, k2, ← c114161] at e_1869_9
  have hr := e_1869_9
  simp only [e_1869_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6388)) := by
  have c114162 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114163 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114164 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114165 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114166 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114167 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_10
  simp only [← c114162, ← c114163, ← c114164] at e_1869_10
  have e_1869_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_11
  simp only [← c114165, ← c114166, k2, ← c114167] at e_1869_11
  have hr := e_1869_11
  simp only [e_1869_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6389)) := by
  have c114168 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114169 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114170 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114171 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114172 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114173 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_12
  simp only [← c114168, ← c114169, ← c114170] at e_1869_12
  have e_1869_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_13
  simp only [← c114171, ← c114172, k2, ← c114173] at e_1869_13
  have hr := e_1869_13
  simp only [e_1869_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6390)) := by
  have c114174 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114175 := (publicBatchWrapper89_copies3567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114176 := (publicBatchWrapper89_copies3568 a h).1
  have c114177 := (publicBatchWrapper89_copies3568 a h).2.1
  have c114178 := (publicBatchWrapper89_copies3568 a h).2.2.1
  have c114179 := (publicBatchWrapper89_copies3568 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_14
  simp only [← c114174, ← c114175, ← c114176] at e_1869_14
  have e_1869_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_15
  simp only [← c114177, ← c114178, k2, ← c114179] at e_1869_15
  have hr := e_1869_15
  simp only [e_1869_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6391)) := by
  have c114180 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.1
  have c114181 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.1
  have c114182 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.1
  have c114183 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.1
  have c114184 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.1
  have c114185 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_16
  simp only [← c114180, ← c114181, ← c114182] at e_1869_16
  have e_1869_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_17
  simp only [← c114183, ← c114184, k2, ← c114185] at e_1869_17
  have hr := e_1869_17
  simp only [e_1869_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1869 79) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6392)) := by
  have c114186 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114187 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114188 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114189 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114190 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114191 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1869_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_18
  simp only [← c114186, ← c114187, ← c114188] at e_1869_18
  have e_1869_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1869 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_19
  simp only [← c114189, ← c114190, k2, ← c114191] at e_1869_19
  have hr := e_1869_19
  simp only [e_1869_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 7) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6393)) := by
  have c114192 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114193 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114194 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114195 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114196 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114197 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_0
  simp only [← c114192, ← c114193, ← c114194] at e_1870_0
  have e_1870_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_1
  simp only [← c114195, ← c114196, k2, ← c114197] at e_1870_1
  have hr := e_1870_1
  simp only [e_1870_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 15) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6394)) := by
  have c114198 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114199 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114200 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114201 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114202 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114203 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_2
  simp only [← c114198, ← c114199, ← c114200] at e_1870_2
  have e_1870_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_3
  simp only [← c114201, ← c114202, k2, ← c114203] at e_1870_3
  have hr := e_1870_3
  simp only [e_1870_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 23) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6395)) := by
  have c114204 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114205 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114206 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114207 := (publicBatchWrapper89_copies3568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114208 := (publicBatchWrapper89_copies3569 a h).1
  have c114209 := (publicBatchWrapper89_copies3569 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_4
  simp only [← c114204, ← c114205, ← c114206] at e_1870_4
  have e_1870_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_5
  simp only [← c114207, ← c114208, k2, ← c114209] at e_1870_5
  have hr := e_1870_5
  simp only [e_1870_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 31) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6396)) := by
  have c114210 := (publicBatchWrapper89_copies3569 a h).2.2.1
  have c114211 := (publicBatchWrapper89_copies3569 a h).2.2.2.1
  have c114212 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.1
  have c114213 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.1
  have c114214 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.1
  have c114215 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_6
  simp only [← c114210, ← c114211, ← c114212] at e_1870_6
  have e_1870_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_7
  simp only [← c114213, ← c114214, k2, ← c114215] at e_1870_7
  have hr := e_1870_7
  simp only [e_1870_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 39) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6397)) := by
  have c114216 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.1
  have c114217 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.1
  have c114218 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114219 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114220 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114221 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_8
  simp only [← c114216, ← c114217, ← c114218] at e_1870_8
  have e_1870_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_9
  simp only [← c114219, ← c114220, k2, ← c114221] at e_1870_9
  have hr := e_1870_9
  simp only [e_1870_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 47) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6398)) := by
  have c114222 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114223 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114224 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114225 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114226 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114227 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_10
  simp only [← c114222, ← c114223, ← c114224] at e_1870_10
  have e_1870_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_11
  simp only [← c114225, ← c114226, k2, ← c114227] at e_1870_11
  have hr := e_1870_11
  simp only [e_1870_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 55) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6399)) := by
  have c114228 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114229 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114230 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114231 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114232 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114233 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_12
  simp only [← c114228, ← c114229, ← c114230] at e_1870_12
  have e_1870_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_13
  simp only [← c114231, ← c114232, k2, ← c114233] at e_1870_13
  have hr := e_1870_13
  simp only [e_1870_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 63) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6400)) := by
  have c114234 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114235 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114236 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114237 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114238 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114239 := (publicBatchWrapper89_copies3569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_14
  simp only [← c114234, ← c114235, ← c114236] at e_1870_14
  have e_1870_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_15
  simp only [← c114237, ← c114238, k2, ← c114239] at e_1870_15
  have hr := e_1870_15
  simp only [e_1870_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 71) = bselect (a (.wire 31 63)) (a (.virt 200)) (a (.virt 6401)) := by
  have c114240 := (publicBatchWrapper89_copies3570 a h).1
  have c114241 := (publicBatchWrapper89_copies3570 a h).2.1
  have c114242 := (publicBatchWrapper89_copies3570 a h).2.2.1
  have c114243 := (publicBatchWrapper89_copies3570 a h).2.2.2.1
  have c114244 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.1
  have c114245 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_16
  simp only [← c114240, ← c114241, ← c114242] at e_1870_16
  have e_1870_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_17
  simp only [← c114243, ← c114244, k2, ← c114245] at e_1870_17
  have hr := e_1870_17
  simp only [e_1870_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1870 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6570)) := by
  have c114246 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.1
  have c114247 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.1
  have c114248 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.1
  have c114249 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.1
  have c114250 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114251 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1870_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_18
  simp only [← c114246, ← c114247, ← c114248] at e_1870_18
  have e_1870_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1870 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_19
  simp only [← c114249, ← c114250, k2, ← c114251] at e_1870_19
  have hr := e_1870_19
  simp only [e_1870_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6571)) := by
  have c114252 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114253 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114254 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114255 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114256 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114257 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_0
  simp only [← c114252, ← c114253, ← c114254] at e_1871_0
  have e_1871_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_1
  simp only [← c114255, ← c114256, k2, ← c114257] at e_1871_1
  have hr := e_1871_1
  simp only [e_1871_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6572)) := by
  have c114258 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114259 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114260 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114261 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114262 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114263 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_2
  simp only [← c114258, ← c114259, ← c114260] at e_1871_2
  have e_1871_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_3
  simp only [← c114261, ← c114262, k2, ← c114263] at e_1871_3
  have hr := e_1871_3
  simp only [e_1871_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6573)) := by
  have c114264 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114265 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114266 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114267 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114268 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114269 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_4
  simp only [← c114264, ← c114265, ← c114266] at e_1871_4
  have e_1871_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_5
  simp only [← c114267, ← c114268, k2, ← c114269] at e_1871_5
  have hr := e_1871_5
  simp only [e_1871_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6574)) := by
  have c114270 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114271 := (publicBatchWrapper89_copies3570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114272 := (publicBatchWrapper89_copies3571 a h).1
  have c114273 := (publicBatchWrapper89_copies3571 a h).2.1
  have c114274 := (publicBatchWrapper89_copies3571 a h).2.2.1
  have c114275 := (publicBatchWrapper89_copies3571 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_6
  simp only [← c114270, ← c114271, ← c114272] at e_1871_6
  have e_1871_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_7
  simp only [← c114273, ← c114274, k2, ← c114275] at e_1871_7
  have hr := e_1871_7
  simp only [e_1871_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6575)) := by
  have c114276 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.1
  have c114277 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.1
  have c114278 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.1
  have c114279 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.1
  have c114280 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.1
  have c114281 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_8
  simp only [← c114276, ← c114277, ← c114278] at e_1871_8
  have e_1871_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_9
  simp only [← c114279, ← c114280, k2, ← c114281] at e_1871_9
  have hr := e_1871_9
  simp only [e_1871_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6576)) := by
  have c114282 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114283 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114284 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114285 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114286 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114287 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_10
  simp only [← c114282, ← c114283, ← c114284] at e_1871_10
  have e_1871_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_11
  simp only [← c114285, ← c114286, k2, ← c114287] at e_1871_11
  have hr := e_1871_11
  simp only [e_1871_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6577)) := by
  have c114288 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114289 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114290 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114291 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114292 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114293 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_12
  simp only [← c114288, ← c114289, ← c114290] at e_1871_12
  have e_1871_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_13
  simp only [← c114291, ← c114292, k2, ← c114293] at e_1871_13
  have hr := e_1871_13
  simp only [e_1871_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6578)) := by
  have c114294 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114295 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114296 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114297 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114298 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114299 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_14
  simp only [← c114294, ← c114295, ← c114296] at e_1871_14
  have e_1871_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_15
  simp only [← c114297, ← c114298, k2, ← c114299] at e_1871_15
  have hr := e_1871_15
  simp only [e_1871_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6579)) := by
  have c114300 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114301 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114302 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114303 := (publicBatchWrapper89_copies3571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114304 := (publicBatchWrapper89_copies3572 a h).1
  have c114305 := (publicBatchWrapper89_copies3572 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_16
  simp only [← c114300, ← c114301, ← c114302] at e_1871_16
  have e_1871_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_17
  simp only [← c114303, ← c114304, k2, ← c114305] at e_1871_17
  have hr := e_1871_17
  simp only [e_1871_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1871 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6580)) := by
  have c114306 := (publicBatchWrapper89_copies3572 a h).2.2.1
  have c114307 := (publicBatchWrapper89_copies3572 a h).2.2.2.1
  have c114308 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.1
  have c114309 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.1
  have c114310 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.1
  have c114311 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1871_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_18
  simp only [← c114306, ← c114307, ← c114308] at e_1871_18
  have e_1871_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1871 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_19
  simp only [← c114309, ← c114310, k2, ← c114311] at e_1871_19
  have hr := e_1871_19
  simp only [e_1871_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6581)) := by
  have c114312 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.1
  have c114313 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.1
  have c114314 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114315 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114316 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114317 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_0
  simp only [← c114312, ← c114313, ← c114314] at e_1872_0
  have e_1872_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_1
  simp only [← c114315, ← c114316, k2, ← c114317] at e_1872_1
  have hr := e_1872_1
  simp only [e_1872_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6582)) := by
  have c114318 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114319 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114320 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114321 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114322 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114323 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_2
  simp only [← c114318, ← c114319, ← c114320] at e_1872_2
  have e_1872_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_3
  simp only [← c114321, ← c114322, k2, ← c114323] at e_1872_3
  have hr := e_1872_3
  simp only [e_1872_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6583)) := by
  have c114324 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114325 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114326 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114327 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114328 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114329 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_4
  simp only [← c114324, ← c114325, ← c114326] at e_1872_4
  have e_1872_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_5
  simp only [← c114327, ← c114328, k2, ← c114329] at e_1872_5
  have hr := e_1872_5
  simp only [e_1872_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6584)) := by
  have c114330 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114331 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114332 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114333 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114334 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114335 := (publicBatchWrapper89_copies3572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_6
  simp only [← c114330, ← c114331, ← c114332] at e_1872_6
  have e_1872_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_7
  simp only [← c114333, ← c114334, k2, ← c114335] at e_1872_7
  have hr := e_1872_7
  simp only [e_1872_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6585)) := by
  have c114336 := (publicBatchWrapper89_copies3573 a h).1
  have c114337 := (publicBatchWrapper89_copies3573 a h).2.1
  have c114338 := (publicBatchWrapper89_copies3573 a h).2.2.1
  have c114339 := (publicBatchWrapper89_copies3573 a h).2.2.2.1
  have c114340 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.1
  have c114341 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_8
  simp only [← c114336, ← c114337, ← c114338] at e_1872_8
  have e_1872_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_9
  simp only [← c114339, ← c114340, k2, ← c114341] at e_1872_9
  have hr := e_1872_9
  simp only [e_1872_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6586)) := by
  have c114342 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.1
  have c114343 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.1
  have c114344 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.1
  have c114345 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.1
  have c114346 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114347 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_10
  simp only [← c114342, ← c114343, ← c114344] at e_1872_10
  have e_1872_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_11
  simp only [← c114345, ← c114346, k2, ← c114347] at e_1872_11
  have hr := e_1872_11
  simp only [e_1872_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6587)) := by
  have c114348 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114349 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114350 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114351 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114352 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114353 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_12
  simp only [← c114348, ← c114349, ← c114350] at e_1872_12
  have e_1872_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_13
  simp only [← c114351, ← c114352, k2, ← c114353] at e_1872_13
  have hr := e_1872_13
  simp only [e_1872_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6588)) := by
  have c114354 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114355 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114356 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114357 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114358 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114359 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_14
  simp only [← c114354, ← c114355, ← c114356] at e_1872_14
  have e_1872_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_15
  simp only [← c114357, ← c114358, k2, ← c114359] at e_1872_15
  have hr := e_1872_15
  simp only [e_1872_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6589)) := by
  have c114360 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114361 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114362 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114363 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114364 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114365 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_16
  simp only [← c114360, ← c114361, ← c114362] at e_1872_16
  have e_1872_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_17
  simp only [← c114363, ← c114364, k2, ← c114365] at e_1872_17
  have hr := e_1872_17
  simp only [e_1872_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1872 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6590)) := by
  have c114366 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114367 := (publicBatchWrapper89_copies3573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114368 := (publicBatchWrapper89_copies3574 a h).1
  have c114369 := (publicBatchWrapper89_copies3574 a h).2.1
  have c114370 := (publicBatchWrapper89_copies3574 a h).2.2.1
  have c114371 := (publicBatchWrapper89_copies3574 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1872_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_18
  simp only [← c114366, ← c114367, ← c114368] at e_1872_18
  have e_1872_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1872 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_19
  simp only [← c114369, ← c114370, k2, ← c114371] at e_1872_19
  have hr := e_1872_19
  simp only [e_1872_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6591)) := by
  have c114372 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.1
  have c114373 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.1
  have c114374 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.1
  have c114375 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.1
  have c114376 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.1
  have c114377 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_0
  simp only [← c114372, ← c114373, ← c114374] at e_1873_0
  have e_1873_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_1
  simp only [← c114375, ← c114376, k2, ← c114377] at e_1873_1
  have hr := e_1873_1
  simp only [e_1873_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 15) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6592)) := by
  have c114378 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114379 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114380 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114381 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114382 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114383 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_2
  simp only [← c114378, ← c114379, ← c114380] at e_1873_2
  have e_1873_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_3
  simp only [← c114381, ← c114382, k2, ← c114383] at e_1873_3
  have hr := e_1873_3
  simp only [e_1873_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 23) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6593)) := by
  have c114384 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114385 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114386 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114387 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114388 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114389 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_4
  simp only [← c114384, ← c114385, ← c114386] at e_1873_4
  have e_1873_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_5
  simp only [← c114387, ← c114388, k2, ← c114389] at e_1873_5
  have hr := e_1873_5
  simp only [e_1873_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 31) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6594)) := by
  have c114390 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114391 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114392 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114393 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114394 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114395 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_6
  simp only [← c114390, ← c114391, ← c114392] at e_1873_6
  have e_1873_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_7
  simp only [← c114393, ← c114394, k2, ← c114395] at e_1873_7
  have hr := e_1873_7
  simp only [e_1873_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 39) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6595)) := by
  have c114396 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114397 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114398 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114399 := (publicBatchWrapper89_copies3574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114400 := (publicBatchWrapper89_copies3575 a h).1
  have c114401 := (publicBatchWrapper89_copies3575 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_8
  simp only [← c114396, ← c114397, ← c114398] at e_1873_8
  have e_1873_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_9
  simp only [← c114399, ← c114400, k2, ← c114401] at e_1873_9
  have hr := e_1873_9
  simp only [e_1873_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 47) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6596)) := by
  have c114402 := (publicBatchWrapper89_copies3575 a h).2.2.1
  have c114403 := (publicBatchWrapper89_copies3575 a h).2.2.2.1
  have c114404 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.1
  have c114405 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.1
  have c114406 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.1
  have c114407 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_10
  simp only [← c114402, ← c114403, ← c114404] at e_1873_10
  have e_1873_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_11
  simp only [← c114405, ← c114406, k2, ← c114407] at e_1873_11
  have hr := e_1873_11
  simp only [e_1873_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 55) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6597)) := by
  have c114408 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.1
  have c114409 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.1
  have c114410 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114411 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114412 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114413 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_12
  simp only [← c114408, ← c114409, ← c114410] at e_1873_12
  have e_1873_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_13
  simp only [← c114411, ← c114412, k2, ← c114413] at e_1873_13
  have hr := e_1873_13
  simp only [e_1873_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 63) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6598)) := by
  have c114414 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114415 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114416 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114417 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114418 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114419 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_14
  simp only [← c114414, ← c114415, ← c114416] at e_1873_14
  have e_1873_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_15
  simp only [← c114417, ← c114418, k2, ← c114419] at e_1873_15
  have hr := e_1873_15
  simp only [e_1873_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 71) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6599)) := by
  have c114420 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114421 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114422 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114423 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114424 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114425 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_16
  simp only [← c114420, ← c114421, ← c114422] at e_1873_16
  have e_1873_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_17
  simp only [← c114423, ← c114424, k2, ← c114425] at e_1873_17
  have hr := e_1873_17
  simp only [e_1873_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1873 79) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6600)) := by
  have c114426 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114427 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114428 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114429 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114430 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114431 := (publicBatchWrapper89_copies3575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1873_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_18
  simp only [← c114426, ← c114427, ← c114428] at e_1873_18
  have e_1873_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1873 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_19
  simp only [← c114429, ← c114430, k2, ← c114431] at e_1873_19
  have hr := e_1873_19
  simp only [e_1873_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 7) = bselect (a (.wire 33 27)) (a (.virt 200)) (a (.virt 6601)) := by
  have c114432 := (publicBatchWrapper89_copies3576 a h).1
  have c114433 := (publicBatchWrapper89_copies3576 a h).2.1
  have c114434 := (publicBatchWrapper89_copies3576 a h).2.2.1
  have c114435 := (publicBatchWrapper89_copies3576 a h).2.2.2.1
  have c114436 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.1
  have c114437 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_0
  simp only [← c114432, ← c114433, ← c114434] at e_1874_0
  have e_1874_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_1
  simp only [← c114435, ← c114436, k2, ← c114437] at e_1874_1
  have hr := e_1874_1
  simp only [e_1874_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6770)) := by
  have c114438 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.1
  have c114439 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.1
  have c114440 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.1
  have c114441 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.1
  have c114442 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114443 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_2
  simp only [← c114438, ← c114439, ← c114440] at e_1874_2
  have e_1874_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_3
  simp only [← c114441, ← c114442, k2, ← c114443] at e_1874_3
  have hr := e_1874_3
  simp only [e_1874_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6771)) := by
  have c114444 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114445 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114446 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114447 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114448 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114449 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_4
  simp only [← c114444, ← c114445, ← c114446] at e_1874_4
  have e_1874_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_5
  simp only [← c114447, ← c114448, k2, ← c114449] at e_1874_5
  have hr := e_1874_5
  simp only [e_1874_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6772)) := by
  have c114450 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114451 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114452 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114453 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114454 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114455 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_6
  simp only [← c114450, ← c114451, ← c114452] at e_1874_6
  have e_1874_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_7
  simp only [← c114453, ← c114454, k2, ← c114455] at e_1874_7
  have hr := e_1874_7
  simp only [e_1874_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6773)) := by
  have c114456 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114457 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114458 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114459 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114460 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114461 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_8
  simp only [← c114456, ← c114457, ← c114458] at e_1874_8
  have e_1874_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_9
  simp only [← c114459, ← c114460, k2, ← c114461] at e_1874_9
  have hr := e_1874_9
  simp only [e_1874_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6774)) := by
  have c114462 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114463 := (publicBatchWrapper89_copies3576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114464 := (publicBatchWrapper89_copies3577 a h).1
  have c114465 := (publicBatchWrapper89_copies3577 a h).2.1
  have c114466 := (publicBatchWrapper89_copies3577 a h).2.2.1
  have c114467 := (publicBatchWrapper89_copies3577 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_10
  simp only [← c114462, ← c114463, ← c114464] at e_1874_10
  have e_1874_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_11
  simp only [← c114465, ← c114466, k2, ← c114467] at e_1874_11
  have hr := e_1874_11
  simp only [e_1874_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6775)) := by
  have c114468 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.1
  have c114469 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.1
  have c114470 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.1
  have c114471 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.1
  have c114472 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.1
  have c114473 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_12
  simp only [← c114468, ← c114469, ← c114470] at e_1874_12
  have e_1874_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_13
  simp only [← c114471, ← c114472, k2, ← c114473] at e_1874_13
  have hr := e_1874_13
  simp only [e_1874_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6776)) := by
  have c114474 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114475 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114476 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114477 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114478 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114479 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_14
  simp only [← c114474, ← c114475, ← c114476] at e_1874_14
  have e_1874_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_15
  simp only [← c114477, ← c114478, k2, ← c114479] at e_1874_15
  have hr := e_1874_15
  simp only [e_1874_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6777)) := by
  have c114480 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114481 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114482 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114483 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114484 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114485 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_16
  simp only [← c114480, ← c114481, ← c114482] at e_1874_16
  have e_1874_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_17
  simp only [← c114483, ← c114484, k2, ← c114485] at e_1874_17
  have hr := e_1874_17
  simp only [e_1874_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1874 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6778)) := by
  have c114486 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114487 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114488 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114489 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114490 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114491 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1874_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_18
  simp only [← c114486, ← c114487, ← c114488] at e_1874_18
  have e_1874_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1874 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_19
  simp only [← c114489, ← c114490, k2, ← c114491] at e_1874_19
  have hr := e_1874_19
  simp only [e_1874_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6779)) := by
  have c114492 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114493 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114494 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114495 := (publicBatchWrapper89_copies3577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114496 := (publicBatchWrapper89_copies3578 a h).1
  have c114497 := (publicBatchWrapper89_copies3578 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_0
  simp only [← c114492, ← c114493, ← c114494] at e_1875_0
  have e_1875_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_1
  simp only [← c114495, ← c114496, k2, ← c114497] at e_1875_1
  have hr := e_1875_1
  simp only [e_1875_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6780)) := by
  have c114498 := (publicBatchWrapper89_copies3578 a h).2.2.1
  have c114499 := (publicBatchWrapper89_copies3578 a h).2.2.2.1
  have c114500 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.1
  have c114501 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.1
  have c114502 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.1
  have c114503 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_2
  simp only [← c114498, ← c114499, ← c114500] at e_1875_2
  have e_1875_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_3
  simp only [← c114501, ← c114502, k2, ← c114503] at e_1875_3
  have hr := e_1875_3
  simp only [e_1875_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6781)) := by
  have c114504 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.1
  have c114505 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.1
  have c114506 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114507 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114508 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114509 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_4
  simp only [← c114504, ← c114505, ← c114506] at e_1875_4
  have e_1875_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_5
  simp only [← c114507, ← c114508, k2, ← c114509] at e_1875_5
  have hr := e_1875_5
  simp only [e_1875_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6782)) := by
  have c114510 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114511 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114512 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114513 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114514 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114515 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_6
  simp only [← c114510, ← c114511, ← c114512] at e_1875_6
  have e_1875_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_7
  simp only [← c114513, ← c114514, k2, ← c114515] at e_1875_7
  have hr := e_1875_7
  simp only [e_1875_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6783)) := by
  have c114516 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114517 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114518 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114519 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114520 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114521 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_8
  simp only [← c114516, ← c114517, ← c114518] at e_1875_8
  have e_1875_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_9
  simp only [← c114519, ← c114520, k2, ← c114521] at e_1875_9
  have hr := e_1875_9
  simp only [e_1875_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6784)) := by
  have c114522 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114523 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114524 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114525 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114526 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114527 := (publicBatchWrapper89_copies3578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_10
  simp only [← c114522, ← c114523, ← c114524] at e_1875_10
  have e_1875_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_11
  simp only [← c114525, ← c114526, k2, ← c114527] at e_1875_11
  have hr := e_1875_11
  simp only [e_1875_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6785)) := by
  have c114528 := (publicBatchWrapper89_copies3579 a h).1
  have c114529 := (publicBatchWrapper89_copies3579 a h).2.1
  have c114530 := (publicBatchWrapper89_copies3579 a h).2.2.1
  have c114531 := (publicBatchWrapper89_copies3579 a h).2.2.2.1
  have c114532 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.1
  have c114533 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_12
  simp only [← c114528, ← c114529, ← c114530] at e_1875_12
  have e_1875_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_13
  simp only [← c114531, ← c114532, k2, ← c114533] at e_1875_13
  have hr := e_1875_13
  simp only [e_1875_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6786)) := by
  have c114534 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.1
  have c114535 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.1
  have c114536 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.1
  have c114537 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.1
  have c114538 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114539 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_14
  simp only [← c114534, ← c114535, ← c114536] at e_1875_14
  have e_1875_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_15
  simp only [← c114537, ← c114538, k2, ← c114539] at e_1875_15
  have hr := e_1875_15
  simp only [e_1875_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6787)) := by
  have c114540 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114541 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114542 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114543 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114544 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114545 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_16
  simp only [← c114540, ← c114541, ← c114542] at e_1875_16
  have e_1875_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_17
  simp only [← c114543, ← c114544, k2, ← c114545] at e_1875_17
  have hr := e_1875_17
  simp only [e_1875_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1875 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6788)) := by
  have c114546 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114547 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114548 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114549 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114550 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114551 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1875_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_18
  simp only [← c114546, ← c114547, ← c114548] at e_1875_18
  have e_1875_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1875 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_19
  simp only [← c114549, ← c114550, k2, ← c114551] at e_1875_19
  have hr := e_1875_19
  simp only [e_1875_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6789)) := by
  have c114552 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114553 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114554 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114555 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114556 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114557 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_0
  simp only [← c114552, ← c114553, ← c114554] at e_1876_0
  have e_1876_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_1
  simp only [← c114555, ← c114556, k2, ← c114557] at e_1876_1
  have hr := e_1876_1
  simp only [e_1876_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6790)) := by
  have c114558 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114559 := (publicBatchWrapper89_copies3579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114560 := (publicBatchWrapper89_copies3580 a h).1
  have c114561 := (publicBatchWrapper89_copies3580 a h).2.1
  have c114562 := (publicBatchWrapper89_copies3580 a h).2.2.1
  have c114563 := (publicBatchWrapper89_copies3580 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_2
  simp only [← c114558, ← c114559, ← c114560] at e_1876_2
  have e_1876_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_3
  simp only [← c114561, ← c114562, k2, ← c114563] at e_1876_3
  have hr := e_1876_3
  simp only [e_1876_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6791)) := by
  have c114564 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.1
  have c114565 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.1
  have c114566 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.1
  have c114567 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.1
  have c114568 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.1
  have c114569 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_4
  simp only [← c114564, ← c114565, ← c114566] at e_1876_4
  have e_1876_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_5
  simp only [← c114567, ← c114568, k2, ← c114569] at e_1876_5
  have hr := e_1876_5
  simp only [e_1876_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 31) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6792)) := by
  have c114570 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114571 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114572 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114573 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114574 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114575 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_6
  simp only [← c114570, ← c114571, ← c114572] at e_1876_6
  have e_1876_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_7
  simp only [← c114573, ← c114574, k2, ← c114575] at e_1876_7
  have hr := e_1876_7
  simp only [e_1876_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 39) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6793)) := by
  have c114576 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114577 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114578 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114579 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114580 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114581 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_8
  simp only [← c114576, ← c114577, ← c114578] at e_1876_8
  have e_1876_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_9
  simp only [← c114579, ← c114580, k2, ← c114581] at e_1876_9
  have hr := e_1876_9
  simp only [e_1876_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 47) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6794)) := by
  have c114582 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114583 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114584 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114585 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114586 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114587 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_10
  simp only [← c114582, ← c114583, ← c114584] at e_1876_10
  have e_1876_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_11
  simp only [← c114585, ← c114586, k2, ← c114587] at e_1876_11
  have hr := e_1876_11
  simp only [e_1876_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 55) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6795)) := by
  have c114588 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114589 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114590 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114591 := (publicBatchWrapper89_copies3580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114592 := (publicBatchWrapper89_copies3581 a h).1
  have c114593 := (publicBatchWrapper89_copies3581 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_12
  simp only [← c114588, ← c114589, ← c114590] at e_1876_12
  have e_1876_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_13
  simp only [← c114591, ← c114592, k2, ← c114593] at e_1876_13
  have hr := e_1876_13
  simp only [e_1876_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 63) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6796)) := by
  have c114594 := (publicBatchWrapper89_copies3581 a h).2.2.1
  have c114595 := (publicBatchWrapper89_copies3581 a h).2.2.2.1
  have c114596 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.1
  have c114597 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.1
  have c114598 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.1
  have c114599 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_14
  simp only [← c114594, ← c114595, ← c114596] at e_1876_14
  have e_1876_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_15
  simp only [← c114597, ← c114598, k2, ← c114599] at e_1876_15
  have hr := e_1876_15
  simp only [e_1876_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 71) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6797)) := by
  have c114600 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.1
  have c114601 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.1
  have c114602 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114603 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114604 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114605 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_16
  simp only [← c114600, ← c114601, ← c114602] at e_1876_16
  have e_1876_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_17
  simp only [← c114603, ← c114604, k2, ← c114605] at e_1876_17
  have hr := e_1876_17
  simp only [e_1876_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1876 79) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6798)) := by
  have c114606 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114607 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114608 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114609 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114610 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114611 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1876_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_18
  simp only [← c114606, ← c114607, ← c114608] at e_1876_18
  have e_1876_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1876 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_19
  simp only [← c114609, ← c114610, k2, ← c114611] at e_1876_19
  have hr := e_1876_19
  simp only [e_1876_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 7) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6799)) := by
  have c114612 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114613 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114614 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114615 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114616 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114617 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_0
  simp only [← c114612, ← c114613, ← c114614] at e_1877_0
  have e_1877_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_1
  simp only [← c114615, ← c114616, k2, ← c114617] at e_1877_1
  have hr := e_1877_1
  simp only [e_1877_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 15) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6800)) := by
  have c114618 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114619 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114620 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114621 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114622 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114623 := (publicBatchWrapper89_copies3581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_2
  simp only [← c114618, ← c114619, ← c114620] at e_1877_2
  have e_1877_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_3
  simp only [← c114621, ← c114622, k2, ← c114623] at e_1877_3
  have hr := e_1877_3
  simp only [e_1877_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 23) = bselect (a (.wire 33 71)) (a (.virt 200)) (a (.virt 6801)) := by
  have c114624 := (publicBatchWrapper89_copies3582 a h).1
  have c114625 := (publicBatchWrapper89_copies3582 a h).2.1
  have c114626 := (publicBatchWrapper89_copies3582 a h).2.2.1
  have c114627 := (publicBatchWrapper89_copies3582 a h).2.2.2.1
  have c114628 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.1
  have c114629 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_4
  simp only [← c114624, ← c114625, ← c114626] at e_1877_4
  have e_1877_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_5
  simp only [← c114627, ← c114628, k2, ← c114629] at e_1877_5
  have hr := e_1877_5
  simp only [e_1877_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6970)) := by
  have c114630 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.1
  have c114631 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.1
  have c114632 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.1
  have c114633 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.1
  have c114634 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114635 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_6
  simp only [← c114630, ← c114631, ← c114632] at e_1877_6
  have e_1877_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_7
  simp only [← c114633, ← c114634, k2, ← c114635] at e_1877_7
  have hr := e_1877_7
  simp only [e_1877_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6971)) := by
  have c114636 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114637 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114638 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114639 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114640 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114641 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_8
  simp only [← c114636, ← c114637, ← c114638] at e_1877_8
  have e_1877_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_9
  simp only [← c114639, ← c114640, k2, ← c114641] at e_1877_9
  have hr := e_1877_9
  simp only [e_1877_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6972)) := by
  have c114642 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114643 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114644 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114645 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114646 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114647 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_10
  simp only [← c114642, ← c114643, ← c114644] at e_1877_10
  have e_1877_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_11
  simp only [← c114645, ← c114646, k2, ← c114647] at e_1877_11
  have hr := e_1877_11
  simp only [e_1877_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6973)) := by
  have c114648 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114649 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114650 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114651 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114652 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114653 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_12
  simp only [← c114648, ← c114649, ← c114650] at e_1877_12
  have e_1877_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_13
  simp only [← c114651, ← c114652, k2, ← c114653] at e_1877_13
  have hr := e_1877_13
  simp only [e_1877_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6974)) := by
  have c114654 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114655 := (publicBatchWrapper89_copies3582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114656 := (publicBatchWrapper89_copies3583 a h).1
  have c114657 := (publicBatchWrapper89_copies3583 a h).2.1
  have c114658 := (publicBatchWrapper89_copies3583 a h).2.2.1
  have c114659 := (publicBatchWrapper89_copies3583 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_14
  simp only [← c114654, ← c114655, ← c114656] at e_1877_14
  have e_1877_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_15
  simp only [← c114657, ← c114658, k2, ← c114659] at e_1877_15
  have hr := e_1877_15
  simp only [e_1877_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6975)) := by
  have c114660 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.1
  have c114661 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.1
  have c114662 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.1
  have c114663 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.1
  have c114664 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.1
  have c114665 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_16
  simp only [← c114660, ← c114661, ← c114662] at e_1877_16
  have e_1877_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_17
  simp only [← c114663, ← c114664, k2, ← c114665] at e_1877_17
  have hr := e_1877_17
  simp only [e_1877_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1877 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6976)) := by
  have c114666 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114667 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114668 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114669 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114670 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114671 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1877_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_18
  simp only [← c114666, ← c114667, ← c114668] at e_1877_18
  have e_1877_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1877 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_19
  simp only [← c114669, ← c114670, k2, ← c114671] at e_1877_19
  have hr := e_1877_19
  simp only [e_1877_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6977)) := by
  have c114672 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114673 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114674 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114675 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114676 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114677 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_0
  simp only [← c114672, ← c114673, ← c114674] at e_1878_0
  have e_1878_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_1
  simp only [← c114675, ← c114676, k2, ← c114677] at e_1878_1
  have hr := e_1878_1
  simp only [e_1878_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6978)) := by
  have c114678 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114679 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114680 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114681 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114682 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114683 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_2
  simp only [← c114678, ← c114679, ← c114680] at e_1878_2
  have e_1878_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_3
  simp only [← c114681, ← c114682, k2, ← c114683] at e_1878_3
  have hr := e_1878_3
  simp only [e_1878_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6979)) := by
  have c114684 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114685 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114686 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114687 := (publicBatchWrapper89_copies3583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114688 := (publicBatchWrapper89_copies3584 a h).1
  have c114689 := (publicBatchWrapper89_copies3584 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_4
  simp only [← c114684, ← c114685, ← c114686] at e_1878_4
  have e_1878_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_5
  simp only [← c114687, ← c114688, k2, ← c114689] at e_1878_5
  have hr := e_1878_5
  simp only [e_1878_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6980)) := by
  have c114690 := (publicBatchWrapper89_copies3584 a h).2.2.1
  have c114691 := (publicBatchWrapper89_copies3584 a h).2.2.2.1
  have c114692 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.1
  have c114693 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.1
  have c114694 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.1
  have c114695 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_6
  simp only [← c114690, ← c114691, ← c114692] at e_1878_6
  have e_1878_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_7
  simp only [← c114693, ← c114694, k2, ← c114695] at e_1878_7
  have hr := e_1878_7
  simp only [e_1878_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6981)) := by
  have c114696 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.1
  have c114697 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.1
  have c114698 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114699 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114700 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114701 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_8
  simp only [← c114696, ← c114697, ← c114698] at e_1878_8
  have e_1878_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_9
  simp only [← c114699, ← c114700, k2, ← c114701] at e_1878_9
  have hr := e_1878_9
  simp only [e_1878_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6982)) := by
  have c114702 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114703 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114704 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114705 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114706 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114707 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_10
  simp only [← c114702, ← c114703, ← c114704] at e_1878_10
  have e_1878_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_11
  simp only [← c114705, ← c114706, k2, ← c114707] at e_1878_11
  have hr := e_1878_11
  simp only [e_1878_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6983)) := by
  have c114708 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114709 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114710 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114711 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114712 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114713 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_12
  simp only [← c114708, ← c114709, ← c114710] at e_1878_12
  have e_1878_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_13
  simp only [← c114711, ← c114712, k2, ← c114713] at e_1878_13
  have hr := e_1878_13
  simp only [e_1878_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6984)) := by
  have c114714 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114715 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114716 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114717 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114718 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114719 := (publicBatchWrapper89_copies3584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_14
  simp only [← c114714, ← c114715, ← c114716] at e_1878_14
  have e_1878_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_15
  simp only [← c114717, ← c114718, k2, ← c114719] at e_1878_15
  have hr := e_1878_15
  simp only [e_1878_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6985)) := by
  have c114720 := (publicBatchWrapper89_copies3585 a h).1
  have c114721 := (publicBatchWrapper89_copies3585 a h).2.1
  have c114722 := (publicBatchWrapper89_copies3585 a h).2.2.1
  have c114723 := (publicBatchWrapper89_copies3585 a h).2.2.2.1
  have c114724 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.1
  have c114725 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_16
  simp only [← c114720, ← c114721, ← c114722] at e_1878_16
  have e_1878_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_17
  simp only [← c114723, ← c114724, k2, ← c114725] at e_1878_17
  have hr := e_1878_17
  simp only [e_1878_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1878 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6986)) := by
  have c114726 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.1
  have c114727 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.1
  have c114728 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.1
  have c114729 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.1
  have c114730 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114731 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1878_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_18
  simp only [← c114726, ← c114727, ← c114728] at e_1878_18
  have e_1878_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1878 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_19
  simp only [← c114729, ← c114730, k2, ← c114731] at e_1878_19
  have hr := e_1878_19
  simp only [e_1878_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6987)) := by
  have c114732 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114733 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114734 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114735 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114736 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114737 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_0
  simp only [← c114732, ← c114733, ← c114734] at e_1879_0
  have e_1879_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_1
  simp only [← c114735, ← c114736, k2, ← c114737] at e_1879_1
  have hr := e_1879_1
  simp only [e_1879_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6988)) := by
  have c114738 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114739 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114740 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114741 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114742 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114743 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_2
  simp only [← c114738, ← c114739, ← c114740] at e_1879_2
  have e_1879_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_3
  simp only [← c114741, ← c114742, k2, ← c114743] at e_1879_3
  have hr := e_1879_3
  simp only [e_1879_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6989)) := by
  have c114744 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114745 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114746 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114747 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114748 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114749 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_4
  simp only [← c114744, ← c114745, ← c114746] at e_1879_4
  have e_1879_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_5
  simp only [← c114747, ← c114748, k2, ← c114749] at e_1879_5
  have hr := e_1879_5
  simp only [e_1879_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6990)) := by
  have c114750 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114751 := (publicBatchWrapper89_copies3585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114752 := (publicBatchWrapper89_copies3586 a h).1
  have c114753 := (publicBatchWrapper89_copies3586 a h).2.1
  have c114754 := (publicBatchWrapper89_copies3586 a h).2.2.1
  have c114755 := (publicBatchWrapper89_copies3586 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_6
  simp only [← c114750, ← c114751, ← c114752] at e_1879_6
  have e_1879_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_7
  simp only [← c114753, ← c114754, k2, ← c114755] at e_1879_7
  have hr := e_1879_7
  simp only [e_1879_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6991)) := by
  have c114756 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.1
  have c114757 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.1
  have c114758 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.1
  have c114759 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.1
  have c114760 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.1
  have c114761 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_8
  simp only [← c114756, ← c114757, ← c114758] at e_1879_8
  have e_1879_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_9
  simp only [← c114759, ← c114760, k2, ← c114761] at e_1879_9
  have hr := e_1879_9
  simp only [e_1879_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 47) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6992)) := by
  have c114762 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114763 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114764 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114765 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114766 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114767 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_10
  simp only [← c114762, ← c114763, ← c114764] at e_1879_10
  have e_1879_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_11
  simp only [← c114765, ← c114766, k2, ← c114767] at e_1879_11
  have hr := e_1879_11
  simp only [e_1879_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 55) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6993)) := by
  have c114768 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114769 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114770 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114771 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114772 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114773 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_12
  simp only [← c114768, ← c114769, ← c114770] at e_1879_12
  have e_1879_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_13
  simp only [← c114771, ← c114772, k2, ← c114773] at e_1879_13
  have hr := e_1879_13
  simp only [e_1879_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 63) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6994)) := by
  have c114774 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114775 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114776 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114777 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114778 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114779 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_14
  simp only [← c114774, ← c114775, ← c114776] at e_1879_14
  have e_1879_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_15
  simp only [← c114777, ← c114778, k2, ← c114779] at e_1879_15
  have hr := e_1879_15
  simp only [e_1879_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 71) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6995)) := by
  have c114780 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114781 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114782 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114783 := (publicBatchWrapper89_copies3586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114784 := (publicBatchWrapper89_copies3587 a h).1
  have c114785 := (publicBatchWrapper89_copies3587 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_16
  simp only [← c114780, ← c114781, ← c114782] at e_1879_16
  have e_1879_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_17
  simp only [← c114783, ← c114784, k2, ← c114785] at e_1879_17
  have hr := e_1879_17
  simp only [e_1879_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1879 79) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6996)) := by
  have c114786 := (publicBatchWrapper89_copies3587 a h).2.2.1
  have c114787 := (publicBatchWrapper89_copies3587 a h).2.2.2.1
  have c114788 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.1
  have c114789 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.1
  have c114790 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.1
  have c114791 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1879_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_18
  simp only [← c114786, ← c114787, ← c114788] at e_1879_18
  have e_1879_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1879 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_19
  simp only [← c114789, ← c114790, k2, ← c114791] at e_1879_19
  have hr := e_1879_19
  simp only [e_1879_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 7) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6997)) := by
  have c114792 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.1
  have c114793 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.1
  have c114794 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114795 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114796 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114797 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_0
  simp only [← c114792, ← c114793, ← c114794] at e_1880_0
  have e_1880_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_1
  simp only [← c114795, ← c114796, k2, ← c114797] at e_1880_1
  have hr := e_1880_1
  simp only [e_1880_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 15) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6998)) := by
  have c114798 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114799 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114800 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114801 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114802 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114803 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_2
  simp only [← c114798, ← c114799, ← c114800] at e_1880_2
  have e_1880_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_3
  simp only [← c114801, ← c114802, k2, ← c114803] at e_1880_3
  have hr := e_1880_3
  simp only [e_1880_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 23) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 6999)) := by
  have c114804 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114805 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114806 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114807 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114808 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114809 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_4
  simp only [← c114804, ← c114805, ← c114806] at e_1880_4
  have e_1880_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_5
  simp only [← c114807, ← c114808, k2, ← c114809] at e_1880_5
  have hr := e_1880_5
  simp only [e_1880_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 31) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 7000)) := by
  have c114810 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114811 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114812 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114813 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114814 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114815 := (publicBatchWrapper89_copies3587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_6
  simp only [← c114810, ← c114811, ← c114812] at e_1880_6
  have e_1880_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_7
  simp only [← c114813, ← c114814, k2, ← c114815] at e_1880_7
  have hr := e_1880_7
  simp only [e_1880_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 39) = bselect (a (.wire 34 35)) (a (.virt 200)) (a (.virt 7001)) := by
  have c114816 := (publicBatchWrapper89_copies3588 a h).1
  have c114817 := (publicBatchWrapper89_copies3588 a h).2.1
  have c114818 := (publicBatchWrapper89_copies3588 a h).2.2.1
  have c114819 := (publicBatchWrapper89_copies3588 a h).2.2.2.1
  have c114820 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.1
  have c114821 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_8
  simp only [← c114816, ← c114817, ← c114818] at e_1880_8
  have e_1880_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_9
  simp only [← c114819, ← c114820, k2, ← c114821] at e_1880_9
  have hr := e_1880_9
  simp only [e_1880_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7170)) := by
  have c114822 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.1
  have c114823 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.1
  have c114824 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.1
  have c114825 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.1
  have c114826 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114827 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_10
  simp only [← c114822, ← c114823, ← c114824] at e_1880_10
  have e_1880_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_11
  simp only [← c114825, ← c114826, k2, ← c114827] at e_1880_11
  have hr := e_1880_11
  simp only [e_1880_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7171)) := by
  have c114828 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114829 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114830 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114831 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114832 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114833 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_12
  simp only [← c114828, ← c114829, ← c114830] at e_1880_12
  have e_1880_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_13
  simp only [← c114831, ← c114832, k2, ← c114833] at e_1880_13
  have hr := e_1880_13
  simp only [e_1880_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7172)) := by
  have c114834 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114835 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114836 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114837 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114838 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114839 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_14
  simp only [← c114834, ← c114835, ← c114836] at e_1880_14
  have e_1880_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_15
  simp only [← c114837, ← c114838, k2, ← c114839] at e_1880_15
  have hr := e_1880_15
  simp only [e_1880_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7173)) := by
  have c114840 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114841 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114842 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114843 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114844 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114845 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_16
  simp only [← c114840, ← c114841, ← c114842] at e_1880_16
  have e_1880_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_17
  simp only [← c114843, ← c114844, k2, ← c114845] at e_1880_17
  have hr := e_1880_17
  simp only [e_1880_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1880 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7174)) := by
  have c114846 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114847 := (publicBatchWrapper89_copies3588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114848 := (publicBatchWrapper89_copies3589 a h).1
  have c114849 := (publicBatchWrapper89_copies3589 a h).2.1
  have c114850 := (publicBatchWrapper89_copies3589 a h).2.2.1
  have c114851 := (publicBatchWrapper89_copies3589 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1880_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_18
  simp only [← c114846, ← c114847, ← c114848] at e_1880_18
  have e_1880_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1880 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_19
  simp only [← c114849, ← c114850, k2, ← c114851] at e_1880_19
  have hr := e_1880_19
  simp only [e_1880_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7175)) := by
  have c114852 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.1
  have c114853 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.1
  have c114854 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.1
  have c114855 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.1
  have c114856 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.1
  have c114857 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_0
  simp only [← c114852, ← c114853, ← c114854] at e_1881_0
  have e_1881_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_1
  simp only [← c114855, ← c114856, k2, ← c114857] at e_1881_1
  have hr := e_1881_1
  simp only [e_1881_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7176)) := by
  have c114858 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114859 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114860 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114861 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114862 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114863 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_2
  simp only [← c114858, ← c114859, ← c114860] at e_1881_2
  have e_1881_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_3
  simp only [← c114861, ← c114862, k2, ← c114863] at e_1881_3
  have hr := e_1881_3
  simp only [e_1881_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7177)) := by
  have c114864 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114865 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114866 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114867 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114868 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114869 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_4
  simp only [← c114864, ← c114865, ← c114866] at e_1881_4
  have e_1881_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_5
  simp only [← c114867, ← c114868, k2, ← c114869] at e_1881_5
  have hr := e_1881_5
  simp only [e_1881_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7178)) := by
  have c114870 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114871 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114872 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114873 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114874 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114875 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_6
  simp only [← c114870, ← c114871, ← c114872] at e_1881_6
  have e_1881_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_7
  simp only [← c114873, ← c114874, k2, ← c114875] at e_1881_7
  have hr := e_1881_7
  simp only [e_1881_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7179)) := by
  have c114876 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114877 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114878 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114879 := (publicBatchWrapper89_copies3589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114880 := (publicBatchWrapper89_copies3590 a h).1
  have c114881 := (publicBatchWrapper89_copies3590 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_8
  simp only [← c114876, ← c114877, ← c114878] at e_1881_8
  have e_1881_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_9
  simp only [← c114879, ← c114880, k2, ← c114881] at e_1881_9
  have hr := e_1881_9
  simp only [e_1881_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7180)) := by
  have c114882 := (publicBatchWrapper89_copies3590 a h).2.2.1
  have c114883 := (publicBatchWrapper89_copies3590 a h).2.2.2.1
  have c114884 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.1
  have c114885 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.1
  have c114886 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.1
  have c114887 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_10
  simp only [← c114882, ← c114883, ← c114884] at e_1881_10
  have e_1881_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_11
  simp only [← c114885, ← c114886, k2, ← c114887] at e_1881_11
  have hr := e_1881_11
  simp only [e_1881_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7181)) := by
  have c114888 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.1
  have c114889 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.1
  have c114890 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114891 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114892 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114893 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_12
  simp only [← c114888, ← c114889, ← c114890] at e_1881_12
  have e_1881_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_13
  simp only [← c114891, ← c114892, k2, ← c114893] at e_1881_13
  have hr := e_1881_13
  simp only [e_1881_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7182)) := by
  have c114894 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114895 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114896 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114897 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114898 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114899 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_14
  simp only [← c114894, ← c114895, ← c114896] at e_1881_14
  have e_1881_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_15
  simp only [← c114897, ← c114898, k2, ← c114899] at e_1881_15
  have hr := e_1881_15
  simp only [e_1881_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7183)) := by
  have c114900 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114901 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114902 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114903 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114904 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114905 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_16
  simp only [← c114900, ← c114901, ← c114902] at e_1881_16
  have e_1881_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_17
  simp only [← c114903, ← c114904, k2, ← c114905] at e_1881_17
  have hr := e_1881_17
  simp only [e_1881_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1881 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7184)) := by
  have c114906 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114907 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114908 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114909 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114910 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114911 := (publicBatchWrapper89_copies3590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1881_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_18
  simp only [← c114906, ← c114907, ← c114908] at e_1881_18
  have e_1881_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1881 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_19
  simp only [← c114909, ← c114910, k2, ← c114911] at e_1881_19
  have hr := e_1881_19
  simp only [e_1881_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7185)) := by
  have c114912 := (publicBatchWrapper89_copies3591 a h).1
  have c114913 := (publicBatchWrapper89_copies3591 a h).2.1
  have c114914 := (publicBatchWrapper89_copies3591 a h).2.2.1
  have c114915 := (publicBatchWrapper89_copies3591 a h).2.2.2.1
  have c114916 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.1
  have c114917 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_0
  simp only [← c114912, ← c114913, ← c114914] at e_1882_0
  have e_1882_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_1
  simp only [← c114915, ← c114916, k2, ← c114917] at e_1882_1
  have hr := e_1882_1
  simp only [e_1882_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7186)) := by
  have c114918 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.1
  have c114919 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.1
  have c114920 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.1
  have c114921 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.1
  have c114922 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114923 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_2
  simp only [← c114918, ← c114919, ← c114920] at e_1882_2
  have e_1882_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_3
  simp only [← c114921, ← c114922, k2, ← c114923] at e_1882_3
  have hr := e_1882_3
  simp only [e_1882_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7187)) := by
  have c114924 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114925 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114926 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114927 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114928 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114929 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_4
  simp only [← c114924, ← c114925, ← c114926] at e_1882_4
  have e_1882_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_5
  simp only [← c114927, ← c114928, k2, ← c114929] at e_1882_5
  have hr := e_1882_5
  simp only [e_1882_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7188)) := by
  have c114930 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114931 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114932 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114933 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114934 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114935 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_6
  simp only [← c114930, ← c114931, ← c114932] at e_1882_6
  have e_1882_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_7
  simp only [← c114933, ← c114934, k2, ← c114935] at e_1882_7
  have hr := e_1882_7
  simp only [e_1882_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7189)) := by
  have c114936 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114937 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114938 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114939 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114940 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114941 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_8
  simp only [← c114936, ← c114937, ← c114938] at e_1882_8
  have e_1882_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_9
  simp only [← c114939, ← c114940, k2, ← c114941] at e_1882_9
  have hr := e_1882_9
  simp only [e_1882_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7190)) := by
  have c114942 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114943 := (publicBatchWrapper89_copies3591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114944 := (publicBatchWrapper89_copies3592 a h).1
  have c114945 := (publicBatchWrapper89_copies3592 a h).2.1
  have c114946 := (publicBatchWrapper89_copies3592 a h).2.2.1
  have c114947 := (publicBatchWrapper89_copies3592 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_10
  simp only [← c114942, ← c114943, ← c114944] at e_1882_10
  have e_1882_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_11
  simp only [← c114945, ← c114946, k2, ← c114947] at e_1882_11
  have hr := e_1882_11
  simp only [e_1882_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7191)) := by
  have c114948 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.1
  have c114949 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.1
  have c114950 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.1
  have c114951 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.1
  have c114952 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.1
  have c114953 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_12
  simp only [← c114948, ← c114949, ← c114950] at e_1882_12
  have e_1882_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_13
  simp only [← c114951, ← c114952, k2, ← c114953] at e_1882_13
  have hr := e_1882_13
  simp only [e_1882_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 63) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7192)) := by
  have c114954 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114955 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114956 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114957 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114958 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114959 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_14
  simp only [← c114954, ← c114955, ← c114956] at e_1882_14
  have e_1882_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_15
  simp only [← c114957, ← c114958, k2, ← c114959] at e_1882_15
  have hr := e_1882_15
  simp only [e_1882_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 71) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7193)) := by
  have c114960 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114961 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114962 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114963 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114964 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114965 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_16
  simp only [← c114960, ← c114961, ← c114962] at e_1882_16
  have e_1882_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_17
  simp only [← c114963, ← c114964, k2, ← c114965] at e_1882_17
  have hr := e_1882_17
  simp only [e_1882_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1882 79) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7194)) := by
  have c114966 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114967 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114968 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114969 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114970 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114971 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1882_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_18
  simp only [← c114966, ← c114967, ← c114968] at e_1882_18
  have e_1882_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1882 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_19
  simp only [← c114969, ← c114970, k2, ← c114971] at e_1882_19
  have hr := e_1882_19
  simp only [e_1882_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 7) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7195)) := by
  have c114972 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114973 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114974 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114975 := (publicBatchWrapper89_copies3592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c114976 := (publicBatchWrapper89_copies3593 a h).1
  have c114977 := (publicBatchWrapper89_copies3593 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_0
  simp only [← c114972, ← c114973, ← c114974] at e_1883_0
  have e_1883_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_1
  simp only [← c114975, ← c114976, k2, ← c114977] at e_1883_1
  have hr := e_1883_1
  simp only [e_1883_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 15) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7196)) := by
  have c114978 := (publicBatchWrapper89_copies3593 a h).2.2.1
  have c114979 := (publicBatchWrapper89_copies3593 a h).2.2.2.1
  have c114980 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.1
  have c114981 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.1
  have c114982 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.1
  have c114983 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_2
  simp only [← c114978, ← c114979, ← c114980] at e_1883_2
  have e_1883_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_3
  simp only [← c114981, ← c114982, k2, ← c114983] at e_1883_3
  have hr := e_1883_3
  simp only [e_1883_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 23) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7197)) := by
  have c114984 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.1
  have c114985 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.1
  have c114986 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c114987 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c114988 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114989 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_4
  simp only [← c114984, ← c114985, ← c114986] at e_1883_4
  have e_1883_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_5
  simp only [← c114987, ← c114988, k2, ← c114989] at e_1883_5
  have hr := e_1883_5
  simp only [e_1883_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 31) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7198)) := by
  have c114990 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114991 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114992 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114993 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114994 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114995 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_6
  simp only [← c114990, ← c114991, ← c114992] at e_1883_6
  have e_1883_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_7
  simp only [← c114993, ← c114994, k2, ← c114995] at e_1883_7
  have hr := e_1883_7
  simp only [e_1883_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 39) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7199)) := by
  have c114996 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114997 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114998 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c114999 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115000 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115001 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_8
  simp only [← c114996, ← c114997, ← c114998] at e_1883_8
  have e_1883_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_9
  simp only [← c114999, ← c115000, k2, ← c115001] at e_1883_9
  have hr := e_1883_9
  simp only [e_1883_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 47) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7200)) := by
  have c115002 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115003 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115004 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115005 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115006 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115007 := (publicBatchWrapper89_copies3593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_10
  simp only [← c115002, ← c115003, ← c115004] at e_1883_10
  have e_1883_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_11
  simp only [← c115005, ← c115006, k2, ← c115007] at e_1883_11
  have hr := e_1883_11
  simp only [e_1883_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 55) = bselect (a (.wire 34 79)) (a (.virt 200)) (a (.virt 7201)) := by
  have c115008 := (publicBatchWrapper89_copies3594 a h).1
  have c115009 := (publicBatchWrapper89_copies3594 a h).2.1
  have c115010 := (publicBatchWrapper89_copies3594 a h).2.2.1
  have c115011 := (publicBatchWrapper89_copies3594 a h).2.2.2.1
  have c115012 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.1
  have c115013 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_12
  simp only [← c115008, ← c115009, ← c115010] at e_1883_12
  have e_1883_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_13
  simp only [← c115011, ← c115012, k2, ← c115013] at e_1883_13
  have hr := e_1883_13
  simp only [e_1883_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7370)) := by
  have c115014 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.1
  have c115015 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.1
  have c115016 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.1
  have c115017 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.1
  have c115018 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115019 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_14
  simp only [← c115014, ← c115015, ← c115016] at e_1883_14
  have e_1883_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_15
  simp only [← c115017, ← c115018, k2, ← c115019] at e_1883_15
  have hr := e_1883_15
  simp only [e_1883_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7371)) := by
  have c115020 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115021 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115022 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115023 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115024 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115025 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_16
  simp only [← c115020, ← c115021, ← c115022] at e_1883_16
  have e_1883_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_17
  simp only [← c115023, ← c115024, k2, ← c115025] at e_1883_17
  have hr := e_1883_17
  simp only [e_1883_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1883 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7372)) := by
  have c115026 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115027 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115028 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115029 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115030 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115031 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1883_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_18
  simp only [← c115026, ← c115027, ← c115028] at e_1883_18
  have e_1883_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1883 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_19
  simp only [← c115029, ← c115030, k2, ← c115031] at e_1883_19
  have hr := e_1883_19
  simp only [e_1883_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7373)) := by
  have c115032 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115033 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115034 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115035 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115036 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115037 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_0
  simp only [← c115032, ← c115033, ← c115034] at e_1884_0
  have e_1884_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_1
  simp only [← c115035, ← c115036, k2, ← c115037] at e_1884_1
  have hr := e_1884_1
  simp only [e_1884_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7374)) := by
  have c115038 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115039 := (publicBatchWrapper89_copies3594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115040 := (publicBatchWrapper89_copies3595 a h).1
  have c115041 := (publicBatchWrapper89_copies3595 a h).2.1
  have c115042 := (publicBatchWrapper89_copies3595 a h).2.2.1
  have c115043 := (publicBatchWrapper89_copies3595 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_2
  simp only [← c115038, ← c115039, ← c115040] at e_1884_2
  have e_1884_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_3
  simp only [← c115041, ← c115042, k2, ← c115043] at e_1884_3
  have hr := e_1884_3
  simp only [e_1884_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7375)) := by
  have c115044 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.1
  have c115045 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.1
  have c115046 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.1
  have c115047 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.1
  have c115048 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.1
  have c115049 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_4
  simp only [← c115044, ← c115045, ← c115046] at e_1884_4
  have e_1884_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_5
  simp only [← c115047, ← c115048, k2, ← c115049] at e_1884_5
  have hr := e_1884_5
  simp only [e_1884_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7376)) := by
  have c115050 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115051 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115052 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115053 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115054 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115055 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_6
  simp only [← c115050, ← c115051, ← c115052] at e_1884_6
  have e_1884_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_7
  simp only [← c115053, ← c115054, k2, ← c115055] at e_1884_7
  have hr := e_1884_7
  simp only [e_1884_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7377)) := by
  have c115056 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115057 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115058 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115059 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115060 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115061 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_8
  simp only [← c115056, ← c115057, ← c115058] at e_1884_8
  have e_1884_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_9
  simp only [← c115059, ← c115060, k2, ← c115061] at e_1884_9
  have hr := e_1884_9
  simp only [e_1884_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7378)) := by
  have c115062 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115063 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115064 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115065 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115066 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115067 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_10
  simp only [← c115062, ← c115063, ← c115064] at e_1884_10
  have e_1884_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_11
  simp only [← c115065, ← c115066, k2, ← c115067] at e_1884_11
  have hr := e_1884_11
  simp only [e_1884_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7379)) := by
  have c115068 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115069 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115070 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115071 := (publicBatchWrapper89_copies3595 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115072 := (publicBatchWrapper89_copies3596 a h).1
  have c115073 := (publicBatchWrapper89_copies3596 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_12
  simp only [← c115068, ← c115069, ← c115070] at e_1884_12
  have e_1884_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_13
  simp only [← c115071, ← c115072, k2, ← c115073] at e_1884_13
  have hr := e_1884_13
  simp only [e_1884_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7380)) := by
  have c115074 := (publicBatchWrapper89_copies3596 a h).2.2.1
  have c115075 := (publicBatchWrapper89_copies3596 a h).2.2.2.1
  have c115076 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.1
  have c115077 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.1
  have c115078 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.1
  have c115079 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_14
  simp only [← c115074, ← c115075, ← c115076] at e_1884_14
  have e_1884_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_15
  simp only [← c115077, ← c115078, k2, ← c115079] at e_1884_15
  have hr := e_1884_15
  simp only [e_1884_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7381)) := by
  have c115080 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.1
  have c115081 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.1
  have c115082 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115083 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115084 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115085 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_16
  simp only [← c115080, ← c115081, ← c115082] at e_1884_16
  have e_1884_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_17
  simp only [← c115083, ← c115084, k2, ← c115085] at e_1884_17
  have hr := e_1884_17
  simp only [e_1884_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1884 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7382)) := by
  have c115086 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115087 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115088 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115089 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115090 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115091 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1884_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_18
  simp only [← c115086, ← c115087, ← c115088] at e_1884_18
  have e_1884_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1884 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_19
  simp only [← c115089, ← c115090, k2, ← c115091] at e_1884_19
  have hr := e_1884_19
  simp only [e_1884_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7383)) := by
  have c115092 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115093 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115094 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115095 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115096 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115097 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_0
  simp only [← c115092, ← c115093, ← c115094] at e_1885_0
  have e_1885_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_1
  simp only [← c115095, ← c115096, k2, ← c115097] at e_1885_1
  have hr := e_1885_1
  simp only [e_1885_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7384)) := by
  have c115098 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115099 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115100 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115101 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115102 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115103 := (publicBatchWrapper89_copies3596 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_2
  simp only [← c115098, ← c115099, ← c115100] at e_1885_2
  have e_1885_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_3
  simp only [← c115101, ← c115102, k2, ← c115103] at e_1885_3
  have hr := e_1885_3
  simp only [e_1885_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7385)) := by
  have c115104 := (publicBatchWrapper89_copies3597 a h).1
  have c115105 := (publicBatchWrapper89_copies3597 a h).2.1
  have c115106 := (publicBatchWrapper89_copies3597 a h).2.2.1
  have c115107 := (publicBatchWrapper89_copies3597 a h).2.2.2.1
  have c115108 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.1
  have c115109 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_4
  simp only [← c115104, ← c115105, ← c115106] at e_1885_4
  have e_1885_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_5
  simp only [← c115107, ← c115108, k2, ← c115109] at e_1885_5
  have hr := e_1885_5
  simp only [e_1885_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7386)) := by
  have c115110 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.1
  have c115111 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.1
  have c115112 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.1
  have c115113 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.1
  have c115114 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115115 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_6
  simp only [← c115110, ← c115111, ← c115112] at e_1885_6
  have e_1885_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_7
  simp only [← c115113, ← c115114, k2, ← c115115] at e_1885_7
  have hr := e_1885_7
  simp only [e_1885_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7387)) := by
  have c115116 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115117 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115118 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115119 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115120 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115121 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_8
  simp only [← c115116, ← c115117, ← c115118] at e_1885_8
  have e_1885_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_9
  simp only [← c115119, ← c115120, k2, ← c115121] at e_1885_9
  have hr := e_1885_9
  simp only [e_1885_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7388)) := by
  have c115122 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115123 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115124 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115125 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115126 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115127 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_10
  simp only [← c115122, ← c115123, ← c115124] at e_1885_10
  have e_1885_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_11
  simp only [← c115125, ← c115126, k2, ← c115127] at e_1885_11
  have hr := e_1885_11
  simp only [e_1885_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7389)) := by
  have c115128 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115129 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115130 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115131 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115132 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115133 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_12
  simp only [← c115128, ← c115129, ← c115130] at e_1885_12
  have e_1885_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_13
  simp only [← c115131, ← c115132, k2, ← c115133] at e_1885_13
  have hr := e_1885_13
  simp only [e_1885_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7390)) := by
  have c115134 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115135 := (publicBatchWrapper89_copies3597 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115136 := (publicBatchWrapper89_copies3598 a h).1
  have c115137 := (publicBatchWrapper89_copies3598 a h).2.1
  have c115138 := (publicBatchWrapper89_copies3598 a h).2.2.1
  have c115139 := (publicBatchWrapper89_copies3598 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_14
  simp only [← c115134, ← c115135, ← c115136] at e_1885_14
  have e_1885_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_15
  simp only [← c115137, ← c115138, k2, ← c115139] at e_1885_15
  have hr := e_1885_15
  simp only [e_1885_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7391)) := by
  have c115140 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.1
  have c115141 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.1
  have c115142 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.1
  have c115143 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.1
  have c115144 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.1
  have c115145 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_16
  simp only [← c115140, ← c115141, ← c115142] at e_1885_16
  have e_1885_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_17
  simp only [← c115143, ← c115144, k2, ← c115145] at e_1885_17
  have hr := e_1885_17
  simp only [e_1885_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1885 79) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7392)) := by
  have c115146 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115147 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115148 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115149 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115150 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115151 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1885_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_18
  simp only [← c115146, ← c115147, ← c115148] at e_1885_18
  have e_1885_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1885 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_19
  simp only [← c115149, ← c115150, k2, ← c115151] at e_1885_19
  have hr := e_1885_19
  simp only [e_1885_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 7) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7393)) := by
  have c115152 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115153 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115154 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115155 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115156 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115157 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_0
  simp only [← c115152, ← c115153, ← c115154] at e_1886_0
  have e_1886_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_1
  simp only [← c115155, ← c115156, k2, ← c115157] at e_1886_1
  have hr := e_1886_1
  simp only [e_1886_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 15) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7394)) := by
  have c115158 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115159 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115160 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115161 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115162 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115163 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_2
  simp only [← c115158, ← c115159, ← c115160] at e_1886_2
  have e_1886_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_3
  simp only [← c115161, ← c115162, k2, ← c115163] at e_1886_3
  have hr := e_1886_3
  simp only [e_1886_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 23) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7395)) := by
  have c115164 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115165 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115166 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115167 := (publicBatchWrapper89_copies3598 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115168 := (publicBatchWrapper89_copies3599 a h).1
  have c115169 := (publicBatchWrapper89_copies3599 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_4
  simp only [← c115164, ← c115165, ← c115166] at e_1886_4
  have e_1886_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_5
  simp only [← c115167, ← c115168, k2, ← c115169] at e_1886_5
  have hr := e_1886_5
  simp only [e_1886_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 31) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7396)) := by
  have c115170 := (publicBatchWrapper89_copies3599 a h).2.2.1
  have c115171 := (publicBatchWrapper89_copies3599 a h).2.2.2.1
  have c115172 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.1
  have c115173 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.1
  have c115174 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.1
  have c115175 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_6
  simp only [← c115170, ← c115171, ← c115172] at e_1886_6
  have e_1886_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_7
  simp only [← c115173, ← c115174, k2, ← c115175] at e_1886_7
  have hr := e_1886_7
  simp only [e_1886_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 39) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7397)) := by
  have c115176 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.1
  have c115177 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.1
  have c115178 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115179 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115180 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115181 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_8
  simp only [← c115176, ← c115177, ← c115178] at e_1886_8
  have e_1886_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_9
  simp only [← c115179, ← c115180, k2, ← c115181] at e_1886_9
  have hr := e_1886_9
  simp only [e_1886_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 47) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7398)) := by
  have c115182 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115183 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115184 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115185 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115186 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115187 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_10
  simp only [← c115182, ← c115183, ← c115184] at e_1886_10
  have e_1886_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_11
  simp only [← c115185, ← c115186, k2, ← c115187] at e_1886_11
  have hr := e_1886_11
  simp only [e_1886_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 55) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7399)) := by
  have c115188 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115189 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115190 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115191 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115192 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115193 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_12
  simp only [← c115188, ← c115189, ← c115190] at e_1886_12
  have e_1886_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_13
  simp only [← c115191, ← c115192, k2, ← c115193] at e_1886_13
  have hr := e_1886_13
  simp only [e_1886_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 63) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7400)) := by
  have c115194 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115195 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115196 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115197 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115198 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115199 := (publicBatchWrapper89_copies3599 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_14
  simp only [← c115194, ← c115195, ← c115196] at e_1886_14
  have e_1886_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_15
  simp only [← c115197, ← c115198, k2, ← c115199] at e_1886_15
  have hr := e_1886_15
  simp only [e_1886_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 71) = bselect (a (.wire 36 43)) (a (.virt 200)) (a (.virt 7401)) := by
  have c115200 := (publicBatchWrapper89_copies3600 a h).1
  have c115201 := (publicBatchWrapper89_copies3600 a h).2.1
  have c115202 := (publicBatchWrapper89_copies3600 a h).2.2.1
  have c115203 := (publicBatchWrapper89_copies3600 a h).2.2.2.1
  have c115204 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.1
  have c115205 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_16
  simp only [← c115200, ← c115201, ← c115202] at e_1886_16
  have e_1886_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_17
  simp only [← c115203, ← c115204, k2, ← c115205] at e_1886_17
  have hr := e_1886_17
  simp only [e_1886_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1886 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7570)) := by
  have c115206 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.1
  have c115207 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.1
  have c115208 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.1
  have c115209 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.1
  have c115210 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115211 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1886_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_18
  simp only [← c115206, ← c115207, ← c115208] at e_1886_18
  have e_1886_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1886 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_19
  simp only [← c115209, ← c115210, k2, ← c115211] at e_1886_19
  have hr := e_1886_19
  simp only [e_1886_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7571)) := by
  have c115212 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115213 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115214 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115215 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115216 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115217 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_0
  simp only [← c115212, ← c115213, ← c115214] at e_1887_0
  have e_1887_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_1
  simp only [← c115215, ← c115216, k2, ← c115217] at e_1887_1
  have hr := e_1887_1
  simp only [e_1887_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7572)) := by
  have c115218 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115219 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115220 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115221 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115222 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115223 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_2
  simp only [← c115218, ← c115219, ← c115220] at e_1887_2
  have e_1887_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_3
  simp only [← c115221, ← c115222, k2, ← c115223] at e_1887_3
  have hr := e_1887_3
  simp only [e_1887_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7573)) := by
  have c115224 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115225 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115226 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115227 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115228 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115229 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_4
  simp only [← c115224, ← c115225, ← c115226] at e_1887_4
  have e_1887_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_5
  simp only [← c115227, ← c115228, k2, ← c115229] at e_1887_5
  have hr := e_1887_5
  simp only [e_1887_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7574)) := by
  have c115230 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115231 := (publicBatchWrapper89_copies3600 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115232 := (publicBatchWrapper89_copies3601 a h).1
  have c115233 := (publicBatchWrapper89_copies3601 a h).2.1
  have c115234 := (publicBatchWrapper89_copies3601 a h).2.2.1
  have c115235 := (publicBatchWrapper89_copies3601 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_6
  simp only [← c115230, ← c115231, ← c115232] at e_1887_6
  have e_1887_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_7
  simp only [← c115233, ← c115234, k2, ← c115235] at e_1887_7
  have hr := e_1887_7
  simp only [e_1887_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7575)) := by
  have c115236 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.1
  have c115237 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.1
  have c115238 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.1
  have c115239 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.1
  have c115240 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.1
  have c115241 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_8
  simp only [← c115236, ← c115237, ← c115238] at e_1887_8
  have e_1887_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_9
  simp only [← c115239, ← c115240, k2, ← c115241] at e_1887_9
  have hr := e_1887_9
  simp only [e_1887_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7576)) := by
  have c115242 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115243 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115244 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115245 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115246 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115247 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_10
  simp only [← c115242, ← c115243, ← c115244] at e_1887_10
  have e_1887_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_11
  simp only [← c115245, ← c115246, k2, ← c115247] at e_1887_11
  have hr := e_1887_11
  simp only [e_1887_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7577)) := by
  have c115248 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115249 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115250 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115251 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115252 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115253 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_12
  simp only [← c115248, ← c115249, ← c115250] at e_1887_12
  have e_1887_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_13
  simp only [← c115251, ← c115252, k2, ← c115253] at e_1887_13
  have hr := e_1887_13
  simp only [e_1887_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7578)) := by
  have c115254 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115255 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115256 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115257 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115258 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115259 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_14
  simp only [← c115254, ← c115255, ← c115256] at e_1887_14
  have e_1887_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_15
  simp only [← c115257, ← c115258, k2, ← c115259] at e_1887_15
  have hr := e_1887_15
  simp only [e_1887_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7579)) := by
  have c115260 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115261 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115262 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115263 := (publicBatchWrapper89_copies3601 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115264 := (publicBatchWrapper89_copies3602 a h).1
  have c115265 := (publicBatchWrapper89_copies3602 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_16
  simp only [← c115260, ← c115261, ← c115262] at e_1887_16
  have e_1887_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_17
  simp only [← c115263, ← c115264, k2, ← c115265] at e_1887_17
  have hr := e_1887_17
  simp only [e_1887_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1887 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7580)) := by
  have c115266 := (publicBatchWrapper89_copies3602 a h).2.2.1
  have c115267 := (publicBatchWrapper89_copies3602 a h).2.2.2.1
  have c115268 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.1
  have c115269 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.1
  have c115270 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.1
  have c115271 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1887_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_18
  simp only [← c115266, ← c115267, ← c115268] at e_1887_18
  have e_1887_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1887 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_19
  simp only [← c115269, ← c115270, k2, ← c115271] at e_1887_19
  have hr := e_1887_19
  simp only [e_1887_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7581)) := by
  have c115272 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.1
  have c115273 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.1
  have c115274 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115275 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115276 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115277 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_0
  simp only [← c115272, ← c115273, ← c115274] at e_1888_0
  have e_1888_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_1
  simp only [← c115275, ← c115276, k2, ← c115277] at e_1888_1
  have hr := e_1888_1
  simp only [e_1888_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7582)) := by
  have c115278 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115279 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115280 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115281 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115282 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115283 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_2
  simp only [← c115278, ← c115279, ← c115280] at e_1888_2
  have e_1888_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_3
  simp only [← c115281, ← c115282, k2, ← c115283] at e_1888_3
  have hr := e_1888_3
  simp only [e_1888_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7583)) := by
  have c115284 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115285 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115286 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115287 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115288 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115289 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_4
  simp only [← c115284, ← c115285, ← c115286] at e_1888_4
  have e_1888_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_5
  simp only [← c115287, ← c115288, k2, ← c115289] at e_1888_5
  have hr := e_1888_5
  simp only [e_1888_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7584)) := by
  have c115290 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115291 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115292 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115293 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115294 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115295 := (publicBatchWrapper89_copies3602 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_6
  simp only [← c115290, ← c115291, ← c115292] at e_1888_6
  have e_1888_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_7
  simp only [← c115293, ← c115294, k2, ← c115295] at e_1888_7
  have hr := e_1888_7
  simp only [e_1888_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7585)) := by
  have c115296 := (publicBatchWrapper89_copies3603 a h).1
  have c115297 := (publicBatchWrapper89_copies3603 a h).2.1
  have c115298 := (publicBatchWrapper89_copies3603 a h).2.2.1
  have c115299 := (publicBatchWrapper89_copies3603 a h).2.2.2.1
  have c115300 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.1
  have c115301 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_8
  simp only [← c115296, ← c115297, ← c115298] at e_1888_8
  have e_1888_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_9
  simp only [← c115299, ← c115300, k2, ← c115301] at e_1888_9
  have hr := e_1888_9
  simp only [e_1888_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7586)) := by
  have c115302 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.1
  have c115303 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.1
  have c115304 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.1
  have c115305 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.1
  have c115306 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115307 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_10
  simp only [← c115302, ← c115303, ← c115304] at e_1888_10
  have e_1888_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_11
  simp only [← c115305, ← c115306, k2, ← c115307] at e_1888_11
  have hr := e_1888_11
  simp only [e_1888_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7587)) := by
  have c115308 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115309 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115310 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115311 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115312 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115313 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_12
  simp only [← c115308, ← c115309, ← c115310] at e_1888_12
  have e_1888_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_13
  simp only [← c115311, ← c115312, k2, ← c115313] at e_1888_13
  have hr := e_1888_13
  simp only [e_1888_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7588)) := by
  have c115314 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115315 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115316 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115317 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115318 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115319 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_14
  simp only [← c115314, ← c115315, ← c115316] at e_1888_14
  have e_1888_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_15
  simp only [← c115317, ← c115318, k2, ← c115319] at e_1888_15
  have hr := e_1888_15
  simp only [e_1888_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7589)) := by
  have c115320 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115321 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115322 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115323 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115324 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115325 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_16
  simp only [← c115320, ← c115321, ← c115322] at e_1888_16
  have e_1888_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_17
  simp only [← c115323, ← c115324, k2, ← c115325] at e_1888_17
  have hr := e_1888_17
  simp only [e_1888_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1888 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7590)) := by
  have c115326 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115327 := (publicBatchWrapper89_copies3603 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115328 := (publicBatchWrapper89_copies3604 a h).1
  have c115329 := (publicBatchWrapper89_copies3604 a h).2.1
  have c115330 := (publicBatchWrapper89_copies3604 a h).2.2.1
  have c115331 := (publicBatchWrapper89_copies3604 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1888_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_18
  simp only [← c115326, ← c115327, ← c115328] at e_1888_18
  have e_1888_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1888 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_19
  simp only [← c115329, ← c115330, k2, ← c115331] at e_1888_19
  have hr := e_1888_19
  simp only [e_1888_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7591)) := by
  have c115332 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.1
  have c115333 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.1
  have c115334 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.1
  have c115335 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.1
  have c115336 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.1
  have c115337 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_0
  simp only [← c115332, ← c115333, ← c115334] at e_1889_0
  have e_1889_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_1
  simp only [← c115335, ← c115336, k2, ← c115337] at e_1889_1
  have hr := e_1889_1
  simp only [e_1889_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 15) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7592)) := by
  have c115338 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115339 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115340 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115341 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115342 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115343 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_2
  simp only [← c115338, ← c115339, ← c115340] at e_1889_2
  have e_1889_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_3
  simp only [← c115341, ← c115342, k2, ← c115343] at e_1889_3
  have hr := e_1889_3
  simp only [e_1889_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 23) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7593)) := by
  have c115344 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115345 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115346 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115347 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115348 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115349 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_4
  simp only [← c115344, ← c115345, ← c115346] at e_1889_4
  have e_1889_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_5
  simp only [← c115347, ← c115348, k2, ← c115349] at e_1889_5
  have hr := e_1889_5
  simp only [e_1889_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 31) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7594)) := by
  have c115350 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115351 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115352 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115353 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115354 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115355 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_6
  simp only [← c115350, ← c115351, ← c115352] at e_1889_6
  have e_1889_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_7
  simp only [← c115353, ← c115354, k2, ← c115355] at e_1889_7
  have hr := e_1889_7
  simp only [e_1889_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 39) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7595)) := by
  have c115356 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115357 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115358 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115359 := (publicBatchWrapper89_copies3604 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115360 := (publicBatchWrapper89_copies3605 a h).1
  have c115361 := (publicBatchWrapper89_copies3605 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_8
  simp only [← c115356, ← c115357, ← c115358] at e_1889_8
  have e_1889_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_9
  simp only [← c115359, ← c115360, k2, ← c115361] at e_1889_9
  have hr := e_1889_9
  simp only [e_1889_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 47) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7596)) := by
  have c115362 := (publicBatchWrapper89_copies3605 a h).2.2.1
  have c115363 := (publicBatchWrapper89_copies3605 a h).2.2.2.1
  have c115364 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.1
  have c115365 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.1
  have c115366 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.1
  have c115367 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_10
  simp only [← c115362, ← c115363, ← c115364] at e_1889_10
  have e_1889_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_11
  simp only [← c115365, ← c115366, k2, ← c115367] at e_1889_11
  have hr := e_1889_11
  simp only [e_1889_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 55) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7597)) := by
  have c115368 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.1
  have c115369 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.1
  have c115370 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115371 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115372 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115373 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_12
  simp only [← c115368, ← c115369, ← c115370] at e_1889_12
  have e_1889_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_13
  simp only [← c115371, ← c115372, k2, ← c115373] at e_1889_13
  have hr := e_1889_13
  simp only [e_1889_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 63) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7598)) := by
  have c115374 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115375 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115376 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115377 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115378 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115379 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_14
  simp only [← c115374, ← c115375, ← c115376] at e_1889_14
  have e_1889_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_15
  simp only [← c115377, ← c115378, k2, ← c115379] at e_1889_15
  have hr := e_1889_15
  simp only [e_1889_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 71) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7599)) := by
  have c115380 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115381 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115382 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115383 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115384 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115385 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_16
  simp only [← c115380, ← c115381, ← c115382] at e_1889_16
  have e_1889_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_17
  simp only [← c115383, ← c115384, k2, ← c115385] at e_1889_17
  have hr := e_1889_17
  simp only [e_1889_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1889 79) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7600)) := by
  have c115386 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115387 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115388 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115389 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115390 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115391 := (publicBatchWrapper89_copies3605 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1889_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_18
  simp only [← c115386, ← c115387, ← c115388] at e_1889_18
  have e_1889_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1889 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_19
  simp only [← c115389, ← c115390, k2, ← c115391] at e_1889_19
  have hr := e_1889_19
  simp only [e_1889_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 7) = bselect (a (.wire 38 7)) (a (.virt 200)) (a (.virt 7601)) := by
  have c115392 := (publicBatchWrapper89_copies3606 a h).1
  have c115393 := (publicBatchWrapper89_copies3606 a h).2.1
  have c115394 := (publicBatchWrapper89_copies3606 a h).2.2.1
  have c115395 := (publicBatchWrapper89_copies3606 a h).2.2.2.1
  have c115396 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.1
  have c115397 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_0
  simp only [← c115392, ← c115393, ← c115394] at e_1890_0
  have e_1890_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_1
  simp only [← c115395, ← c115396, k2, ← c115397] at e_1890_1
  have hr := e_1890_1
  simp only [e_1890_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7770)) := by
  have c115398 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.1
  have c115399 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.1
  have c115400 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.1
  have c115401 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.1
  have c115402 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115403 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_2
  simp only [← c115398, ← c115399, ← c115400] at e_1890_2
  have e_1890_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_3
  simp only [← c115401, ← c115402, k2, ← c115403] at e_1890_3
  have hr := e_1890_3
  simp only [e_1890_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7771)) := by
  have c115404 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115405 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115406 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115407 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115408 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115409 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_4
  simp only [← c115404, ← c115405, ← c115406] at e_1890_4
  have e_1890_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_5
  simp only [← c115407, ← c115408, k2, ← c115409] at e_1890_5
  have hr := e_1890_5
  simp only [e_1890_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7772)) := by
  have c115410 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115411 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115412 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115413 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115414 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115415 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_6
  simp only [← c115410, ← c115411, ← c115412] at e_1890_6
  have e_1890_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_7
  simp only [← c115413, ← c115414, k2, ← c115415] at e_1890_7
  have hr := e_1890_7
  simp only [e_1890_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7773)) := by
  have c115416 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115417 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115418 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115419 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115420 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115421 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_8
  simp only [← c115416, ← c115417, ← c115418] at e_1890_8
  have e_1890_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_9
  simp only [← c115419, ← c115420, k2, ← c115421] at e_1890_9
  have hr := e_1890_9
  simp only [e_1890_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7774)) := by
  have c115422 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115423 := (publicBatchWrapper89_copies3606 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115424 := (publicBatchWrapper89_copies3607 a h).1
  have c115425 := (publicBatchWrapper89_copies3607 a h).2.1
  have c115426 := (publicBatchWrapper89_copies3607 a h).2.2.1
  have c115427 := (publicBatchWrapper89_copies3607 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_10
  simp only [← c115422, ← c115423, ← c115424] at e_1890_10
  have e_1890_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_11
  simp only [← c115425, ← c115426, k2, ← c115427] at e_1890_11
  have hr := e_1890_11
  simp only [e_1890_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7775)) := by
  have c115428 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.1
  have c115429 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.1
  have c115430 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.1
  have c115431 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.1
  have c115432 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.1
  have c115433 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_12
  simp only [← c115428, ← c115429, ← c115430] at e_1890_12
  have e_1890_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_13
  simp only [← c115431, ← c115432, k2, ← c115433] at e_1890_13
  have hr := e_1890_13
  simp only [e_1890_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7776)) := by
  have c115434 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115435 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115436 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115437 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115438 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115439 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_14
  simp only [← c115434, ← c115435, ← c115436] at e_1890_14
  have e_1890_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_15
  simp only [← c115437, ← c115438, k2, ← c115439] at e_1890_15
  have hr := e_1890_15
  simp only [e_1890_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7777)) := by
  have c115440 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115441 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115442 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115443 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115444 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115445 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_16
  simp only [← c115440, ← c115441, ← c115442] at e_1890_16
  have e_1890_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_17
  simp only [← c115443, ← c115444, k2, ← c115445] at e_1890_17
  have hr := e_1890_17
  simp only [e_1890_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1890 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7778)) := by
  have c115446 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115447 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115448 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115449 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115450 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115451 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1890_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_18
  simp only [← c115446, ← c115447, ← c115448] at e_1890_18
  have e_1890_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1890 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_19
  simp only [← c115449, ← c115450, k2, ← c115451] at e_1890_19
  have hr := e_1890_19
  simp only [e_1890_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7779)) := by
  have c115452 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115453 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115454 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115455 := (publicBatchWrapper89_copies3607 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115456 := (publicBatchWrapper89_copies3608 a h).1
  have c115457 := (publicBatchWrapper89_copies3608 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_0
  simp only [← c115452, ← c115453, ← c115454] at e_1891_0
  have e_1891_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_1
  simp only [← c115455, ← c115456, k2, ← c115457] at e_1891_1
  have hr := e_1891_1
  simp only [e_1891_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7780)) := by
  have c115458 := (publicBatchWrapper89_copies3608 a h).2.2.1
  have c115459 := (publicBatchWrapper89_copies3608 a h).2.2.2.1
  have c115460 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.1
  have c115461 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.1
  have c115462 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.1
  have c115463 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_2
  simp only [← c115458, ← c115459, ← c115460] at e_1891_2
  have e_1891_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_3
  simp only [← c115461, ← c115462, k2, ← c115463] at e_1891_3
  have hr := e_1891_3
  simp only [e_1891_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7781)) := by
  have c115464 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.1
  have c115465 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.1
  have c115466 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115467 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115468 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115469 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_4
  simp only [← c115464, ← c115465, ← c115466] at e_1891_4
  have e_1891_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_5
  simp only [← c115467, ← c115468, k2, ← c115469] at e_1891_5
  have hr := e_1891_5
  simp only [e_1891_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7782)) := by
  have c115470 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115471 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115472 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115473 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115474 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115475 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_6
  simp only [← c115470, ← c115471, ← c115472] at e_1891_6
  have e_1891_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_7
  simp only [← c115473, ← c115474, k2, ← c115475] at e_1891_7
  have hr := e_1891_7
  simp only [e_1891_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7783)) := by
  have c115476 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115477 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115478 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115479 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115480 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115481 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_8
  simp only [← c115476, ← c115477, ← c115478] at e_1891_8
  have e_1891_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_9
  simp only [← c115479, ← c115480, k2, ← c115481] at e_1891_9
  have hr := e_1891_9
  simp only [e_1891_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7784)) := by
  have c115482 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115483 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115484 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115485 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115486 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115487 := (publicBatchWrapper89_copies3608 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_10
  simp only [← c115482, ← c115483, ← c115484] at e_1891_10
  have e_1891_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_11
  simp only [← c115485, ← c115486, k2, ← c115487] at e_1891_11
  have hr := e_1891_11
  simp only [e_1891_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7785)) := by
  have c115488 := (publicBatchWrapper89_copies3609 a h).1
  have c115489 := (publicBatchWrapper89_copies3609 a h).2.1
  have c115490 := (publicBatchWrapper89_copies3609 a h).2.2.1
  have c115491 := (publicBatchWrapper89_copies3609 a h).2.2.2.1
  have c115492 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.1
  have c115493 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_12
  simp only [← c115488, ← c115489, ← c115490] at e_1891_12
  have e_1891_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_13
  simp only [← c115491, ← c115492, k2, ← c115493] at e_1891_13
  have hr := e_1891_13
  simp only [e_1891_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7786)) := by
  have c115494 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.1
  have c115495 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.1
  have c115496 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.1
  have c115497 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.1
  have c115498 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115499 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_14
  simp only [← c115494, ← c115495, ← c115496] at e_1891_14
  have e_1891_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_15
  simp only [← c115497, ← c115498, k2, ← c115499] at e_1891_15
  have hr := e_1891_15
  simp only [e_1891_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7787)) := by
  have c115500 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115501 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115502 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115503 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115504 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115505 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_16
  simp only [← c115500, ← c115501, ← c115502] at e_1891_16
  have e_1891_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_17
  simp only [← c115503, ← c115504, k2, ← c115505] at e_1891_17
  have hr := e_1891_17
  simp only [e_1891_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1891 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7788)) := by
  have c115506 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115507 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115508 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115509 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115510 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115511 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1891_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_18
  simp only [← c115506, ← c115507, ← c115508] at e_1891_18
  have e_1891_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1891 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_19
  simp only [← c115509, ← c115510, k2, ← c115511] at e_1891_19
  have hr := e_1891_19
  simp only [e_1891_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7789)) := by
  have c115512 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115513 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115514 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115515 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115516 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115517 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_0
  simp only [← c115512, ← c115513, ← c115514] at e_1892_0
  have e_1892_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_1
  simp only [← c115515, ← c115516, k2, ← c115517] at e_1892_1
  have hr := e_1892_1
  simp only [e_1892_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7790)) := by
  have c115518 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115519 := (publicBatchWrapper89_copies3609 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115520 := (publicBatchWrapper89_copies3610 a h).1
  have c115521 := (publicBatchWrapper89_copies3610 a h).2.1
  have c115522 := (publicBatchWrapper89_copies3610 a h).2.2.1
  have c115523 := (publicBatchWrapper89_copies3610 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_2
  simp only [← c115518, ← c115519, ← c115520] at e_1892_2
  have e_1892_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_3
  simp only [← c115521, ← c115522, k2, ← c115523] at e_1892_3
  have hr := e_1892_3
  simp only [e_1892_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7791)) := by
  have c115524 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.1
  have c115525 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.1
  have c115526 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.1
  have c115527 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.1
  have c115528 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.1
  have c115529 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_4
  simp only [← c115524, ← c115525, ← c115526] at e_1892_4
  have e_1892_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_5
  simp only [← c115527, ← c115528, k2, ← c115529] at e_1892_5
  have hr := e_1892_5
  simp only [e_1892_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 31) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7792)) := by
  have c115530 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115531 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115532 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115533 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115534 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115535 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_6
  simp only [← c115530, ← c115531, ← c115532] at e_1892_6
  have e_1892_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_7
  simp only [← c115533, ← c115534, k2, ← c115535] at e_1892_7
  have hr := e_1892_7
  simp only [e_1892_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 39) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7793)) := by
  have c115536 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115537 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115538 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115539 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115540 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115541 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_8
  simp only [← c115536, ← c115537, ← c115538] at e_1892_8
  have e_1892_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_9
  simp only [← c115539, ← c115540, k2, ← c115541] at e_1892_9
  have hr := e_1892_9
  simp only [e_1892_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 47) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7794)) := by
  have c115542 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115543 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115544 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115545 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115546 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115547 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_10
  simp only [← c115542, ← c115543, ← c115544] at e_1892_10
  have e_1892_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_11
  simp only [← c115545, ← c115546, k2, ← c115547] at e_1892_11
  have hr := e_1892_11
  simp only [e_1892_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 55) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7795)) := by
  have c115548 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115549 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115550 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115551 := (publicBatchWrapper89_copies3610 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115552 := (publicBatchWrapper89_copies3611 a h).1
  have c115553 := (publicBatchWrapper89_copies3611 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_12
  simp only [← c115548, ← c115549, ← c115550] at e_1892_12
  have e_1892_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_13
  simp only [← c115551, ← c115552, k2, ← c115553] at e_1892_13
  have hr := e_1892_13
  simp only [e_1892_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 63) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7796)) := by
  have c115554 := (publicBatchWrapper89_copies3611 a h).2.2.1
  have c115555 := (publicBatchWrapper89_copies3611 a h).2.2.2.1
  have c115556 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.1
  have c115557 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.1
  have c115558 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.1
  have c115559 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_14
  simp only [← c115554, ← c115555, ← c115556] at e_1892_14
  have e_1892_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_15
  simp only [← c115557, ← c115558, k2, ← c115559] at e_1892_15
  have hr := e_1892_15
  simp only [e_1892_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 71) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7797)) := by
  have c115560 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.1
  have c115561 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.1
  have c115562 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115563 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115564 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115565 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_16
  simp only [← c115560, ← c115561, ← c115562] at e_1892_16
  have e_1892_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_17
  simp only [← c115563, ← c115564, k2, ← c115565] at e_1892_17
  have hr := e_1892_17
  simp only [e_1892_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1892 79) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7798)) := by
  have c115566 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115567 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115568 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115569 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115570 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115571 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1892_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_18
  simp only [← c115566, ← c115567, ← c115568] at e_1892_18
  have e_1892_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1892 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_19
  simp only [← c115569, ← c115570, k2, ← c115571] at e_1892_19
  have hr := e_1892_19
  simp only [e_1892_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 7) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7799)) := by
  have c115572 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115573 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115574 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115575 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115576 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115577 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_0
  simp only [← c115572, ← c115573, ← c115574] at e_1893_0
  have e_1893_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_1
  simp only [← c115575, ← c115576, k2, ← c115577] at e_1893_1
  have hr := e_1893_1
  simp only [e_1893_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 15) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7800)) := by
  have c115578 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115579 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115580 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115581 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115582 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115583 := (publicBatchWrapper89_copies3611 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_2
  simp only [← c115578, ← c115579, ← c115580] at e_1893_2
  have e_1893_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_3
  simp only [← c115581, ← c115582, k2, ← c115583] at e_1893_3
  have hr := e_1893_3
  simp only [e_1893_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 23) = bselect (a (.wire 38 51)) (a (.virt 200)) (a (.virt 7801)) := by
  have c115584 := (publicBatchWrapper89_copies3612 a h).1
  have c115585 := (publicBatchWrapper89_copies3612 a h).2.1
  have c115586 := (publicBatchWrapper89_copies3612 a h).2.2.1
  have c115587 := (publicBatchWrapper89_copies3612 a h).2.2.2.1
  have c115588 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.1
  have c115589 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_4
  simp only [← c115584, ← c115585, ← c115586] at e_1893_4
  have e_1893_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_5
  simp only [← c115587, ← c115588, k2, ← c115589] at e_1893_5
  have hr := e_1893_5
  simp only [e_1893_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7970)) := by
  have c115590 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.1
  have c115591 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.1
  have c115592 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.1
  have c115593 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.1
  have c115594 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115595 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_6
  simp only [← c115590, ← c115591, ← c115592] at e_1893_6
  have e_1893_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_7
  simp only [← c115593, ← c115594, k2, ← c115595] at e_1893_7
  have hr := e_1893_7
  simp only [e_1893_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7971)) := by
  have c115596 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115597 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115598 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115599 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115600 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115601 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_8
  simp only [← c115596, ← c115597, ← c115598] at e_1893_8
  have e_1893_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_9
  simp only [← c115599, ← c115600, k2, ← c115601] at e_1893_9
  have hr := e_1893_9
  simp only [e_1893_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7972)) := by
  have c115602 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115603 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115604 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115605 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115606 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115607 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_10
  simp only [← c115602, ← c115603, ← c115604] at e_1893_10
  have e_1893_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_11
  simp only [← c115605, ← c115606, k2, ← c115607] at e_1893_11
  have hr := e_1893_11
  simp only [e_1893_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7973)) := by
  have c115608 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115609 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115610 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115611 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115612 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115613 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_12
  simp only [← c115608, ← c115609, ← c115610] at e_1893_12
  have e_1893_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_13
  simp only [← c115611, ← c115612, k2, ← c115613] at e_1893_13
  have hr := e_1893_13
  simp only [e_1893_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7974)) := by
  have c115614 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115615 := (publicBatchWrapper89_copies3612 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115616 := (publicBatchWrapper89_copies3613 a h).1
  have c115617 := (publicBatchWrapper89_copies3613 a h).2.1
  have c115618 := (publicBatchWrapper89_copies3613 a h).2.2.1
  have c115619 := (publicBatchWrapper89_copies3613 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_14
  simp only [← c115614, ← c115615, ← c115616] at e_1893_14
  have e_1893_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_15
  simp only [← c115617, ← c115618, k2, ← c115619] at e_1893_15
  have hr := e_1893_15
  simp only [e_1893_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7975)) := by
  have c115620 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.1
  have c115621 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.1
  have c115622 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.1
  have c115623 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.1
  have c115624 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.1
  have c115625 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_16
  simp only [← c115620, ← c115621, ← c115622] at e_1893_16
  have e_1893_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_17
  simp only [← c115623, ← c115624, k2, ← c115625] at e_1893_17
  have hr := e_1893_17
  simp only [e_1893_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1893 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7976)) := by
  have c115626 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115627 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115628 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115629 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115630 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115631 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1893_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_18
  simp only [← c115626, ← c115627, ← c115628] at e_1893_18
  have e_1893_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1893 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_19
  simp only [← c115629, ← c115630, k2, ← c115631] at e_1893_19
  have hr := e_1893_19
  simp only [e_1893_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7977)) := by
  have c115632 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115633 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115634 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115635 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115636 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115637 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_0
  simp only [← c115632, ← c115633, ← c115634] at e_1894_0
  have e_1894_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_1
  simp only [← c115635, ← c115636, k2, ← c115637] at e_1894_1
  have hr := e_1894_1
  simp only [e_1894_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7978)) := by
  have c115638 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115639 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115640 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115641 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115642 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115643 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_2
  simp only [← c115638, ← c115639, ← c115640] at e_1894_2
  have e_1894_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_3
  simp only [← c115641, ← c115642, k2, ← c115643] at e_1894_3
  have hr := e_1894_3
  simp only [e_1894_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7979)) := by
  have c115644 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115645 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115646 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115647 := (publicBatchWrapper89_copies3613 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115648 := (publicBatchWrapper89_copies3614 a h).1
  have c115649 := (publicBatchWrapper89_copies3614 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_4
  simp only [← c115644, ← c115645, ← c115646] at e_1894_4
  have e_1894_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_5
  simp only [← c115647, ← c115648, k2, ← c115649] at e_1894_5
  have hr := e_1894_5
  simp only [e_1894_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7980)) := by
  have c115650 := (publicBatchWrapper89_copies3614 a h).2.2.1
  have c115651 := (publicBatchWrapper89_copies3614 a h).2.2.2.1
  have c115652 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.1
  have c115653 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.1
  have c115654 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.1
  have c115655 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_6
  simp only [← c115650, ← c115651, ← c115652] at e_1894_6
  have e_1894_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_7
  simp only [← c115653, ← c115654, k2, ← c115655] at e_1894_7
  have hr := e_1894_7
  simp only [e_1894_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7981)) := by
  have c115656 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.1
  have c115657 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.1
  have c115658 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115659 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115660 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115661 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_8
  simp only [← c115656, ← c115657, ← c115658] at e_1894_8
  have e_1894_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_9
  simp only [← c115659, ← c115660, k2, ← c115661] at e_1894_9
  have hr := e_1894_9
  simp only [e_1894_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7982)) := by
  have c115662 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115663 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115664 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115665 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115666 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115667 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_10
  simp only [← c115662, ← c115663, ← c115664] at e_1894_10
  have e_1894_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_11
  simp only [← c115665, ← c115666, k2, ← c115667] at e_1894_11
  have hr := e_1894_11
  simp only [e_1894_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7983)) := by
  have c115668 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115669 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115670 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115671 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115672 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115673 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_12
  simp only [← c115668, ← c115669, ← c115670] at e_1894_12
  have e_1894_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_13
  simp only [← c115671, ← c115672, k2, ← c115673] at e_1894_13
  have hr := e_1894_13
  simp only [e_1894_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7984)) := by
  have c115674 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115675 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115676 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115677 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115678 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115679 := (publicBatchWrapper89_copies3614 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_14
  simp only [← c115674, ← c115675, ← c115676] at e_1894_14
  have e_1894_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_15
  simp only [← c115677, ← c115678, k2, ← c115679] at e_1894_15
  have hr := e_1894_15
  simp only [e_1894_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7985)) := by
  have c115680 := (publicBatchWrapper89_copies3615 a h).1
  have c115681 := (publicBatchWrapper89_copies3615 a h).2.1
  have c115682 := (publicBatchWrapper89_copies3615 a h).2.2.1
  have c115683 := (publicBatchWrapper89_copies3615 a h).2.2.2.1
  have c115684 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.1
  have c115685 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_16
  simp only [← c115680, ← c115681, ← c115682] at e_1894_16
  have e_1894_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_17
  simp only [← c115683, ← c115684, k2, ← c115685] at e_1894_17
  have hr := e_1894_17
  simp only [e_1894_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1894 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7986)) := by
  have c115686 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.1
  have c115687 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.1
  have c115688 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.1
  have c115689 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.1
  have c115690 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115691 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1894_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_18
  simp only [← c115686, ← c115687, ← c115688] at e_1894_18
  have e_1894_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1894 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_19
  simp only [← c115689, ← c115690, k2, ← c115691] at e_1894_19
  have hr := e_1894_19
  simp only [e_1894_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7987)) := by
  have c115692 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115693 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115694 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115695 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115696 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115697 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_0
  simp only [← c115692, ← c115693, ← c115694] at e_1895_0
  have e_1895_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_1
  simp only [← c115695, ← c115696, k2, ← c115697] at e_1895_1
  have hr := e_1895_1
  simp only [e_1895_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7988)) := by
  have c115698 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115699 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115700 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115701 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115702 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115703 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_2
  simp only [← c115698, ← c115699, ← c115700] at e_1895_2
  have e_1895_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_3
  simp only [← c115701, ← c115702, k2, ← c115703] at e_1895_3
  have hr := e_1895_3
  simp only [e_1895_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7989)) := by
  have c115704 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115705 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115706 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115707 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115708 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115709 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_4
  simp only [← c115704, ← c115705, ← c115706] at e_1895_4
  have e_1895_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_5
  simp only [← c115707, ← c115708, k2, ← c115709] at e_1895_5
  have hr := e_1895_5
  simp only [e_1895_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7990)) := by
  have c115710 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115711 := (publicBatchWrapper89_copies3615 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115712 := (publicBatchWrapper89_copies3616 a h).1
  have c115713 := (publicBatchWrapper89_copies3616 a h).2.1
  have c115714 := (publicBatchWrapper89_copies3616 a h).2.2.1
  have c115715 := (publicBatchWrapper89_copies3616 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_6
  simp only [← c115710, ← c115711, ← c115712] at e_1895_6
  have e_1895_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_7
  simp only [← c115713, ← c115714, k2, ← c115715] at e_1895_7
  have hr := e_1895_7
  simp only [e_1895_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7991)) := by
  have c115716 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.1
  have c115717 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.1
  have c115718 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.1
  have c115719 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.1
  have c115720 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.1
  have c115721 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_8
  simp only [← c115716, ← c115717, ← c115718] at e_1895_8
  have e_1895_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_9
  simp only [← c115719, ← c115720, k2, ← c115721] at e_1895_9
  have hr := e_1895_9
  simp only [e_1895_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 47) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7992)) := by
  have c115722 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115723 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115724 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115725 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115726 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115727 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_10
  simp only [← c115722, ← c115723, ← c115724] at e_1895_10
  have e_1895_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_11
  simp only [← c115725, ← c115726, k2, ← c115727] at e_1895_11
  have hr := e_1895_11
  simp only [e_1895_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 55) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7993)) := by
  have c115728 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115729 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115730 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115731 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115732 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115733 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_12
  simp only [← c115728, ← c115729, ← c115730] at e_1895_12
  have e_1895_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_13
  simp only [← c115731, ← c115732, k2, ← c115733] at e_1895_13
  have hr := e_1895_13
  simp only [e_1895_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 63) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7994)) := by
  have c115734 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115735 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115736 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115737 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115738 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115739 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_14
  simp only [← c115734, ← c115735, ← c115736] at e_1895_14
  have e_1895_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_15
  simp only [← c115737, ← c115738, k2, ← c115739] at e_1895_15
  have hr := e_1895_15
  simp only [e_1895_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 71) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7995)) := by
  have c115740 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115741 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115742 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115743 := (publicBatchWrapper89_copies3616 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115744 := (publicBatchWrapper89_copies3617 a h).1
  have c115745 := (publicBatchWrapper89_copies3617 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_16
  simp only [← c115740, ← c115741, ← c115742] at e_1895_16
  have e_1895_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_17
  simp only [← c115743, ← c115744, k2, ← c115745] at e_1895_17
  have hr := e_1895_17
  simp only [e_1895_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1895 79) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7996)) := by
  have c115746 := (publicBatchWrapper89_copies3617 a h).2.2.1
  have c115747 := (publicBatchWrapper89_copies3617 a h).2.2.2.1
  have c115748 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.1
  have c115749 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.1
  have c115750 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.1
  have c115751 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1895_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_18
  simp only [← c115746, ← c115747, ← c115748] at e_1895_18
  have e_1895_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1895 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_19
  simp only [← c115749, ← c115750, k2, ← c115751] at e_1895_19
  have hr := e_1895_19
  simp only [e_1895_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 7) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7997)) := by
  have c115752 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.1
  have c115753 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.1
  have c115754 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115755 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115756 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115757 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_0
  simp only [← c115752, ← c115753, ← c115754] at e_1896_0
  have e_1896_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_1
  simp only [← c115755, ← c115756, k2, ← c115757] at e_1896_1
  have hr := e_1896_1
  simp only [e_1896_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 15) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7998)) := by
  have c115758 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115759 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115760 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115761 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115762 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115763 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_2
  simp only [← c115758, ← c115759, ← c115760] at e_1896_2
  have e_1896_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_3
  simp only [← c115761, ← c115762, k2, ← c115763] at e_1896_3
  have hr := e_1896_3
  simp only [e_1896_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7999)) := by
  have c115764 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115765 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115766 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115767 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115768 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115769 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_4
  simp only [← c115764, ← c115765, ← c115766] at e_1896_4
  have e_1896_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_6
  simp only [← c115770, ← c115771, ← c115772] at e_1896_6
  have e_1896_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_8
  simp only [← c115776, ← c115777, ← c115778] at e_1896_8
  have e_1896_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_10
  simp only [← c115782, ← c115783, ← c115784] at e_1896_10
  have e_1896_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_12
  simp only [← c115788, ← c115789, ← c115790] at e_1896_12
  have e_1896_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_14
  simp only [← c115794, ← c115795, ← c115796] at e_1896_14
  have e_1896_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_16
  simp only [← c115800, ← c115801, ← c115802] at e_1896_16
  have e_1896_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1896 (by decide)
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
  have e_1896_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1896 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_18
  simp only [← c115806, ← c115807, ← c115808] at e_1896_18
  have e_1896_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1896 (by decide)
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
  have e_1897_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_0
  simp only [← c115812, ← c115813, ← c115814] at e_1897_0
  have e_1897_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_2
  simp only [← c115818, ← c115819, ← c115820] at e_1897_2
  have e_1897_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_4
  simp only [← c115824, ← c115825, ← c115826] at e_1897_4
  have e_1897_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_6
  simp only [← c115830, ← c115831, ← c115832] at e_1897_6
  have e_1897_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_8
  simp only [← c115836, ← c115837, ← c115838] at e_1897_8
  have e_1897_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_10
  simp only [← c115842, ← c115843, ← c115844] at e_1897_10
  have e_1897_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_12
  simp only [← c115848, ← c115849, ← c115850] at e_1897_12
  have e_1897_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_14
  simp only [← c115854, ← c115855, ← c115856] at e_1897_14
  have e_1897_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_16
  simp only [← c115860, ← c115861, ← c115862] at e_1897_16
  have e_1897_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1897 (by decide)
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
  have e_1897_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1897 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_18
  simp only [← c115866, ← c115867, ← c115868] at e_1897_18
  have e_1897_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1897 (by decide)
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
  have e_1898_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_0
  simp only [← c115872, ← c115873, ← c115874] at e_1898_0
  have e_1898_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_2
  simp only [← c115878, ← c115879, ← c115880] at e_1898_2
  have e_1898_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_4
  simp only [← c115884, ← c115885, ← c115886] at e_1898_4
  have e_1898_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_6
  simp only [← c115890, ← c115891, ← c115892] at e_1898_6
  have e_1898_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_8
  simp only [← c115896, ← c115897, ← c115898] at e_1898_8
  have e_1898_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_10
  simp only [← c115902, ← c115903, ← c115904] at e_1898_10
  have e_1898_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_12
  simp only [← c115908, ← c115909, ← c115910] at e_1898_12
  have e_1898_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_14
  simp only [← c115914, ← c115915, ← c115916] at e_1898_14
  have e_1898_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_16
  simp only [← c115920, ← c115921, ← c115922] at e_1898_16
  have e_1898_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1898 (by decide)
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
  have e_1898_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1898 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_18
  simp only [← c115926, ← c115927, ← c115928] at e_1898_18
  have e_1898_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1898 (by decide)
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
  have e_1899_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_0
  simp only [← c115932, ← c115933, ← c115934] at e_1899_0
  have e_1899_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_2
  simp only [← c115938, ← c115939, ← c115940] at e_1899_2
  have e_1899_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_4
  simp only [← c115944, ← c115945, ← c115946] at e_1899_4
  have e_1899_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_6
  simp only [← c115950, ← c115951, ← c115952] at e_1899_6
  have e_1899_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_8
  simp only [← c115956, ← c115957, ← c115958] at e_1899_8
  have e_1899_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_10
  simp only [← c115962, ← c115963, ← c115964] at e_1899_10
  have e_1899_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_12
  simp only [← c115968, ← c115969, ← c115970] at e_1899_12
  have e_1899_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_14
  simp only [← c115974, ← c115975, ← c115976] at e_1899_14
  have e_1899_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_16
  simp only [← c115980, ← c115981, ← c115982] at e_1899_16
  have e_1899_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1899 (by decide)
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
  have e_1899_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1899 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_18
  simp only [← c115986, ← c115987, ← c115988] at e_1899_18
  have e_1899_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1899 (by decide)
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
  have e_1900_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_0
  simp only [← c115992, ← c115993, ← c115994] at e_1900_0
  have e_1900_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_2
  simp only [← c115998, ← c115999, ← c116000] at e_1900_2
  have e_1900_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_4
  simp only [← c116004, ← c116005, ← c116006] at e_1900_4
  have e_1900_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_6
  simp only [← c116010, ← c116011, ← c116012] at e_1900_6
  have e_1900_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_8
  simp only [← c116016, ← c116017, ← c116018] at e_1900_8
  have e_1900_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_10
  simp only [← c116022, ← c116023, ← c116024] at e_1900_10
  have e_1900_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_12
  simp only [← c116028, ← c116029, ← c116030] at e_1900_12
  have e_1900_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_14
  simp only [← c116034, ← c116035, ← c116036] at e_1900_14
  have e_1900_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_16
  simp only [← c116040, ← c116041, ← c116042] at e_1900_16
  have e_1900_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1900 (by decide)
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
  have e_1900_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1900 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_18
  simp only [← c116046, ← c116047, ← c116048] at e_1900_18
  have e_1900_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1900 (by decide)
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
  have e_1901_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_0
  simp only [← c116052, ← c116053, ← c116054] at e_1901_0
  have e_1901_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_2
  simp only [← c116058, ← c116059, ← c116060] at e_1901_2
  have e_1901_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_4
  simp only [← c116064, ← c116065, ← c116066] at e_1901_4
  have e_1901_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_6
  simp only [← c116070, ← c116071, ← c116072] at e_1901_6
  have e_1901_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_8
  simp only [← c116076, ← c116077, ← c116078] at e_1901_8
  have e_1901_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_10
  simp only [← c116082, ← c116083, ← c116084] at e_1901_10
  have e_1901_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_12
  simp only [← c116088, ← c116089, ← c116090] at e_1901_12
  have e_1901_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_14
  simp only [← c116094, ← c116095, ← c116096] at e_1901_14
  have e_1901_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_16
  simp only [← c116100, ← c116101, ← c116102] at e_1901_16
  have e_1901_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1901 (by decide)
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
  have e_1901_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1901 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_18
  simp only [← c116106, ← c116107, ← c116108] at e_1901_18
  have e_1901_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1901 (by decide)
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
  have e_1902_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_0
  simp only [← c116112, ← c116113, ← c116114] at e_1902_0
  have e_1902_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_2
  simp only [← c116118, ← c116119, ← c116120] at e_1902_2
  have e_1902_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_4
  simp only [← c116124, ← c116125, ← c116126] at e_1902_4
  have e_1902_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_6
  simp only [← c116130, ← c116131, ← c116132] at e_1902_6
  have e_1902_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_8
  simp only [← c116136, ← c116137, ← c116138] at e_1902_8
  have e_1902_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_10
  simp only [← c116142, ← c116143, ← c116144] at e_1902_10
  have e_1902_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_12
  simp only [← c116148, ← c116149, ← c116150] at e_1902_12
  have e_1902_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_14
  simp only [← c116154, ← c116155, ← c116156] at e_1902_14
  have e_1902_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_16
  simp only [← c116160, ← c116161, ← c116162] at e_1902_16
  have e_1902_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1902 (by decide)
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
  have e_1902_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1902 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_18
  simp only [← c116166, ← c116167, ← c116168] at e_1902_18
  have e_1902_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1902 (by decide)
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
  have e_1903_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_0
  simp only [← c116172, ← c116173, ← c116174] at e_1903_0
  have e_1903_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_2
  simp only [← c116178, ← c116179, ← c116180] at e_1903_2
  have e_1903_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_4
  simp only [← c116184, ← c116185, ← c116186] at e_1903_4
  have e_1903_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_6
  simp only [← c116190, ← c116191, ← c116192] at e_1903_6
  have e_1903_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_8
  simp only [← c116196, ← c116197, ← c116198] at e_1903_8
  have e_1903_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_10
  simp only [← c116202, ← c116203, ← c116204] at e_1903_10
  have e_1903_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_12
  simp only [← c116208, ← c116209, ← c116210] at e_1903_12
  have e_1903_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_14
  simp only [← c116214, ← c116215, ← c116216] at e_1903_14
  have e_1903_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_16
  simp only [← c116220, ← c116221, ← c116222] at e_1903_16
  have e_1903_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1903 (by decide)
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
  have e_1903_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1903 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_18
  simp only [← c116226, ← c116227, ← c116228] at e_1903_18
  have e_1903_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1903 (by decide)
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
  have e_1904_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_0
  simp only [← c116232, ← c116233, ← c116234] at e_1904_0
  have e_1904_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_2
  simp only [← c116238, ← c116239, ← c116240] at e_1904_2
  have e_1904_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_4
  simp only [← c116244, ← c116245, ← c116246] at e_1904_4
  have e_1904_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_6
  simp only [← c116250, ← c116251, ← c116252] at e_1904_6
  have e_1904_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_8
  simp only [← c116256, ← c116257, ← c116258] at e_1904_8
  have e_1904_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_10
  simp only [← c116262, ← c116263, ← c116264] at e_1904_10
  have e_1904_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_12
  simp only [← c116268, ← c116269, ← c116270] at e_1904_12
  have e_1904_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_14
  simp only [← c116274, ← c116275, ← c116276] at e_1904_14
  have e_1904_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_16
  simp only [← c116280, ← c116281, ← c116282] at e_1904_16
  have e_1904_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1904 (by decide)
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
  have e_1904_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1904 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_18
  simp only [← c116286, ← c116287, ← c116288] at e_1904_18
  have e_1904_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1904 (by decide)
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
  have e_1905_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_0
  simp only [← c116292, ← c116293, ← c116294] at e_1905_0
  have e_1905_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_2
  simp only [← c116298, ← c116299, ← c116300] at e_1905_2
  have e_1905_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_4
  simp only [← c116304, ← c116305, ← c116306] at e_1905_4
  have e_1905_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_6
  simp only [← c116310, ← c116311, ← c116312] at e_1905_6
  have e_1905_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_8
  simp only [← c116316, ← c116317, ← c116318] at e_1905_8
  have e_1905_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_10
  simp only [← c116322, ← c116323, ← c116324] at e_1905_10
  have e_1905_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_12
  simp only [← c116328, ← c116329, ← c116330] at e_1905_12
  have e_1905_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_14
  simp only [← c116334, ← c116335, ← c116336] at e_1905_14
  have e_1905_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_16
  simp only [← c116340, ← c116341, ← c116342] at e_1905_16
  have e_1905_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1905 (by decide)
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
  have e_1905_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1905 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_18
  simp only [← c116346, ← c116347, ← c116348] at e_1905_18
  have e_1905_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1905 (by decide)
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
  have e_1906_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_0
  simp only [← c116352, ← c116353, ← c116354] at e_1906_0
  have e_1906_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_2
  simp only [← c116358, ← c116359, ← c116360] at e_1906_2
  have e_1906_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_4
  simp only [← c116364, ← c116365, ← c116366] at e_1906_4
  have e_1906_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_6
  simp only [← c116370, ← c116371, ← c116372] at e_1906_6
  have e_1906_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_8
  simp only [← c116376, ← c116377, ← c116378] at e_1906_8
  have e_1906_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_10
  simp only [← c116382, ← c116383, ← c116384] at e_1906_10
  have e_1906_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_12
  simp only [← c116388, ← c116389, ← c116390] at e_1906_12
  have e_1906_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_14
  simp only [← c116394, ← c116395, ← c116396] at e_1906_14
  have e_1906_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_16
  simp only [← c116400, ← c116401, ← c116402] at e_1906_16
  have e_1906_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1906 (by decide)
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
  have e_1906_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1906 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_18
  simp only [← c116406, ← c116407, ← c116408] at e_1906_18
  have e_1906_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1906 (by decide)
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
  have e_1907_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_0
  simp only [← c116412, ← c116413, ← c116414] at e_1907_0
  have e_1907_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_2
  simp only [← c116418, ← c116419, ← c116420] at e_1907_2
  have e_1907_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_4
  simp only [← c116424, ← c116425, ← c116426] at e_1907_4
  have e_1907_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_6
  simp only [← c116430, ← c116431, ← c116432] at e_1907_6
  have e_1907_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_8
  simp only [← c116436, ← c116437, ← c116438] at e_1907_8
  have e_1907_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_10
  simp only [← c116442, ← c116443, ← c116444] at e_1907_10
  have e_1907_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_12
  simp only [← c116448, ← c116449, ← c116450] at e_1907_12
  have e_1907_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_14
  simp only [← c116454, ← c116455, ← c116456] at e_1907_14
  have e_1907_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_16
  simp only [← c116460, ← c116461, ← c116462] at e_1907_16
  have e_1907_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1907 (by decide)
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
  have e_1907_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1907 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_18
  simp only [← c116466, ← c116467, ← c116468] at e_1907_18
  have e_1907_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1907 (by decide)
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
  have e_1908_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_0
  simp only [← c116472, ← c116473, ← c116474] at e_1908_0
  have e_1908_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_2
  simp only [← c116478, ← c116479, ← c116480] at e_1908_2
  have e_1908_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_4
  simp only [← c116484, ← c116485, ← c116486] at e_1908_4
  have e_1908_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_6
  simp only [← c116490, ← c116491, ← c116492] at e_1908_6
  have e_1908_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_8
  simp only [← c116496, ← c116497, ← c116498] at e_1908_8
  have e_1908_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_10
  simp only [← c116502, ← c116503, ← c116504] at e_1908_10
  have e_1908_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_12
  simp only [← c116508, ← c116509, ← c116510] at e_1908_12
  have e_1908_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_14
  simp only [← c116514, ← c116515, ← c116516] at e_1908_14
  have e_1908_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_16
  simp only [← c116520, ← c116521, ← c116522] at e_1908_16
  have e_1908_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1908 (by decide)
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
  have e_1908_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1908 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_18
  simp only [← c116526, ← c116527, ← c116528] at e_1908_18
  have e_1908_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1908 (by decide)
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
  have e_1909_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_0
  simp only [← c116532, ← c116533, ← c116534] at e_1909_0
  have e_1909_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_2
  simp only [← c116538, ← c116539, ← c116540] at e_1909_2
  have e_1909_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_4
  simp only [← c116544, ← c116545, ← c116546] at e_1909_4
  have e_1909_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_6
  simp only [← c116550, ← c116551, ← c116552] at e_1909_6
  have e_1909_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_8
  simp only [← c116556, ← c116557, ← c116558] at e_1909_8
  have e_1909_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_10
  simp only [← c116562, ← c116563, ← c116564] at e_1909_10
  have e_1909_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_12
  simp only [← c116568, ← c116569, ← c116570] at e_1909_12
  have e_1909_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_14
  simp only [← c116574, ← c116575, ← c116576] at e_1909_14
  have e_1909_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_16
  simp only [← c116580, ← c116581, ← c116582] at e_1909_16
  have e_1909_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1909 (by decide)
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
  have e_1909_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1909 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_18
  simp only [← c116586, ← c116587, ← c116588] at e_1909_18
  have e_1909_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1909 (by decide)
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
  have e_1910_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_0
  simp only [← c116592, ← c116593, ← c116594] at e_1910_0
  have e_1910_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_2
  simp only [← c116598, ← c116599, ← c116600] at e_1910_2
  have e_1910_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_4
  simp only [← c116604, ← c116605, ← c116606] at e_1910_4
  have e_1910_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_6
  simp only [← c116610, ← c116611, ← c116612] at e_1910_6
  have e_1910_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_8
  simp only [← c116616, ← c116617, ← c116618] at e_1910_8
  have e_1910_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_10
  simp only [← c116622, ← c116623, ← c116624] at e_1910_10
  have e_1910_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_12
  simp only [← c116628, ← c116629, ← c116630] at e_1910_12
  have e_1910_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_14
  simp only [← c116634, ← c116635, ← c116636] at e_1910_14
  have e_1910_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_16
  simp only [← c116640, ← c116641, ← c116642] at e_1910_16
  have e_1910_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1910 (by decide)
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
  have e_1910_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1910 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_18
  simp only [← c116646, ← c116647, ← c116648] at e_1910_18
  have e_1910_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1910 (by decide)
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
  have e_1911_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_0
  simp only [← c116652, ← c116653, ← c116654] at e_1911_0
  have e_1911_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_2
  simp only [← c116658, ← c116659, ← c116660] at e_1911_2
  have e_1911_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_4
  simp only [← c116664, ← c116665, ← c116666] at e_1911_4
  have e_1911_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_6
  simp only [← c116670, ← c116671, ← c116672] at e_1911_6
  have e_1911_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_8
  simp only [← c116676, ← c116677, ← c116678] at e_1911_8
  have e_1911_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_10
  simp only [← c116682, ← c116683, ← c116684] at e_1911_10
  have e_1911_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_12
  simp only [← c116688, ← c116689, ← c116690] at e_1911_12
  have e_1911_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_14
  simp only [← c116694, ← c116695, ← c116696] at e_1911_14
  have e_1911_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_16
  simp only [← c116700, ← c116701, ← c116702] at e_1911_16
  have e_1911_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1911 (by decide)
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
  have e_1911_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1911 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_18
  simp only [← c116706, ← c116707, ← c116708] at e_1911_18
  have e_1911_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1911 (by decide)
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
  have e_1912_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_0
  simp only [← c116712, ← c116713, ← c116714] at e_1912_0
  have e_1912_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_2
  simp only [← c116718, ← c116719, ← c116720] at e_1912_2
  have e_1912_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_4
  simp only [← c116724, ← c116725, ← c116726] at e_1912_4
  have e_1912_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_6
  simp only [← c116730, ← c116731, ← c116732] at e_1912_6
  have e_1912_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_8
  simp only [← c116736, ← c116737, ← c116738] at e_1912_8
  have e_1912_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_10
  simp only [← c116742, ← c116743, ← c116744] at e_1912_10
  have e_1912_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_12
  simp only [← c116748, ← c116749, ← c116750] at e_1912_12
  have e_1912_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_14
  simp only [← c116754, ← c116755, ← c116756] at e_1912_14
  have e_1912_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_16
  simp only [← c116760, ← c116761, ← c116762] at e_1912_16
  have e_1912_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1912 (by decide)
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
  have e_1912_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1912 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_18
  simp only [← c116766, ← c116767, ← c116768] at e_1912_18
  have e_1912_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1912 (by decide)
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
  have e_1913_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_0
  simp only [← c116772, ← c116773, ← c116774] at e_1913_0
  have e_1913_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_2
  simp only [← c116778, ← c116779, ← c116780] at e_1913_2
  have e_1913_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_4
  simp only [← c116784, ← c116785, ← c116786] at e_1913_4
  have e_1913_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_6
  simp only [← c116790, ← c116791, ← c116792] at e_1913_6
  have e_1913_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_8
  simp only [← c116796, ← c116797, ← c116798] at e_1913_8
  have e_1913_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_10
  simp only [← c116802, ← c116803, ← c116804] at e_1913_10
  have e_1913_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_12
  simp only [← c116808, ← c116809, ← c116810] at e_1913_12
  have e_1913_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_14
  simp only [← c116814, ← c116815, ← c116816] at e_1913_14
  have e_1913_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_16
  simp only [← c116820, ← c116821, ← c116822] at e_1913_16
  have e_1913_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1913 (by decide)
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
  have e_1913_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1913 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_18
  simp only [← c116826, ← c116827, ← c116828] at e_1913_18
  have e_1913_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1913 (by decide)
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
  have e_1914_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_0
  simp only [← c116832, ← c116833, ← c116834] at e_1914_0
  have e_1914_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_2
  simp only [← c116838, ← c116839, ← c116840] at e_1914_2
  have e_1914_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_4
  simp only [← c116844, ← c116845, ← c116846] at e_1914_4
  have e_1914_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_6
  simp only [← c116850, ← c116851, ← c116852] at e_1914_6
  have e_1914_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_8
  simp only [← c116856, ← c116857, ← c116858] at e_1914_8
  have e_1914_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_10
  simp only [← c116862, ← c116863, ← c116864] at e_1914_10
  have e_1914_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_12
  simp only [← c116868, ← c116869, ← c116870] at e_1914_12
  have e_1914_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_14
  simp only [← c116874, ← c116875, ← c116876] at e_1914_14
  have e_1914_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_16
  simp only [← c116880, ← c116881, ← c116882] at e_1914_16
  have e_1914_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1914 (by decide)
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
  have e_1914_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1914 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_18
  simp only [← c116886, ← c116887, ← c116888] at e_1914_18
  have e_1914_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1914 (by decide)
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
  have e_1915_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_0
  simp only [← c116892, ← c116893, ← c116894] at e_1915_0
  have e_1915_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_2
  simp only [← c116898, ← c116899, ← c116900] at e_1915_2
  have e_1915_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_4
  simp only [← c116904, ← c116905, ← c116906] at e_1915_4
  have e_1915_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_6
  simp only [← c116910, ← c116911, ← c116912] at e_1915_6
  have e_1915_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_8
  simp only [← c116916, ← c116917, ← c116918] at e_1915_8
  have e_1915_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_10
  simp only [← c116922, ← c116923, ← c116924] at e_1915_10
  have e_1915_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_12
  simp only [← c116928, ← c116929, ← c116930] at e_1915_12
  have e_1915_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_14
  simp only [← c116934, ← c116935, ← c116936] at e_1915_14
  have e_1915_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_16
  simp only [← c116940, ← c116941, ← c116942] at e_1915_16
  have e_1915_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1915 (by decide)
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
  have e_1915_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1915 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_18
  simp only [← c116946, ← c116947, ← c116948] at e_1915_18
  have e_1915_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1915 (by decide)
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
  have e_1916_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_0
  simp only [← c116952, ← c116953, ← c116954] at e_1916_0
  have e_1916_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_2
  simp only [← c116958, ← c116959, ← c116960] at e_1916_2
  have e_1916_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_4
  simp only [← c116964, ← c116965, ← c116966] at e_1916_4
  have e_1916_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_6
  simp only [← c116970, ← c116971, ← c116972] at e_1916_6
  have e_1916_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_8
  simp only [← c116976, ← c116977, ← c116978] at e_1916_8
  have e_1916_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_10
  simp only [← c116982, ← c116983, ← c116984] at e_1916_10
  have e_1916_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_12
  simp only [← c116988, ← c116989, ← c116990] at e_1916_12
  have e_1916_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_14
  simp only [← c116994, ← c116995, ← c116996] at e_1916_14
  have e_1916_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_16
  simp only [← c117000, ← c117001, ← c117002] at e_1916_16
  have e_1916_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1916 (by decide)
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
  have e_1916_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1916 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_18
  simp only [← c117006, ← c117007, ← c117008] at e_1916_18
  have e_1916_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1916 (by decide)
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
  have e_1917_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_0
  simp only [← c117012, ← c117013, ← c117014] at e_1917_0
  have e_1917_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_2
  simp only [← c117018, ← c117019, ← c117020] at e_1917_2
  have e_1917_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_4
  simp only [← c117024, ← c117025, ← c117026] at e_1917_4
  have e_1917_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_6
  simp only [← c117030, ← c117031, ← c117032] at e_1917_6
  have e_1917_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_8
  simp only [← c117036, ← c117037, ← c117038] at e_1917_8
  have e_1917_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_10
  simp only [← c117042, ← c117043, ← c117044] at e_1917_10
  have e_1917_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_12
  simp only [← c117048, ← c117049, ← c117050] at e_1917_12
  have e_1917_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_14
  simp only [← c117054, ← c117055, ← c117056] at e_1917_14
  have e_1917_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_16
  simp only [← c117060, ← c117061, ← c117062] at e_1917_16
  have e_1917_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1917 (by decide)
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
  have e_1917_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1917 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_18
  simp only [← c117066, ← c117067, ← c117068] at e_1917_18
  have e_1917_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1917 (by decide)
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
  have e_1918_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_0
  simp only [← c117072, ← c117073, ← c117074] at e_1918_0
  have e_1918_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_2
  simp only [← c117078, ← c117079, ← c117080] at e_1918_2
  have e_1918_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_4
  simp only [← c117084, ← c117085, ← c117086] at e_1918_4
  have e_1918_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_6
  simp only [← c117090, ← c117091, ← c117092] at e_1918_6
  have e_1918_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_8
  simp only [← c117096, ← c117097, ← c117098] at e_1918_8
  have e_1918_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_10
  simp only [← c117102, ← c117103, ← c117104] at e_1918_10
  have e_1918_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_12
  simp only [← c117108, ← c117109, ← c117110] at e_1918_12
  have e_1918_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_14
  simp only [← c117114, ← c117115, ← c117116] at e_1918_14
  have e_1918_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_16
  simp only [← c117120, ← c117121, ← c117122] at e_1918_16
  have e_1918_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1918 (by decide)
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
  have e_1918_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1918 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_18
  simp only [← c117126, ← c117127, ← c117128] at e_1918_18
  have e_1918_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1918 (by decide)
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
  have e_1919_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_0
  simp only [← c117132, ← c117133, ← c117134] at e_1919_0
  have e_1919_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_2
  simp only [← c117138, ← c117139, ← c117140] at e_1919_2
  have e_1919_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_4
  simp only [← c117144, ← c117145, ← c117146] at e_1919_4
  have e_1919_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_6
  simp only [← c117150, ← c117151, ← c117152] at e_1919_6
  have e_1919_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_8
  simp only [← c117156, ← c117157, ← c117158] at e_1919_8
  have e_1919_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_10
  simp only [← c117162, ← c117163, ← c117164] at e_1919_10
  have e_1919_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_12
  simp only [← c117168, ← c117169, ← c117170] at e_1919_12
  have e_1919_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_14
  simp only [← c117174, ← c117175, ← c117176] at e_1919_14
  have e_1919_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_16
  simp only [← c117180, ← c117181, ← c117182] at e_1919_16
  have e_1919_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1919 (by decide)
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
  have e_1919_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1919 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_18
  simp only [← c117186, ← c117187, ← c117188] at e_1919_18
  have e_1919_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1919 (by decide)
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
  have e_1920_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_0
  simp only [← c117192, ← c117193, ← c117194] at e_1920_0
  have e_1920_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_2
  simp only [← c117198, ← c117199, ← c117200] at e_1920_2
  have e_1920_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_4
  simp only [← c117204, ← c117205, ← c117206] at e_1920_4
  have e_1920_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_6
  simp only [← c117210, ← c117211, ← c117212] at e_1920_6
  have e_1920_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_8
  simp only [← c117216, ← c117217, ← c117218] at e_1920_8
  have e_1920_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_10
  simp only [← c117222, ← c117223, ← c117224] at e_1920_10
  have e_1920_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_12
  simp only [← c117228, ← c117229, ← c117230] at e_1920_12
  have e_1920_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_14
  simp only [← c117234, ← c117235, ← c117236] at e_1920_14
  have e_1920_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_16
  simp only [← c117240, ← c117241, ← c117242] at e_1920_16
  have e_1920_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1920 (by decide)
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
  have e_1920_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1920 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_18
  simp only [← c117246, ← c117247, ← c117248] at e_1920_18
  have e_1920_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1920 (by decide)
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
  have e_1921_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_0
  simp only [← c117252, ← c117253, ← c117254] at e_1921_0
  have e_1921_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_2
  simp only [← c117258, ← c117259, ← c117260] at e_1921_2
  have e_1921_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_4
  simp only [← c117264, ← c117265, ← c117266] at e_1921_4
  have e_1921_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_6
  simp only [← c117270, ← c117271, ← c117272] at e_1921_6
  have e_1921_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_8
  simp only [← c117276, ← c117277, ← c117278] at e_1921_8
  have e_1921_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_10
  simp only [← c117282, ← c117283, ← c117284] at e_1921_10
  have e_1921_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_12
  simp only [← c117288, ← c117289, ← c117290] at e_1921_12
  have e_1921_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_14
  simp only [← c117294, ← c117295, ← c117296] at e_1921_14
  have e_1921_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_16
  simp only [← c117300, ← c117301, ← c117302] at e_1921_16
  have e_1921_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1921 (by decide)
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
  have e_1921_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1921 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_18
  simp only [← c117306, ← c117307, ← c117308] at e_1921_18
  have e_1921_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1921 (by decide)
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
  have e_1922_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_0
  simp only [← c117312, ← c117313, ← c117314] at e_1922_0
  have e_1922_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_2
  simp only [← c117318, ← c117319, ← c117320] at e_1922_2
  have e_1922_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_4
  simp only [← c117324, ← c117325, ← c117326] at e_1922_4
  have e_1922_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_6
  simp only [← c117330, ← c117331, ← c117332] at e_1922_6
  have e_1922_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_8
  simp only [← c117336, ← c117337, ← c117338] at e_1922_8
  have e_1922_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_10
  simp only [← c117342, ← c117343, ← c117344] at e_1922_10
  have e_1922_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_12
  simp only [← c117348, ← c117349, ← c117350] at e_1922_12
  have e_1922_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_14
  simp only [← c117354, ← c117355, ← c117356] at e_1922_14
  have e_1922_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_16
  simp only [← c117360, ← c117361, ← c117362] at e_1922_16
  have e_1922_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1922 (by decide)
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
  have e_1922_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1922 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_18
  simp only [← c117366, ← c117367, ← c117368] at e_1922_18
  have e_1922_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1922 (by decide)
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
  have e_1923_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_0
  simp only [← c117372, ← c117373, ← c117374] at e_1923_0
  have e_1923_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_2
  simp only [← c117378, ← c117379, ← c117380] at e_1923_2
  have e_1923_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_4
  simp only [← c117384, ← c117385, ← c117386] at e_1923_4
  have e_1923_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_6
  simp only [← c117390, ← c117391, ← c117392] at e_1923_6
  have e_1923_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_8
  simp only [← c117396, ← c117397, ← c117398] at e_1923_8
  have e_1923_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_10
  simp only [← c117402, ← c117403, ← c117404] at e_1923_10
  have e_1923_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_12
  simp only [← c117408, ← c117409, ← c117410] at e_1923_12
  have e_1923_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_14
  simp only [← c117414, ← c117415, ← c117416] at e_1923_14
  have e_1923_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_16
  simp only [← c117420, ← c117421, ← c117422] at e_1923_16
  have e_1923_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1923 (by decide)
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
  have e_1923_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1923 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_18
  simp only [← c117426, ← c117427, ← c117428] at e_1923_18
  have e_1923_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1923 (by decide)
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
  have e_1924_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_0
  simp only [← c117432, ← c117433, ← c117434] at e_1924_0
  have e_1924_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_2
  simp only [← c117438, ← c117439, ← c117440] at e_1924_2
  have e_1924_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_4
  simp only [← c117444, ← c117445, ← c117446] at e_1924_4
  have e_1924_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_6
  simp only [← c117450, ← c117451, ← c117452] at e_1924_6
  have e_1924_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_8
  simp only [← c117456, ← c117457, ← c117458] at e_1924_8
  have e_1924_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_10
  simp only [← c117462, ← c117463, ← c117464] at e_1924_10
  have e_1924_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_12
  simp only [← c117468, ← c117469, ← c117470] at e_1924_12
  have e_1924_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_14
  simp only [← c117474, ← c117475, ← c117476] at e_1924_14
  have e_1924_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_16
  simp only [← c117480, ← c117481, ← c117482] at e_1924_16
  have e_1924_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1924 (by decide)
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
  have e_1924_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1924 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_18
  simp only [← c117486, ← c117487, ← c117488] at e_1924_18
  have e_1924_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1924 (by decide)
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
  have e_1925_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_0
  simp only [← c117492, ← c117493, ← c117494] at e_1925_0
  have e_1925_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_2
  simp only [← c117498, ← c117499, ← c117500] at e_1925_2
  have e_1925_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_4
  simp only [← c117504, ← c117505, ← c117506] at e_1925_4
  have e_1925_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_6
  simp only [← c117510, ← c117511, ← c117512] at e_1925_6
  have e_1925_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_8
  simp only [← c117516, ← c117517, ← c117518] at e_1925_8
  have e_1925_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_10
  simp only [← c117522, ← c117523, ← c117524] at e_1925_10
  have e_1925_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_12
  simp only [← c117528, ← c117529, ← c117530] at e_1925_12
  have e_1925_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_14
  simp only [← c117534, ← c117535, ← c117536] at e_1925_14
  have e_1925_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_16
  simp only [← c117540, ← c117541, ← c117542] at e_1925_16
  have e_1925_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1925 (by decide)
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
  have e_1925_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1925 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_18
  simp only [← c117546, ← c117547, ← c117548] at e_1925_18
  have e_1925_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1925 (by decide)
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
  have e_1926_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_0
  simp only [← c117552, ← c117553, ← c117554] at e_1926_0
  have e_1926_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_2
  simp only [← c117558, ← c117559, ← c117560] at e_1926_2
  have e_1926_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_4
  simp only [← c117564, ← c117565, ← c117566] at e_1926_4
  have e_1926_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_6
  simp only [← c117570, ← c117571, ← c117572] at e_1926_6
  have e_1926_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_8
  simp only [← c117576, ← c117577, ← c117578] at e_1926_8
  have e_1926_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_10
  simp only [← c117582, ← c117583, ← c117584] at e_1926_10
  have e_1926_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_12
  simp only [← c117588, ← c117589, ← c117590] at e_1926_12
  have e_1926_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_14
  simp only [← c117594, ← c117595, ← c117596] at e_1926_14
  have e_1926_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_16
  simp only [← c117600, ← c117601, ← c117602] at e_1926_16
  have e_1926_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1926 (by decide)
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
  have e_1926_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1926 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_18
  simp only [← c117606, ← c117607, ← c117608] at e_1926_18
  have e_1926_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1926 (by decide)
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
  have e_1927_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_0
  simp only [← c117612, ← c117613, ← c117614] at e_1927_0
  have e_1927_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_2
  simp only [← c117618, ← c117619, ← c117620] at e_1927_2
  have e_1927_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_4
  simp only [← c117624, ← c117625, ← c117626] at e_1927_4
  have e_1927_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_6
  simp only [← c117630, ← c117631, ← c117632] at e_1927_6
  have e_1927_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_8
  simp only [← c117636, ← c117637, ← c117638] at e_1927_8
  have e_1927_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_10
  simp only [← c117642, ← c117643, ← c117644] at e_1927_10
  have e_1927_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_12
  simp only [← c117648, ← c117649, ← c117650] at e_1927_12
  have e_1927_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_14
  simp only [← c117654, ← c117655, ← c117656] at e_1927_14
  have e_1927_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_16
  simp only [← c117660, ← c117661, ← c117662] at e_1927_16
  have e_1927_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1927 (by decide)
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
  have e_1927_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1927 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_18
  simp only [← c117666, ← c117667, ← c117668] at e_1927_18
  have e_1927_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1927 (by decide)
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
  have e_1928_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_0
  simp only [← c117672, ← c117673, ← c117674] at e_1928_0
  have e_1928_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_2
  simp only [← c117678, ← c117679, ← c117680] at e_1928_2
  have e_1928_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_4
  simp only [← c117684, ← c117685, ← c117686] at e_1928_4
  have e_1928_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_6
  simp only [← c117690, ← c117691, ← c117692] at e_1928_6
  have e_1928_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_8
  simp only [← c117696, ← c117697, ← c117698] at e_1928_8
  have e_1928_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_10
  simp only [← c117702, ← c117703, ← c117704] at e_1928_10
  have e_1928_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_12
  simp only [← c117708, ← c117709, ← c117710] at e_1928_12
  have e_1928_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_14
  simp only [← c117714, ← c117715, ← c117716] at e_1928_14
  have e_1928_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_16
  simp only [← c117720, ← c117721, ← c117722] at e_1928_16
  have e_1928_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1928 (by decide)
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
  have e_1928_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1928 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_18
  simp only [← c117726, ← c117727, ← c117728] at e_1928_18
  have e_1928_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1928 (by decide)
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
  have e_1929_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_0
  simp only [← c117732, ← c117733, ← c117734] at e_1929_0
  have e_1929_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_2
  simp only [← c117738, ← c117739, ← c117740] at e_1929_2
  have e_1929_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_4
  simp only [← c117744, ← c117745, ← c117746] at e_1929_4
  have e_1929_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_6
  simp only [← c117750, ← c117751, ← c117752] at e_1929_6
  have e_1929_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_8
  simp only [← c117756, ← c117757, ← c117758] at e_1929_8
  have e_1929_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_10
  simp only [← c117762, ← c117763, ← c117764] at e_1929_10
  have e_1929_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_12
  simp only [← c117768, ← c117769, ← c117770] at e_1929_12
  have e_1929_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_14
  simp only [← c117774, ← c117775, ← c117776] at e_1929_14
  have e_1929_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_16
  simp only [← c117780, ← c117781, ← c117782] at e_1929_16
  have e_1929_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1929 (by decide)
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
  have e_1929_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1929 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_18
  simp only [← c117786, ← c117787, ← c117788] at e_1929_18
  have e_1929_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1929 (by decide)
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
  have e_1930_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_0
  simp only [← c117792, ← c117793, ← c117794] at e_1930_0
  have e_1930_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_2
  simp only [← c117798, ← c117799, ← c117800] at e_1930_2
  have e_1930_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_4
  simp only [← c117804, ← c117805, ← c117806] at e_1930_4
  have e_1930_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_6
  simp only [← c117810, ← c117811, ← c117812] at e_1930_6
  have e_1930_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_8
  simp only [← c117816, ← c117817, ← c117818] at e_1930_8
  have e_1930_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_10
  simp only [← c117822, ← c117823, ← c117824] at e_1930_10
  have e_1930_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_12
  simp only [← c117828, ← c117829, ← c117830] at e_1930_12
  have e_1930_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_14
  simp only [← c117834, ← c117835, ← c117836] at e_1930_14
  have e_1930_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_16
  simp only [← c117840, ← c117841, ← c117842] at e_1930_16
  have e_1930_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1930 (by decide)
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
  have e_1930_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1930 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_18
  simp only [← c117846, ← c117847, ← c117848] at e_1930_18
  have e_1930_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1930 (by decide)
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
  have e_1931_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_0
  simp only [← c117852, ← c117853, ← c117854] at e_1931_0
  have e_1931_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_2
  simp only [← c117858, ← c117859, ← c117860] at e_1931_2
  have e_1931_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_4
  simp only [← c117864, ← c117865, ← c117866] at e_1931_4
  have e_1931_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_6
  simp only [← c117870, ← c117871, ← c117872] at e_1931_6
  have e_1931_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_8
  simp only [← c117876, ← c117877, ← c117878] at e_1931_8
  have e_1931_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_10
  simp only [← c117882, ← c117883, ← c117884] at e_1931_10
  have e_1931_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_12
  simp only [← c117888, ← c117889, ← c117890] at e_1931_12
  have e_1931_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_14
  simp only [← c117894, ← c117895, ← c117896] at e_1931_14
  have e_1931_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_16
  simp only [← c117900, ← c117901, ← c117902] at e_1931_16
  have e_1931_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1931 (by decide)
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
  have e_1931_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1931 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_18
  simp only [← c117906, ← c117907, ← c117908] at e_1931_18
  have e_1931_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1931 (by decide)
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
  have e_1932_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_0
  simp only [← c117912, ← c117913, ← c117914] at e_1932_0
  have e_1932_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_2
  simp only [← c117918, ← c117919, ← c117920] at e_1932_2
  have e_1932_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_4
  simp only [← c117924, ← c117925, ← c117926] at e_1932_4
  have e_1932_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_6
  simp only [← c117930, ← c117931, ← c117932] at e_1932_6
  have e_1932_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_8
  simp only [← c117936, ← c117937, ← c117938] at e_1932_8
  have e_1932_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_10
  simp only [← c117942, ← c117943, ← c117944] at e_1932_10
  have e_1932_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_12
  simp only [← c117948, ← c117949, ← c117950] at e_1932_12
  have e_1932_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_14
  simp only [← c117954, ← c117955, ← c117956] at e_1932_14
  have e_1932_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_16
  simp only [← c117960, ← c117961, ← c117962] at e_1932_16
  have e_1932_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1932 (by decide)
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
  have e_1932_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1932 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_18
  simp only [← c117966, ← c117967, ← c117968] at e_1932_18
  have e_1932_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1932 (by decide)
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
  have e_1933_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_0
  simp only [← c117972, ← c117973, ← c117974] at e_1933_0
  have e_1933_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_2
  simp only [← c117978, ← c117979, ← c117980] at e_1933_2
  have e_1933_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_4
  simp only [← c117984, ← c117985, ← c117986] at e_1933_4
  have e_1933_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_6
  simp only [← c117990, ← c117991, ← c117992] at e_1933_6
  have e_1933_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_8
  simp only [← c117996, ← c117997, ← c117998] at e_1933_8
  have e_1933_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_10
  simp only [← c118002, ← c118003, ← c118004] at e_1933_10
  have e_1933_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_12
  simp only [← c118008, ← c118009, ← c118010] at e_1933_12
  have e_1933_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_14
  simp only [← c118014, ← c118015, ← c118016] at e_1933_14
  have e_1933_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_16
  simp only [← c118020, ← c118021, ← c118022] at e_1933_16
  have e_1933_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1933 (by decide)
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
  have e_1933_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1933 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_18
  simp only [← c118026, ← c118027, ← c118028] at e_1933_18
  have e_1933_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1933 (by decide)
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
  have e_1934_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_0
  simp only [← c118032, ← c118033, ← c118034] at e_1934_0
  have e_1934_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_2
  simp only [← c118038, ← c118039, ← c118040] at e_1934_2
  have e_1934_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_4
  simp only [← c118044, ← c118045, ← c118046] at e_1934_4
  have e_1934_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_6
  simp only [← c118050, ← c118051, ← c118052] at e_1934_6
  have e_1934_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_8
  simp only [← c118056, ← c118057, ← c118058] at e_1934_8
  have e_1934_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_10
  simp only [← c118062, ← c118063, ← c118064] at e_1934_10
  have e_1934_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_12
  simp only [← c118068, ← c118069, ← c118070] at e_1934_12
  have e_1934_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_14
  simp only [← c118074, ← c118075, ← c118076] at e_1934_14
  have e_1934_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_16
  simp only [← c118080, ← c118081, ← c118082] at e_1934_16
  have e_1934_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1934 (by decide)
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
  have e_1934_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1934 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_18
  simp only [← c118086, ← c118087, ← c118088] at e_1934_18
  have e_1934_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1934 (by decide)
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
  have e_1935_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_0
  simp only [← c118092, ← c118093, ← c118094] at e_1935_0
  have e_1935_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_2
  simp only [← c118098, ← c118099, ← c118100] at e_1935_2
  have e_1935_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_4
  simp only [← c118104, ← c118105, ← c118106] at e_1935_4
  have e_1935_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_6
  simp only [← c118110, ← c118111, ← c118112] at e_1935_6
  have e_1935_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_8
  simp only [← c118116, ← c118117, ← c118118] at e_1935_8
  have e_1935_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_10
  simp only [← c118122, ← c118123, ← c118124] at e_1935_10
  have e_1935_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_12
  simp only [← c118128, ← c118129, ← c118130] at e_1935_12
  have e_1935_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_14
  simp only [← c118134, ← c118135, ← c118136] at e_1935_14
  have e_1935_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_16
  simp only [← c118140, ← c118141, ← c118142] at e_1935_16
  have e_1935_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1935 (by decide)
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
  have e_1935_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1935 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_18
  simp only [← c118146, ← c118147, ← c118148] at e_1935_18
  have e_1935_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1935 (by decide)
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
  have e_1936_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_0
  simp only [← c118152, ← c118153, ← c118154] at e_1936_0
  have e_1936_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_2
  simp only [← c118158, ← c118159, ← c118160] at e_1936_2
  have e_1936_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_4
  simp only [← c118164, ← c118165, ← c118166] at e_1936_4
  have e_1936_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_6
  simp only [← c118170, ← c118171, ← c118172] at e_1936_6
  have e_1936_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_8
  simp only [← c118176, ← c118177, ← c118178] at e_1936_8
  have e_1936_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_10
  simp only [← c118182, ← c118183, ← c118184] at e_1936_10
  have e_1936_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_12
  simp only [← c118188, ← c118189, ← c118190] at e_1936_12
  have e_1936_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_14
  simp only [← c118194, ← c118195, ← c118196] at e_1936_14
  have e_1936_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_16
  simp only [← c118200, ← c118201, ← c118202] at e_1936_16
  have e_1936_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1936 (by decide)
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
  have e_1936_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1936 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_18
  simp only [← c118206, ← c118207, ← c118208] at e_1936_18
  have e_1936_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1936 (by decide)
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
  have e_1937_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_0
  simp only [← c118212, ← c118213, ← c118214] at e_1937_0
  have e_1937_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_2
  simp only [← c118218, ← c118219, ← c118220] at e_1937_2
  have e_1937_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_4
  simp only [← c118224, ← c118225, ← c118226] at e_1937_4
  have e_1937_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_6
  simp only [← c118230, ← c118231, ← c118232] at e_1937_6
  have e_1937_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_8
  simp only [← c118236, ← c118237, ← c118238] at e_1937_8
  have e_1937_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_10
  simp only [← c118242, ← c118243, ← c118244] at e_1937_10
  have e_1937_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_12
  simp only [← c118248, ← c118249, ← c118250] at e_1937_12
  have e_1937_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_14
  simp only [← c118254, ← c118255, ← c118256] at e_1937_14
  have e_1937_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_16
  simp only [← c118260, ← c118261, ← c118262] at e_1937_16
  have e_1937_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1937 (by decide)
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
  have e_1937_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1937 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_18
  simp only [← c118266, ← c118267, ← c118268] at e_1937_18
  have e_1937_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1937 (by decide)
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
  have e_1938_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_0
  simp only [← c118272, ← c118273, ← c118274] at e_1938_0
  have e_1938_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_2
  simp only [← c118278, ← c118279, ← c118280] at e_1938_2
  have e_1938_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_4
  simp only [← c118284, ← c118285, ← c118286] at e_1938_4
  have e_1938_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_6
  simp only [← c118290, ← c118291, ← c118292] at e_1938_6
  have e_1938_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_8
  simp only [← c118296, ← c118297, ← c118298] at e_1938_8
  have e_1938_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_10
  simp only [← c118302, ← c118303, ← c118304] at e_1938_10
  have e_1938_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_12
  simp only [← c118308, ← c118309, ← c118310] at e_1938_12
  have e_1938_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_14
  simp only [← c118314, ← c118315, ← c118316] at e_1938_14
  have e_1938_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_16
  simp only [← c118320, ← c118321, ← c118322] at e_1938_16
  have e_1938_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1938 (by decide)
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
  have e_1938_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1938 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_18
  simp only [← c118326, ← c118327, ← c118328] at e_1938_18
  have e_1938_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1938 (by decide)
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
  have e_1939_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_0
  simp only [← c118332, ← c118333, ← c118334] at e_1939_0
  have e_1939_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_2
  simp only [← c118338, ← c118339, ← c118340] at e_1939_2
  have e_1939_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_4
  simp only [← c118344, ← c118345, ← c118346] at e_1939_4
  have e_1939_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_6
  simp only [← c118350, ← c118351, ← c118352] at e_1939_6
  have e_1939_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_8
  simp only [← c118356, ← c118357, ← c118358] at e_1939_8
  have e_1939_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_10
  simp only [← c118362, ← c118363, ← c118364] at e_1939_10
  have e_1939_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_12
  simp only [← c118368, ← c118369, ← c118370] at e_1939_12
  have e_1939_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_14
  simp only [← c118374, ← c118375, ← c118376] at e_1939_14
  have e_1939_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_16
  simp only [← c118380, ← c118381, ← c118382] at e_1939_16
  have e_1939_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1939 (by decide)
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
  have e_1939_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1939 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_18
  simp only [← c118386, ← c118387, ← c118388] at e_1939_18
  have e_1939_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1939 (by decide)
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
  have e_1940_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_0
  simp only [← c118392, ← c118393, ← c118394] at e_1940_0
  have e_1940_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_2
  simp only [← c118398, ← c118399, ← c118400] at e_1940_2
  have e_1940_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_4
  simp only [← c118404, ← c118405, ← c118406] at e_1940_4
  have e_1940_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_6
  simp only [← c118410, ← c118411, ← c118412] at e_1940_6
  have e_1940_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_8
  simp only [← c118416, ← c118417, ← c118418] at e_1940_8
  have e_1940_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_10
  simp only [← c118422, ← c118423, ← c118424] at e_1940_10
  have e_1940_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_12
  simp only [← c118428, ← c118429, ← c118430] at e_1940_12
  have e_1940_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_14
  simp only [← c118434, ← c118435, ← c118436] at e_1940_14
  have e_1940_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_16
  simp only [← c118440, ← c118441, ← c118442] at e_1940_16
  have e_1940_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1940 (by decide)
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
  have e_1940_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_18
  simp only [← c118446, ← c118447, ← c118448] at e_1940_18
  have e_1940_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1940 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_19
  simp only [← c118449, ← c118450, k2, ← c118451] at e_1940_19
  have hr := e_1940_19
  simp only [e_1940_18] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
