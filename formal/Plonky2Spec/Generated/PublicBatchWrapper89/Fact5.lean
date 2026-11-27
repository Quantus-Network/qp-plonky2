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

theorem publicBatchWrapper89_f3570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 398 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 674)) := by
  have c25926 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.1
  have c25927 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.1
  have c25928 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.1
  have c25929 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.1
  have c25930 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25931 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_398_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_18
  simp only [← c25926, ← c25927, ← c25928] at e_398_18
  have e_398_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row398 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_19
  simp only [← c25929, ← c25930, k2, ← c25931] at e_398_19
  have hr := e_398_19
  simp only [e_398_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 675)) := by
  have c25932 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25933 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25934 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25935 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25936 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25937 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_0
  simp only [← c25932, ← c25933, ← c25934] at e_399_0
  have e_399_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_1
  simp only [← c25935, ← c25936, k2, ← c25937] at e_399_1
  have hr := e_399_1
  simp only [e_399_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 676)) := by
  have c25938 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25939 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25940 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25941 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25942 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25943 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_2
  simp only [← c25938, ← c25939, ← c25940] at e_399_2
  have e_399_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_3
  simp only [← c25941, ← c25942, k2, ← c25943] at e_399_3
  have hr := e_399_3
  simp only [e_399_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 677)) := by
  have c25944 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25945 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25946 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25947 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25948 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25949 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_4
  simp only [← c25944, ← c25945, ← c25946] at e_399_4
  have e_399_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_5
  simp only [← c25947, ← c25948, k2, ← c25949] at e_399_5
  have hr := e_399_5
  simp only [e_399_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 678)) := by
  have c25950 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25951 := (publicBatchWrapper89_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25952 := (publicBatchWrapper89_copies811 a h).1
  have c25953 := (publicBatchWrapper89_copies811 a h).2.1
  have c25954 := (publicBatchWrapper89_copies811 a h).2.2.1
  have c25955 := (publicBatchWrapper89_copies811 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_6
  simp only [← c25950, ← c25951, ← c25952] at e_399_6
  have e_399_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_7
  simp only [← c25953, ← c25954, k2, ← c25955] at e_399_7
  have hr := e_399_7
  simp only [e_399_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 679)) := by
  have c25956 := (publicBatchWrapper89_copies811 a h).2.2.2.2.1
  have c25957 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.1
  have c25958 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.1
  have c25959 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.1
  have c25960 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.1
  have c25961 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_8
  simp only [← c25956, ← c25957, ← c25958] at e_399_8
  have e_399_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_9
  simp only [← c25959, ← c25960, k2, ← c25961] at e_399_9
  have hr := e_399_9
  simp only [e_399_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 680)) := by
  have c25962 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25963 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25964 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25965 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25966 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25967 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_10
  simp only [← c25962, ← c25963, ← c25964] at e_399_10
  have e_399_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_11
  simp only [← c25965, ← c25966, k2, ← c25967] at e_399_11
  have hr := e_399_11
  simp only [e_399_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 681)) := by
  have c25968 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25969 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25970 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25971 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25972 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25973 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_12
  simp only [← c25968, ← c25969, ← c25970] at e_399_12
  have e_399_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_13
  simp only [← c25971, ← c25972, k2, ← c25973] at e_399_13
  have hr := e_399_13
  simp only [e_399_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 682)) := by
  have c25974 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25975 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25976 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25977 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25978 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25979 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_14
  simp only [← c25974, ← c25975, ← c25976] at e_399_14
  have e_399_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_15
  simp only [← c25977, ← c25978, k2, ← c25979] at e_399_15
  have hr := e_399_15
  simp only [e_399_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 683)) := by
  have c25980 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25981 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25982 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25983 := (publicBatchWrapper89_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25984 := (publicBatchWrapper89_copies812 a h).1
  have c25985 := (publicBatchWrapper89_copies812 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_16
  simp only [← c25980, ← c25981, ← c25982] at e_399_16
  have e_399_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_17
  simp only [← c25983, ← c25984, k2, ← c25985] at e_399_17
  have hr := e_399_17
  simp only [e_399_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 399 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 684)) := by
  have c25986 := (publicBatchWrapper89_copies812 a h).2.2.1
  have c25987 := (publicBatchWrapper89_copies812 a h).2.2.2.1
  have c25988 := (publicBatchWrapper89_copies812 a h).2.2.2.2.1
  have c25989 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.1
  have c25990 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.1
  have c25991 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_399_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_18
  simp only [← c25986, ← c25987, ← c25988] at e_399_18
  have e_399_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row399 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_19
  simp only [← c25989, ← c25990, k2, ← c25991] at e_399_19
  have hr := e_399_19
  simp only [e_399_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 685)) := by
  have c25992 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.1
  have c25993 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.1
  have c25994 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25995 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25996 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25997 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_0
  simp only [← c25992, ← c25993, ← c25994] at e_400_0
  have e_400_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_1
  simp only [← c25995, ← c25996, k2, ← c25997] at e_400_1
  have hr := e_400_1
  simp only [e_400_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 686)) := by
  have c25998 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25999 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26000 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26001 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26002 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26003 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_2
  simp only [← c25998, ← c25999, ← c26000] at e_400_2
  have e_400_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_3
  simp only [← c26001, ← c26002, k2, ← c26003] at e_400_3
  have hr := e_400_3
  simp only [e_400_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 687)) := by
  have c26004 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26005 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26006 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26007 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26008 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26009 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_4
  simp only [← c26004, ← c26005, ← c26006] at e_400_4
  have e_400_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_5
  simp only [← c26007, ← c26008, k2, ← c26009] at e_400_5
  have hr := e_400_5
  simp only [e_400_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 688)) := by
  have c26010 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26011 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26012 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26013 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26014 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26015 := (publicBatchWrapper89_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_6
  simp only [← c26010, ← c26011, ← c26012] at e_400_6
  have e_400_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_7
  simp only [← c26013, ← c26014, k2, ← c26015] at e_400_7
  have hr := e_400_7
  simp only [e_400_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 689)) := by
  have c26016 := (publicBatchWrapper89_copies813 a h).1
  have c26017 := (publicBatchWrapper89_copies813 a h).2.1
  have c26018 := (publicBatchWrapper89_copies813 a h).2.2.1
  have c26019 := (publicBatchWrapper89_copies813 a h).2.2.2.1
  have c26020 := (publicBatchWrapper89_copies813 a h).2.2.2.2.1
  have c26021 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_8
  simp only [← c26016, ← c26017, ← c26018] at e_400_8
  have e_400_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_9
  simp only [← c26019, ← c26020, k2, ← c26021] at e_400_9
  have hr := e_400_9
  simp only [e_400_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 690)) := by
  have c26022 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.1
  have c26023 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.1
  have c26024 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.1
  have c26025 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.1
  have c26026 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26027 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_10
  simp only [← c26022, ← c26023, ← c26024] at e_400_10
  have e_400_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_11
  simp only [← c26025, ← c26026, k2, ← c26027] at e_400_11
  have hr := e_400_11
  simp only [e_400_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 691)) := by
  have c26028 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26029 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26030 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26031 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26032 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26033 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_12
  simp only [← c26028, ← c26029, ← c26030] at e_400_12
  have e_400_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_13
  simp only [← c26031, ← c26032, k2, ← c26033] at e_400_13
  have hr := e_400_13
  simp only [e_400_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 692)) := by
  have c26034 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26035 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26036 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26037 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26038 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26039 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_14
  simp only [← c26034, ← c26035, ← c26036] at e_400_14
  have e_400_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_15
  simp only [← c26037, ← c26038, k2, ← c26039] at e_400_15
  have hr := e_400_15
  simp only [e_400_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 693)) := by
  have c26040 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26041 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26042 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26043 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26044 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26045 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_16
  simp only [← c26040, ← c26041, ← c26042] at e_400_16
  have e_400_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_17
  simp only [← c26043, ← c26044, k2, ← c26045] at e_400_17
  have hr := e_400_17
  simp only [e_400_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 400 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 694)) := by
  have c26046 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26047 := (publicBatchWrapper89_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26048 := (publicBatchWrapper89_copies814 a h).1
  have c26049 := (publicBatchWrapper89_copies814 a h).2.1
  have c26050 := (publicBatchWrapper89_copies814 a h).2.2.1
  have c26051 := (publicBatchWrapper89_copies814 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_400_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_18
  simp only [← c26046, ← c26047, ← c26048] at e_400_18
  have e_400_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row400 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_19
  simp only [← c26049, ← c26050, k2, ← c26051] at e_400_19
  have hr := e_400_19
  simp only [e_400_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 695)) := by
  have c26052 := (publicBatchWrapper89_copies814 a h).2.2.2.2.1
  have c26053 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.1
  have c26054 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.1
  have c26055 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.1
  have c26056 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.1
  have c26057 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_0
  simp only [← c26052, ← c26053, ← c26054] at e_401_0
  have e_401_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_1
  simp only [← c26055, ← c26056, k2, ← c26057] at e_401_1
  have hr := e_401_1
  simp only [e_401_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 696)) := by
  have c26058 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26059 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26060 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26061 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26062 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26063 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_2
  simp only [← c26058, ← c26059, ← c26060] at e_401_2
  have e_401_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_3
  simp only [← c26061, ← c26062, k2, ← c26063] at e_401_3
  have hr := e_401_3
  simp only [e_401_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 697)) := by
  have c26064 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26065 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26066 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26067 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26068 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26069 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_4
  simp only [← c26064, ← c26065, ← c26066] at e_401_4
  have e_401_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_5
  simp only [← c26067, ← c26068, k2, ← c26069] at e_401_5
  have hr := e_401_5
  simp only [e_401_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 698)) := by
  have c26070 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26071 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26072 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26073 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26074 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26075 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_6
  simp only [← c26070, ← c26071, ← c26072] at e_401_6
  have e_401_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_7
  simp only [← c26073, ← c26074, k2, ← c26075] at e_401_7
  have hr := e_401_7
  simp only [e_401_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 699)) := by
  have c26076 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26077 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26078 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26079 := (publicBatchWrapper89_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26080 := (publicBatchWrapper89_copies815 a h).1
  have c26081 := (publicBatchWrapper89_copies815 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_8
  simp only [← c26076, ← c26077, ← c26078] at e_401_8
  have e_401_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_9
  simp only [← c26079, ← c26080, k2, ← c26081] at e_401_9
  have hr := e_401_9
  simp only [e_401_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 700)) := by
  have c26082 := (publicBatchWrapper89_copies815 a h).2.2.1
  have c26083 := (publicBatchWrapper89_copies815 a h).2.2.2.1
  have c26084 := (publicBatchWrapper89_copies815 a h).2.2.2.2.1
  have c26085 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.1
  have c26086 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.1
  have c26087 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_10
  simp only [← c26082, ← c26083, ← c26084] at e_401_10
  have e_401_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_11
  simp only [← c26085, ← c26086, k2, ← c26087] at e_401_11
  have hr := e_401_11
  simp only [e_401_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 701)) := by
  have c26088 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.1
  have c26089 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.1
  have c26090 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26091 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26092 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26093 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_12
  simp only [← c26088, ← c26089, ← c26090] at e_401_12
  have e_401_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_13
  simp only [← c26091, ← c26092, k2, ← c26093] at e_401_13
  have hr := e_401_13
  simp only [e_401_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 702)) := by
  have c26094 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26095 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26096 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26097 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26098 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26099 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_14
  simp only [← c26094, ← c26095, ← c26096] at e_401_14
  have e_401_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_15
  simp only [← c26097, ← c26098, k2, ← c26099] at e_401_15
  have hr := e_401_15
  simp only [e_401_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 703)) := by
  have c26100 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26101 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26102 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26103 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26104 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26105 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_16
  simp only [← c26100, ← c26101, ← c26102] at e_401_16
  have e_401_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_17
  simp only [← c26103, ← c26104, k2, ← c26105] at e_401_17
  have hr := e_401_17
  simp only [e_401_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 401 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 704)) := by
  have c26106 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26107 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26108 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26109 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26110 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26111 := (publicBatchWrapper89_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_401_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_18
  simp only [← c26106, ← c26107, ← c26108] at e_401_18
  have e_401_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row401 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_19
  simp only [← c26109, ← c26110, k2, ← c26111] at e_401_19
  have hr := e_401_19
  simp only [e_401_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 705)) := by
  have c26112 := (publicBatchWrapper89_copies816 a h).1
  have c26113 := (publicBatchWrapper89_copies816 a h).2.1
  have c26114 := (publicBatchWrapper89_copies816 a h).2.2.1
  have c26115 := (publicBatchWrapper89_copies816 a h).2.2.2.1
  have c26116 := (publicBatchWrapper89_copies816 a h).2.2.2.2.1
  have c26117 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_0
  simp only [← c26112, ← c26113, ← c26114] at e_402_0
  have e_402_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_1
  simp only [← c26115, ← c26116, k2, ← c26117] at e_402_1
  have hr := e_402_1
  simp only [e_402_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 706)) := by
  have c26118 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.1
  have c26119 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.1
  have c26120 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.1
  have c26121 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.1
  have c26122 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26123 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_2
  simp only [← c26118, ← c26119, ← c26120] at e_402_2
  have e_402_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_3
  simp only [← c26121, ← c26122, k2, ← c26123] at e_402_3
  have hr := e_402_3
  simp only [e_402_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 707)) := by
  have c26124 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26125 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26126 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26127 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26128 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26129 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_4
  simp only [← c26124, ← c26125, ← c26126] at e_402_4
  have e_402_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_5
  simp only [← c26127, ← c26128, k2, ← c26129] at e_402_5
  have hr := e_402_5
  simp only [e_402_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 708)) := by
  have c26130 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26131 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26132 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26133 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26134 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26135 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_6
  simp only [← c26130, ← c26131, ← c26132] at e_402_6
  have e_402_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_7
  simp only [← c26133, ← c26134, k2, ← c26135] at e_402_7
  have hr := e_402_7
  simp only [e_402_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 709)) := by
  have c26136 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26137 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26138 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26139 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26140 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26141 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_8
  simp only [← c26136, ← c26137, ← c26138] at e_402_8
  have e_402_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_9
  simp only [← c26139, ← c26140, k2, ← c26141] at e_402_9
  have hr := e_402_9
  simp only [e_402_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 710)) := by
  have c26142 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26143 := (publicBatchWrapper89_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26144 := (publicBatchWrapper89_copies817 a h).1
  have c26145 := (publicBatchWrapper89_copies817 a h).2.1
  have c26146 := (publicBatchWrapper89_copies817 a h).2.2.1
  have c26147 := (publicBatchWrapper89_copies817 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_10
  simp only [← c26142, ← c26143, ← c26144] at e_402_10
  have e_402_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_11
  simp only [← c26145, ← c26146, k2, ← c26147] at e_402_11
  have hr := e_402_11
  simp only [e_402_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 711)) := by
  have c26148 := (publicBatchWrapper89_copies817 a h).2.2.2.2.1
  have c26149 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.1
  have c26150 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.1
  have c26151 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.1
  have c26152 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.1
  have c26153 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_12
  simp only [← c26148, ← c26149, ← c26150] at e_402_12
  have e_402_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_13
  simp only [← c26151, ← c26152, k2, ← c26153] at e_402_13
  have hr := e_402_13
  simp only [e_402_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 712)) := by
  have c26154 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26155 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26156 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26157 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26158 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26159 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_14
  simp only [← c26154, ← c26155, ← c26156] at e_402_14
  have e_402_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_15
  simp only [← c26157, ← c26158, k2, ← c26159] at e_402_15
  have hr := e_402_15
  simp only [e_402_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 713)) := by
  have c26160 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26161 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26162 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26163 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26164 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26165 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_16
  simp only [← c26160, ← c26161, ← c26162] at e_402_16
  have e_402_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_17
  simp only [← c26163, ← c26164, k2, ← c26165] at e_402_17
  have hr := e_402_17
  simp only [e_402_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 402 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 714)) := by
  have c26166 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26167 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26168 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26169 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26170 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26171 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_402_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_18
  simp only [← c26166, ← c26167, ← c26168] at e_402_18
  have e_402_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row402 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_19
  simp only [← c26169, ← c26170, k2, ← c26171] at e_402_19
  have hr := e_402_19
  simp only [e_402_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 715)) := by
  have c26172 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26173 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26174 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26175 := (publicBatchWrapper89_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26176 := (publicBatchWrapper89_copies818 a h).1
  have c26177 := (publicBatchWrapper89_copies818 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_0
  simp only [← c26172, ← c26173, ← c26174] at e_403_0
  have e_403_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_1
  simp only [← c26175, ← c26176, k2, ← c26177] at e_403_1
  have hr := e_403_1
  simp only [e_403_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 716)) := by
  have c26178 := (publicBatchWrapper89_copies818 a h).2.2.1
  have c26179 := (publicBatchWrapper89_copies818 a h).2.2.2.1
  have c26180 := (publicBatchWrapper89_copies818 a h).2.2.2.2.1
  have c26181 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.1
  have c26182 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.1
  have c26183 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_2
  simp only [← c26178, ← c26179, ← c26180] at e_403_2
  have e_403_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_3
  simp only [← c26181, ← c26182, k2, ← c26183] at e_403_3
  have hr := e_403_3
  simp only [e_403_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 717)) := by
  have c26184 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.1
  have c26185 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.1
  have c26186 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26187 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26188 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26189 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_4
  simp only [← c26184, ← c26185, ← c26186] at e_403_4
  have e_403_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_5
  simp only [← c26187, ← c26188, k2, ← c26189] at e_403_5
  have hr := e_403_5
  simp only [e_403_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 718)) := by
  have c26190 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26191 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26192 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26193 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26194 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26195 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_6
  simp only [← c26190, ← c26191, ← c26192] at e_403_6
  have e_403_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_7
  simp only [← c26193, ← c26194, k2, ← c26195] at e_403_7
  have hr := e_403_7
  simp only [e_403_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 719)) := by
  have c26196 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26197 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26198 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26199 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26200 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26201 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_8
  simp only [← c26196, ← c26197, ← c26198] at e_403_8
  have e_403_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_9
  simp only [← c26199, ← c26200, k2, ← c26201] at e_403_9
  have hr := e_403_9
  simp only [e_403_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 720)) := by
  have c26202 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26203 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26204 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26205 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26206 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26207 := (publicBatchWrapper89_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_10
  simp only [← c26202, ← c26203, ← c26204] at e_403_10
  have e_403_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_11
  simp only [← c26205, ← c26206, k2, ← c26207] at e_403_11
  have hr := e_403_11
  simp only [e_403_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 721)) := by
  have c26208 := (publicBatchWrapper89_copies819 a h).1
  have c26209 := (publicBatchWrapper89_copies819 a h).2.1
  have c26210 := (publicBatchWrapper89_copies819 a h).2.2.1
  have c26211 := (publicBatchWrapper89_copies819 a h).2.2.2.1
  have c26212 := (publicBatchWrapper89_copies819 a h).2.2.2.2.1
  have c26213 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_12
  simp only [← c26208, ← c26209, ← c26210] at e_403_12
  have e_403_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_13
  simp only [← c26211, ← c26212, k2, ← c26213] at e_403_13
  have hr := e_403_13
  simp only [e_403_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 722)) := by
  have c26214 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.1
  have c26215 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.1
  have c26216 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.1
  have c26217 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.1
  have c26218 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26219 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_14
  simp only [← c26214, ← c26215, ← c26216] at e_403_14
  have e_403_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_15
  simp only [← c26217, ← c26218, k2, ← c26219] at e_403_15
  have hr := e_403_15
  simp only [e_403_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 723)) := by
  have c26220 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26221 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26222 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26223 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26224 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26225 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_16
  simp only [← c26220, ← c26221, ← c26222] at e_403_16
  have e_403_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_17
  simp only [← c26223, ← c26224, k2, ← c26225] at e_403_17
  have hr := e_403_17
  simp only [e_403_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 403 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 724)) := by
  have c26226 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26227 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26228 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26229 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26230 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26231 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_403_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_18
  simp only [← c26226, ← c26227, ← c26228] at e_403_18
  have e_403_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row403 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_19
  simp only [← c26229, ← c26230, k2, ← c26231] at e_403_19
  have hr := e_403_19
  simp only [e_403_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 725)) := by
  have c26232 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26233 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26234 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26235 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26236 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26237 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_0
  simp only [← c26232, ← c26233, ← c26234] at e_404_0
  have e_404_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_1
  simp only [← c26235, ← c26236, k2, ← c26237] at e_404_1
  have hr := e_404_1
  simp only [e_404_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 726)) := by
  have c26238 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26239 := (publicBatchWrapper89_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26240 := (publicBatchWrapper89_copies820 a h).1
  have c26241 := (publicBatchWrapper89_copies820 a h).2.1
  have c26242 := (publicBatchWrapper89_copies820 a h).2.2.1
  have c26243 := (publicBatchWrapper89_copies820 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_2
  simp only [← c26238, ← c26239, ← c26240] at e_404_2
  have e_404_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_3
  simp only [← c26241, ← c26242, k2, ← c26243] at e_404_3
  have hr := e_404_3
  simp only [e_404_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 727)) := by
  have c26244 := (publicBatchWrapper89_copies820 a h).2.2.2.2.1
  have c26245 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.1
  have c26246 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.1
  have c26247 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.1
  have c26248 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.1
  have c26249 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_4
  simp only [← c26244, ← c26245, ← c26246] at e_404_4
  have e_404_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_5
  simp only [← c26247, ← c26248, k2, ← c26249] at e_404_5
  have hr := e_404_5
  simp only [e_404_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 728)) := by
  have c26250 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26251 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26252 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26253 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26254 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26255 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_6
  simp only [← c26250, ← c26251, ← c26252] at e_404_6
  have e_404_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_7
  simp only [← c26253, ← c26254, k2, ← c26255] at e_404_7
  have hr := e_404_7
  simp only [e_404_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 729)) := by
  have c26256 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26257 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26258 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26259 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26260 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26261 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_8
  simp only [← c26256, ← c26257, ← c26258] at e_404_8
  have e_404_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_9
  simp only [← c26259, ← c26260, k2, ← c26261] at e_404_9
  have hr := e_404_9
  simp only [e_404_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 730)) := by
  have c26262 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26263 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26264 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26265 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26266 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26267 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_10
  simp only [← c26262, ← c26263, ← c26264] at e_404_10
  have e_404_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_11
  simp only [← c26265, ← c26266, k2, ← c26267] at e_404_11
  have hr := e_404_11
  simp only [e_404_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 731)) := by
  have c26268 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26269 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26270 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26271 := (publicBatchWrapper89_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26272 := (publicBatchWrapper89_copies821 a h).1
  have c26273 := (publicBatchWrapper89_copies821 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_12
  simp only [← c26268, ← c26269, ← c26270] at e_404_12
  have e_404_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_13
  simp only [← c26271, ← c26272, k2, ← c26273] at e_404_13
  have hr := e_404_13
  simp only [e_404_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 732)) := by
  have c26274 := (publicBatchWrapper89_copies821 a h).2.2.1
  have c26275 := (publicBatchWrapper89_copies821 a h).2.2.2.1
  have c26276 := (publicBatchWrapper89_copies821 a h).2.2.2.2.1
  have c26277 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.1
  have c26278 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.1
  have c26279 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_14
  simp only [← c26274, ← c26275, ← c26276] at e_404_14
  have e_404_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_15
  simp only [← c26277, ← c26278, k2, ← c26279] at e_404_15
  have hr := e_404_15
  simp only [e_404_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 733)) := by
  have c26280 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.1
  have c26281 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.1
  have c26282 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26283 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26284 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26285 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_16
  simp only [← c26280, ← c26281, ← c26282] at e_404_16
  have e_404_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_17
  simp only [← c26283, ← c26284, k2, ← c26285] at e_404_17
  have hr := e_404_17
  simp only [e_404_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 404 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 734)) := by
  have c26286 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26287 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26288 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26289 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26290 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26291 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_404_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_18
  simp only [← c26286, ← c26287, ← c26288] at e_404_18
  have e_404_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row404 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_19
  simp only [← c26289, ← c26290, k2, ← c26291] at e_404_19
  have hr := e_404_19
  simp only [e_404_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 735)) := by
  have c26292 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26293 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26294 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26295 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26296 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26297 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_0
  simp only [← c26292, ← c26293, ← c26294] at e_405_0
  have e_405_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_1
  simp only [← c26295, ← c26296, k2, ← c26297] at e_405_1
  have hr := e_405_1
  simp only [e_405_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 736)) := by
  have c26298 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26299 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26300 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26301 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26302 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26303 := (publicBatchWrapper89_copies821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_2
  simp only [← c26298, ← c26299, ← c26300] at e_405_2
  have e_405_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_3
  simp only [← c26301, ← c26302, k2, ← c26303] at e_405_3
  have hr := e_405_3
  simp only [e_405_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 737)) := by
  have c26304 := (publicBatchWrapper89_copies822 a h).1
  have c26305 := (publicBatchWrapper89_copies822 a h).2.1
  have c26306 := (publicBatchWrapper89_copies822 a h).2.2.1
  have c26307 := (publicBatchWrapper89_copies822 a h).2.2.2.1
  have c26308 := (publicBatchWrapper89_copies822 a h).2.2.2.2.1
  have c26309 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_4
  simp only [← c26304, ← c26305, ← c26306] at e_405_4
  have e_405_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_5
  simp only [← c26307, ← c26308, k2, ← c26309] at e_405_5
  have hr := e_405_5
  simp only [e_405_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 738)) := by
  have c26310 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.1
  have c26311 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.1
  have c26312 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.1
  have c26313 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.1
  have c26314 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26315 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_6
  simp only [← c26310, ← c26311, ← c26312] at e_405_6
  have e_405_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_7
  simp only [← c26313, ← c26314, k2, ← c26315] at e_405_7
  have hr := e_405_7
  simp only [e_405_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 739)) := by
  have c26316 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26317 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26318 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26319 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26320 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26321 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_8
  simp only [← c26316, ← c26317, ← c26318] at e_405_8
  have e_405_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_9
  simp only [← c26319, ← c26320, k2, ← c26321] at e_405_9
  have hr := e_405_9
  simp only [e_405_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 740)) := by
  have c26322 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26323 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26324 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26325 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26326 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26327 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_10
  simp only [← c26322, ← c26323, ← c26324] at e_405_10
  have e_405_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_11
  simp only [← c26325, ← c26326, k2, ← c26327] at e_405_11
  have hr := e_405_11
  simp only [e_405_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 741)) := by
  have c26328 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26329 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26330 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26331 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26332 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26333 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_12
  simp only [← c26328, ← c26329, ← c26330] at e_405_12
  have e_405_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_13
  simp only [← c26331, ← c26332, k2, ← c26333] at e_405_13
  have hr := e_405_13
  simp only [e_405_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 742)) := by
  have c26334 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26335 := (publicBatchWrapper89_copies822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26336 := (publicBatchWrapper89_copies823 a h).1
  have c26337 := (publicBatchWrapper89_copies823 a h).2.1
  have c26338 := (publicBatchWrapper89_copies823 a h).2.2.1
  have c26339 := (publicBatchWrapper89_copies823 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_14
  simp only [← c26334, ← c26335, ← c26336] at e_405_14
  have e_405_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_15
  simp only [← c26337, ← c26338, k2, ← c26339] at e_405_15
  have hr := e_405_15
  simp only [e_405_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 743)) := by
  have c26340 := (publicBatchWrapper89_copies823 a h).2.2.2.2.1
  have c26341 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.1
  have c26342 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.1
  have c26343 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.1
  have c26344 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.1
  have c26345 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_16
  simp only [← c26340, ← c26341, ← c26342] at e_405_16
  have e_405_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_17
  simp only [← c26343, ← c26344, k2, ← c26345] at e_405_17
  have hr := e_405_17
  simp only [e_405_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 405 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 744)) := by
  have c26346 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26347 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26348 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26349 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26350 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26351 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_405_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_18
  simp only [← c26346, ← c26347, ← c26348] at e_405_18
  have e_405_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_19
  simp only [← c26349, ← c26350, k2, ← c26351] at e_405_19
  have hr := e_405_19
  simp only [e_405_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 745)) := by
  have c26352 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26353 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26354 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26355 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26356 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26357 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_0
  simp only [← c26352, ← c26353, ← c26354] at e_406_0
  have e_406_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_1
  simp only [← c26355, ← c26356, k2, ← c26357] at e_406_1
  have hr := e_406_1
  simp only [e_406_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 746)) := by
  have c26358 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26359 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26360 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26361 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26362 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26363 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_2
  simp only [← c26358, ← c26359, ← c26360] at e_406_2
  have e_406_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_3
  simp only [← c26361, ← c26362, k2, ← c26363] at e_406_3
  have hr := e_406_3
  simp only [e_406_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 747)) := by
  have c26364 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26365 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26366 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26367 := (publicBatchWrapper89_copies823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26368 := (publicBatchWrapper89_copies824 a h).1
  have c26369 := (publicBatchWrapper89_copies824 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_4
  simp only [← c26364, ← c26365, ← c26366] at e_406_4
  have e_406_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_5
  simp only [← c26367, ← c26368, k2, ← c26369] at e_406_5
  have hr := e_406_5
  simp only [e_406_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 748)) := by
  have c26370 := (publicBatchWrapper89_copies824 a h).2.2.1
  have c26371 := (publicBatchWrapper89_copies824 a h).2.2.2.1
  have c26372 := (publicBatchWrapper89_copies824 a h).2.2.2.2.1
  have c26373 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.1
  have c26374 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.1
  have c26375 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_6
  simp only [← c26370, ← c26371, ← c26372] at e_406_6
  have e_406_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_7
  simp only [← c26373, ← c26374, k2, ← c26375] at e_406_7
  have hr := e_406_7
  simp only [e_406_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 749)) := by
  have c26376 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.1
  have c26377 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.1
  have c26378 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26379 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26380 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26381 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_8
  simp only [← c26376, ← c26377, ← c26378] at e_406_8
  have e_406_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_9
  simp only [← c26379, ← c26380, k2, ← c26381] at e_406_9
  have hr := e_406_9
  simp only [e_406_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 750)) := by
  have c26382 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26383 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26384 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26385 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26386 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26387 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_10
  simp only [← c26382, ← c26383, ← c26384] at e_406_10
  have e_406_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_11
  simp only [← c26385, ← c26386, k2, ← c26387] at e_406_11
  have hr := e_406_11
  simp only [e_406_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 751)) := by
  have c26388 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26389 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26390 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26391 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26392 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26393 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_12
  simp only [← c26388, ← c26389, ← c26390] at e_406_12
  have e_406_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_13
  simp only [← c26391, ← c26392, k2, ← c26393] at e_406_13
  have hr := e_406_13
  simp only [e_406_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 752)) := by
  have c26394 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26395 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26396 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26397 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26398 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26399 := (publicBatchWrapper89_copies824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_14
  simp only [← c26394, ← c26395, ← c26396] at e_406_14
  have e_406_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_15
  simp only [← c26397, ← c26398, k2, ← c26399] at e_406_15
  have hr := e_406_15
  simp only [e_406_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 753)) := by
  have c26400 := (publicBatchWrapper89_copies825 a h).1
  have c26401 := (publicBatchWrapper89_copies825 a h).2.1
  have c26402 := (publicBatchWrapper89_copies825 a h).2.2.1
  have c26403 := (publicBatchWrapper89_copies825 a h).2.2.2.1
  have c26404 := (publicBatchWrapper89_copies825 a h).2.2.2.2.1
  have c26405 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_16
  simp only [← c26400, ← c26401, ← c26402] at e_406_16
  have e_406_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_17
  simp only [← c26403, ← c26404, k2, ← c26405] at e_406_17
  have hr := e_406_17
  simp only [e_406_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 406 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 754)) := by
  have c26406 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.1
  have c26407 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.1
  have c26408 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.1
  have c26409 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.1
  have c26410 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26411 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_406_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_18
  simp only [← c26406, ← c26407, ← c26408] at e_406_18
  have e_406_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_406_19
  simp only [← c26409, ← c26410, k2, ← c26411] at e_406_19
  have hr := e_406_19
  simp only [e_406_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 755)) := by
  have c26412 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26413 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26414 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26415 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26416 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26417 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_0
  simp only [← c26412, ← c26413, ← c26414] at e_407_0
  have e_407_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_1
  simp only [← c26415, ← c26416, k2, ← c26417] at e_407_1
  have hr := e_407_1
  simp only [e_407_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 756)) := by
  have c26418 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26419 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26420 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26421 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26422 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26423 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_2
  simp only [← c26418, ← c26419, ← c26420] at e_407_2
  have e_407_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_3
  simp only [← c26421, ← c26422, k2, ← c26423] at e_407_3
  have hr := e_407_3
  simp only [e_407_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 757)) := by
  have c26424 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26425 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26426 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26427 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26428 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26429 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_4
  simp only [← c26424, ← c26425, ← c26426] at e_407_4
  have e_407_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_5
  simp only [← c26427, ← c26428, k2, ← c26429] at e_407_5
  have hr := e_407_5
  simp only [e_407_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 758)) := by
  have c26430 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26431 := (publicBatchWrapper89_copies825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26432 := (publicBatchWrapper89_copies826 a h).1
  have c26433 := (publicBatchWrapper89_copies826 a h).2.1
  have c26434 := (publicBatchWrapper89_copies826 a h).2.2.1
  have c26435 := (publicBatchWrapper89_copies826 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_6
  simp only [← c26430, ← c26431, ← c26432] at e_407_6
  have e_407_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_7
  simp only [← c26433, ← c26434, k2, ← c26435] at e_407_7
  have hr := e_407_7
  simp only [e_407_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 759)) := by
  have c26436 := (publicBatchWrapper89_copies826 a h).2.2.2.2.1
  have c26437 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.1
  have c26438 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.1
  have c26439 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.1
  have c26440 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.1
  have c26441 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_8
  simp only [← c26436, ← c26437, ← c26438] at e_407_8
  have e_407_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_9
  simp only [← c26439, ← c26440, k2, ← c26441] at e_407_9
  have hr := e_407_9
  simp only [e_407_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 760)) := by
  have c26442 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26443 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26444 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26445 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26446 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26447 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_10
  simp only [← c26442, ← c26443, ← c26444] at e_407_10
  have e_407_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_11
  simp only [← c26445, ← c26446, k2, ← c26447] at e_407_11
  have hr := e_407_11
  simp only [e_407_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 761)) := by
  have c26448 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26449 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26450 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26451 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26452 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26453 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_12
  simp only [← c26448, ← c26449, ← c26450] at e_407_12
  have e_407_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_13
  simp only [← c26451, ← c26452, k2, ← c26453] at e_407_13
  have hr := e_407_13
  simp only [e_407_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 762)) := by
  have c26454 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26455 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26456 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26457 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26458 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26459 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_14
  simp only [← c26454, ← c26455, ← c26456] at e_407_14
  have e_407_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_15
  simp only [← c26457, ← c26458, k2, ← c26459] at e_407_15
  have hr := e_407_15
  simp only [e_407_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 763)) := by
  have c26460 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26461 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26462 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26463 := (publicBatchWrapper89_copies826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26464 := (publicBatchWrapper89_copies827 a h).1
  have c26465 := (publicBatchWrapper89_copies827 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_16
  simp only [← c26460, ← c26461, ← c26462] at e_407_16
  have e_407_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_17
  simp only [← c26463, ← c26464, k2, ← c26465] at e_407_17
  have hr := e_407_17
  simp only [e_407_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 407 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 764)) := by
  have c26466 := (publicBatchWrapper89_copies827 a h).2.2.1
  have c26467 := (publicBatchWrapper89_copies827 a h).2.2.2.1
  have c26468 := (publicBatchWrapper89_copies827 a h).2.2.2.2.1
  have c26469 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.1
  have c26470 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.1
  have c26471 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_407_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_18
  simp only [← c26466, ← c26467, ← c26468] at e_407_18
  have e_407_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_407_19
  simp only [← c26469, ← c26470, k2, ← c26471] at e_407_19
  have hr := e_407_19
  simp only [e_407_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 765)) := by
  have c26472 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.1
  have c26473 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.1
  have c26474 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26475 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26476 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26477 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_0
  simp only [← c26472, ← c26473, ← c26474] at e_408_0
  have e_408_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_1
  simp only [← c26475, ← c26476, k2, ← c26477] at e_408_1
  have hr := e_408_1
  simp only [e_408_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 766)) := by
  have c26478 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26479 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26480 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26481 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26482 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26483 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_2
  simp only [← c26478, ← c26479, ← c26480] at e_408_2
  have e_408_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_3
  simp only [← c26481, ← c26482, k2, ← c26483] at e_408_3
  have hr := e_408_3
  simp only [e_408_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 767)) := by
  have c26484 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26485 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26486 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26487 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26488 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26489 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_4
  simp only [← c26484, ← c26485, ← c26486] at e_408_4
  have e_408_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_5
  simp only [← c26487, ← c26488, k2, ← c26489] at e_408_5
  have hr := e_408_5
  simp only [e_408_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 768)) := by
  have c26490 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26491 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26492 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26493 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26494 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26495 := (publicBatchWrapper89_copies827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_6
  simp only [← c26490, ← c26491, ← c26492] at e_408_6
  have e_408_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_7
  simp only [← c26493, ← c26494, k2, ← c26495] at e_408_7
  have hr := e_408_7
  simp only [e_408_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 769)) := by
  have c26496 := (publicBatchWrapper89_copies828 a h).1
  have c26497 := (publicBatchWrapper89_copies828 a h).2.1
  have c26498 := (publicBatchWrapper89_copies828 a h).2.2.1
  have c26499 := (publicBatchWrapper89_copies828 a h).2.2.2.1
  have c26500 := (publicBatchWrapper89_copies828 a h).2.2.2.2.1
  have c26501 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_8
  simp only [← c26496, ← c26497, ← c26498] at e_408_8
  have e_408_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_9
  simp only [← c26499, ← c26500, k2, ← c26501] at e_408_9
  have hr := e_408_9
  simp only [e_408_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 810)) := by
  have c26502 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.1
  have c26503 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.1
  have c26504 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.1
  have c26505 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.1
  have c26506 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26507 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_10
  simp only [← c26502, ← c26503, ← c26504] at e_408_10
  have e_408_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_11
  simp only [← c26505, ← c26506, k2, ← c26507] at e_408_11
  have hr := e_408_11
  simp only [e_408_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 811)) := by
  have c26508 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26509 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26510 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26511 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26512 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26513 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_12
  simp only [← c26508, ← c26509, ← c26510] at e_408_12
  have e_408_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_13
  simp only [← c26511, ← c26512, k2, ← c26513] at e_408_13
  have hr := e_408_13
  simp only [e_408_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 812)) := by
  have c26514 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26515 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26516 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26517 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26518 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26519 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_14
  simp only [← c26514, ← c26515, ← c26516] at e_408_14
  have e_408_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_15
  simp only [← c26517, ← c26518, k2, ← c26519] at e_408_15
  have hr := e_408_15
  simp only [e_408_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 813)) := by
  have c26520 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26521 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26522 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26523 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26524 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26525 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_16
  simp only [← c26520, ← c26521, ← c26522] at e_408_16
  have e_408_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_17
  simp only [← c26523, ← c26524, k2, ← c26525] at e_408_17
  have hr := e_408_17
  simp only [e_408_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 408 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 814)) := by
  have c26526 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26527 := (publicBatchWrapper89_copies828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26528 := (publicBatchWrapper89_copies829 a h).1
  have c26529 := (publicBatchWrapper89_copies829 a h).2.1
  have c26530 := (publicBatchWrapper89_copies829 a h).2.2.1
  have c26531 := (publicBatchWrapper89_copies829 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_408_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_18
  simp only [← c26526, ← c26527, ← c26528] at e_408_18
  have e_408_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_408_19
  simp only [← c26529, ← c26530, k2, ← c26531] at e_408_19
  have hr := e_408_19
  simp only [e_408_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 815)) := by
  have c26532 := (publicBatchWrapper89_copies829 a h).2.2.2.2.1
  have c26533 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.1
  have c26534 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.1
  have c26535 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.1
  have c26536 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.1
  have c26537 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_0
  simp only [← c26532, ← c26533, ← c26534] at e_409_0
  have e_409_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_1
  simp only [← c26535, ← c26536, k2, ← c26537] at e_409_1
  have hr := e_409_1
  simp only [e_409_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 816)) := by
  have c26538 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26539 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26540 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26541 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26542 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26543 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_2
  simp only [← c26538, ← c26539, ← c26540] at e_409_2
  have e_409_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_3
  simp only [← c26541, ← c26542, k2, ← c26543] at e_409_3
  have hr := e_409_3
  simp only [e_409_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 817)) := by
  have c26544 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26545 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26546 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26547 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26548 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26549 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_4
  simp only [← c26544, ← c26545, ← c26546] at e_409_4
  have e_409_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_5
  simp only [← c26547, ← c26548, k2, ← c26549] at e_409_5
  have hr := e_409_5
  simp only [e_409_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 818)) := by
  have c26550 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26551 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26552 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26553 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26554 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26555 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_6
  simp only [← c26550, ← c26551, ← c26552] at e_409_6
  have e_409_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_7
  simp only [← c26553, ← c26554, k2, ← c26555] at e_409_7
  have hr := e_409_7
  simp only [e_409_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 819)) := by
  have c26556 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26557 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26558 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26559 := (publicBatchWrapper89_copies829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26560 := (publicBatchWrapper89_copies830 a h).1
  have c26561 := (publicBatchWrapper89_copies830 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_8
  simp only [← c26556, ← c26557, ← c26558] at e_409_8
  have e_409_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_9
  simp only [← c26559, ← c26560, k2, ← c26561] at e_409_9
  have hr := e_409_9
  simp only [e_409_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 820)) := by
  have c26562 := (publicBatchWrapper89_copies830 a h).2.2.1
  have c26563 := (publicBatchWrapper89_copies830 a h).2.2.2.1
  have c26564 := (publicBatchWrapper89_copies830 a h).2.2.2.2.1
  have c26565 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.1
  have c26566 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.1
  have c26567 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_10
  simp only [← c26562, ← c26563, ← c26564] at e_409_10
  have e_409_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_11
  simp only [← c26565, ← c26566, k2, ← c26567] at e_409_11
  have hr := e_409_11
  simp only [e_409_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 821)) := by
  have c26568 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.1
  have c26569 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.1
  have c26570 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26571 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26572 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26573 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_12
  simp only [← c26568, ← c26569, ← c26570] at e_409_12
  have e_409_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_13
  simp only [← c26571, ← c26572, k2, ← c26573] at e_409_13
  have hr := e_409_13
  simp only [e_409_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 822)) := by
  have c26574 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26575 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26576 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26577 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26578 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26579 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_14
  simp only [← c26574, ← c26575, ← c26576] at e_409_14
  have e_409_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_15
  simp only [← c26577, ← c26578, k2, ← c26579] at e_409_15
  have hr := e_409_15
  simp only [e_409_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 823)) := by
  have c26580 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26581 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26582 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26583 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26584 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26585 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_16
  simp only [← c26580, ← c26581, ← c26582] at e_409_16
  have e_409_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_17
  simp only [← c26583, ← c26584, k2, ← c26585] at e_409_17
  have hr := e_409_17
  simp only [e_409_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 409 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 824)) := by
  have c26586 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26587 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26588 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26589 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26590 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26591 := (publicBatchWrapper89_copies830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_409_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_18
  simp only [← c26586, ← c26587, ← c26588] at e_409_18
  have e_409_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_409_19
  simp only [← c26589, ← c26590, k2, ← c26591] at e_409_19
  have hr := e_409_19
  simp only [e_409_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 825)) := by
  have c26592 := (publicBatchWrapper89_copies831 a h).1
  have c26593 := (publicBatchWrapper89_copies831 a h).2.1
  have c26594 := (publicBatchWrapper89_copies831 a h).2.2.1
  have c26595 := (publicBatchWrapper89_copies831 a h).2.2.2.1
  have c26596 := (publicBatchWrapper89_copies831 a h).2.2.2.2.1
  have c26597 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_0
  simp only [← c26592, ← c26593, ← c26594] at e_410_0
  have e_410_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_1
  simp only [← c26595, ← c26596, k2, ← c26597] at e_410_1
  have hr := e_410_1
  simp only [e_410_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 826)) := by
  have c26598 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.1
  have c26599 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.1
  have c26600 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.1
  have c26601 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.1
  have c26602 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26603 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_2
  simp only [← c26598, ← c26599, ← c26600] at e_410_2
  have e_410_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_3
  simp only [← c26601, ← c26602, k2, ← c26603] at e_410_3
  have hr := e_410_3
  simp only [e_410_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 827)) := by
  have c26604 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26605 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26606 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26607 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26608 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26609 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_4
  simp only [← c26604, ← c26605, ← c26606] at e_410_4
  have e_410_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_5
  simp only [← c26607, ← c26608, k2, ← c26609] at e_410_5
  have hr := e_410_5
  simp only [e_410_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 828)) := by
  have c26610 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26611 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26612 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26613 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26614 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26615 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_6
  simp only [← c26610, ← c26611, ← c26612] at e_410_6
  have e_410_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_7
  simp only [← c26613, ← c26614, k2, ← c26615] at e_410_7
  have hr := e_410_7
  simp only [e_410_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 829)) := by
  have c26616 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26617 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26618 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26619 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26620 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26621 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_8
  simp only [← c26616, ← c26617, ← c26618] at e_410_8
  have e_410_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_9
  simp only [← c26619, ← c26620, k2, ← c26621] at e_410_9
  have hr := e_410_9
  simp only [e_410_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 830)) := by
  have c26622 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26623 := (publicBatchWrapper89_copies831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26624 := (publicBatchWrapper89_copies832 a h).1
  have c26625 := (publicBatchWrapper89_copies832 a h).2.1
  have c26626 := (publicBatchWrapper89_copies832 a h).2.2.1
  have c26627 := (publicBatchWrapper89_copies832 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_10
  simp only [← c26622, ← c26623, ← c26624] at e_410_10
  have e_410_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_11
  simp only [← c26625, ← c26626, k2, ← c26627] at e_410_11
  have hr := e_410_11
  simp only [e_410_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 831)) := by
  have c26628 := (publicBatchWrapper89_copies832 a h).2.2.2.2.1
  have c26629 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.1
  have c26630 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.1
  have c26631 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.1
  have c26632 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.1
  have c26633 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_12
  simp only [← c26628, ← c26629, ← c26630] at e_410_12
  have e_410_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_13
  simp only [← c26631, ← c26632, k2, ← c26633] at e_410_13
  have hr := e_410_13
  simp only [e_410_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 832)) := by
  have c26634 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26635 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26636 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26637 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26638 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26639 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_14
  simp only [← c26634, ← c26635, ← c26636] at e_410_14
  have e_410_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_15
  simp only [← c26637, ← c26638, k2, ← c26639] at e_410_15
  have hr := e_410_15
  simp only [e_410_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 833)) := by
  have c26640 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26641 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26642 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26643 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26644 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26645 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_16
  simp only [← c26640, ← c26641, ← c26642] at e_410_16
  have e_410_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_17
  simp only [← c26643, ← c26644, k2, ← c26645] at e_410_17
  have hr := e_410_17
  simp only [e_410_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 410 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 834)) := by
  have c26646 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26647 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26648 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26649 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26650 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26651 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_410_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_18
  simp only [← c26646, ← c26647, ← c26648] at e_410_18
  have e_410_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_410_19
  simp only [← c26649, ← c26650, k2, ← c26651] at e_410_19
  have hr := e_410_19
  simp only [e_410_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 835)) := by
  have c26652 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26653 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26654 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26655 := (publicBatchWrapper89_copies832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26656 := (publicBatchWrapper89_copies833 a h).1
  have c26657 := (publicBatchWrapper89_copies833 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_0
  simp only [← c26652, ← c26653, ← c26654] at e_411_0
  have e_411_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_1
  simp only [← c26655, ← c26656, k2, ← c26657] at e_411_1
  have hr := e_411_1
  simp only [e_411_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 836)) := by
  have c26658 := (publicBatchWrapper89_copies833 a h).2.2.1
  have c26659 := (publicBatchWrapper89_copies833 a h).2.2.2.1
  have c26660 := (publicBatchWrapper89_copies833 a h).2.2.2.2.1
  have c26661 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.1
  have c26662 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.1
  have c26663 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_2
  simp only [← c26658, ← c26659, ← c26660] at e_411_2
  have e_411_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_3
  simp only [← c26661, ← c26662, k2, ← c26663] at e_411_3
  have hr := e_411_3
  simp only [e_411_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 837)) := by
  have c26664 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.1
  have c26665 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.1
  have c26666 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26667 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26668 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26669 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_4
  simp only [← c26664, ← c26665, ← c26666] at e_411_4
  have e_411_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_5
  simp only [← c26667, ← c26668, k2, ← c26669] at e_411_5
  have hr := e_411_5
  simp only [e_411_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 838)) := by
  have c26670 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26671 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26672 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26673 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26674 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26675 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_6
  simp only [← c26670, ← c26671, ← c26672] at e_411_6
  have e_411_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_7
  simp only [← c26673, ← c26674, k2, ← c26675] at e_411_7
  have hr := e_411_7
  simp only [e_411_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 839)) := by
  have c26676 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26677 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26678 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26679 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26680 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26681 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_8
  simp only [← c26676, ← c26677, ← c26678] at e_411_8
  have e_411_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_9
  simp only [← c26679, ← c26680, k2, ← c26681] at e_411_9
  have hr := e_411_9
  simp only [e_411_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 840)) := by
  have c26682 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26683 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26684 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26685 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26686 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26687 := (publicBatchWrapper89_copies833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_10
  simp only [← c26682, ← c26683, ← c26684] at e_411_10
  have e_411_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_11
  simp only [← c26685, ← c26686, k2, ← c26687] at e_411_11
  have hr := e_411_11
  simp only [e_411_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 841)) := by
  have c26688 := (publicBatchWrapper89_copies834 a h).1
  have c26689 := (publicBatchWrapper89_copies834 a h).2.1
  have c26690 := (publicBatchWrapper89_copies834 a h).2.2.1
  have c26691 := (publicBatchWrapper89_copies834 a h).2.2.2.1
  have c26692 := (publicBatchWrapper89_copies834 a h).2.2.2.2.1
  have c26693 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_12
  simp only [← c26688, ← c26689, ← c26690] at e_411_12
  have e_411_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_13
  simp only [← c26691, ← c26692, k2, ← c26693] at e_411_13
  have hr := e_411_13
  simp only [e_411_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 842)) := by
  have c26694 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.1
  have c26695 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.1
  have c26696 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.1
  have c26697 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.1
  have c26698 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26699 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_14
  simp only [← c26694, ← c26695, ← c26696] at e_411_14
  have e_411_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_15
  simp only [← c26697, ← c26698, k2, ← c26699] at e_411_15
  have hr := e_411_15
  simp only [e_411_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 843)) := by
  have c26700 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26701 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26702 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26703 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26704 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26705 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_16
  simp only [← c26700, ← c26701, ← c26702] at e_411_16
  have e_411_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_17
  simp only [← c26703, ← c26704, k2, ← c26705] at e_411_17
  have hr := e_411_17
  simp only [e_411_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 411 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 844)) := by
  have c26706 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26707 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26708 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26709 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26710 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26711 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_411_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_18
  simp only [← c26706, ← c26707, ← c26708] at e_411_18
  have e_411_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_411_19
  simp only [← c26709, ← c26710, k2, ← c26711] at e_411_19
  have hr := e_411_19
  simp only [e_411_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 845)) := by
  have c26712 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26713 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26714 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26715 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26716 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26717 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_0
  simp only [← c26712, ← c26713, ← c26714] at e_412_0
  have e_412_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_1
  simp only [← c26715, ← c26716, k2, ← c26717] at e_412_1
  have hr := e_412_1
  simp only [e_412_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 846)) := by
  have c26718 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26719 := (publicBatchWrapper89_copies834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26720 := (publicBatchWrapper89_copies835 a h).1
  have c26721 := (publicBatchWrapper89_copies835 a h).2.1
  have c26722 := (publicBatchWrapper89_copies835 a h).2.2.1
  have c26723 := (publicBatchWrapper89_copies835 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_2
  simp only [← c26718, ← c26719, ← c26720] at e_412_2
  have e_412_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_3
  simp only [← c26721, ← c26722, k2, ← c26723] at e_412_3
  have hr := e_412_3
  simp only [e_412_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 847)) := by
  have c26724 := (publicBatchWrapper89_copies835 a h).2.2.2.2.1
  have c26725 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.1
  have c26726 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.1
  have c26727 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.1
  have c26728 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.1
  have c26729 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_4
  simp only [← c26724, ← c26725, ← c26726] at e_412_4
  have e_412_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_5
  simp only [← c26727, ← c26728, k2, ← c26729] at e_412_5
  have hr := e_412_5
  simp only [e_412_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 848)) := by
  have c26730 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26731 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26732 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26733 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26734 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26735 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_6
  simp only [← c26730, ← c26731, ← c26732] at e_412_6
  have e_412_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_7
  simp only [← c26733, ← c26734, k2, ← c26735] at e_412_7
  have hr := e_412_7
  simp only [e_412_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 849)) := by
  have c26736 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26737 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26738 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26739 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26740 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26741 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_8
  simp only [← c26736, ← c26737, ← c26738] at e_412_8
  have e_412_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_9
  simp only [← c26739, ← c26740, k2, ← c26741] at e_412_9
  have hr := e_412_9
  simp only [e_412_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 850)) := by
  have c26742 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26743 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26744 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26745 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26746 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26747 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_10
  simp only [← c26742, ← c26743, ← c26744] at e_412_10
  have e_412_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_11
  simp only [← c26745, ← c26746, k2, ← c26747] at e_412_11
  have hr := e_412_11
  simp only [e_412_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 851)) := by
  have c26748 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26749 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26750 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26751 := (publicBatchWrapper89_copies835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26752 := (publicBatchWrapper89_copies836 a h).1
  have c26753 := (publicBatchWrapper89_copies836 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_12
  simp only [← c26748, ← c26749, ← c26750] at e_412_12
  have e_412_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_13
  simp only [← c26751, ← c26752, k2, ← c26753] at e_412_13
  have hr := e_412_13
  simp only [e_412_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 852)) := by
  have c26754 := (publicBatchWrapper89_copies836 a h).2.2.1
  have c26755 := (publicBatchWrapper89_copies836 a h).2.2.2.1
  have c26756 := (publicBatchWrapper89_copies836 a h).2.2.2.2.1
  have c26757 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.1
  have c26758 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.1
  have c26759 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_14
  simp only [← c26754, ← c26755, ← c26756] at e_412_14
  have e_412_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_15
  simp only [← c26757, ← c26758, k2, ← c26759] at e_412_15
  have hr := e_412_15
  simp only [e_412_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 853)) := by
  have c26760 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.1
  have c26761 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.1
  have c26762 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26763 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26764 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26765 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_16
  simp only [← c26760, ← c26761, ← c26762] at e_412_16
  have e_412_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_17
  simp only [← c26763, ← c26764, k2, ← c26765] at e_412_17
  have hr := e_412_17
  simp only [e_412_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 412 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 854)) := by
  have c26766 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26767 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26768 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26769 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26770 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26771 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_412_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_18
  simp only [← c26766, ← c26767, ← c26768] at e_412_18
  have e_412_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_412_19
  simp only [← c26769, ← c26770, k2, ← c26771] at e_412_19
  have hr := e_412_19
  simp only [e_412_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 855)) := by
  have c26772 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26773 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26774 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26775 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26776 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26777 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_0
  simp only [← c26772, ← c26773, ← c26774] at e_413_0
  have e_413_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_1
  simp only [← c26775, ← c26776, k2, ← c26777] at e_413_1
  have hr := e_413_1
  simp only [e_413_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 856)) := by
  have c26778 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26779 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26780 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26781 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26782 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26783 := (publicBatchWrapper89_copies836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_2
  simp only [← c26778, ← c26779, ← c26780] at e_413_2
  have e_413_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_3
  simp only [← c26781, ← c26782, k2, ← c26783] at e_413_3
  have hr := e_413_3
  simp only [e_413_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 857)) := by
  have c26784 := (publicBatchWrapper89_copies837 a h).1
  have c26785 := (publicBatchWrapper89_copies837 a h).2.1
  have c26786 := (publicBatchWrapper89_copies837 a h).2.2.1
  have c26787 := (publicBatchWrapper89_copies837 a h).2.2.2.1
  have c26788 := (publicBatchWrapper89_copies837 a h).2.2.2.2.1
  have c26789 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_4
  simp only [← c26784, ← c26785, ← c26786] at e_413_4
  have e_413_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_5
  simp only [← c26787, ← c26788, k2, ← c26789] at e_413_5
  have hr := e_413_5
  simp only [e_413_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 858)) := by
  have c26790 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.1
  have c26791 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.1
  have c26792 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.1
  have c26793 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.1
  have c26794 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26795 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_6
  simp only [← c26790, ← c26791, ← c26792] at e_413_6
  have e_413_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_7
  simp only [← c26793, ← c26794, k2, ← c26795] at e_413_7
  have hr := e_413_7
  simp only [e_413_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 859)) := by
  have c26796 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26797 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26798 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26799 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26800 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26801 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_8
  simp only [← c26796, ← c26797, ← c26798] at e_413_8
  have e_413_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_9
  simp only [← c26799, ← c26800, k2, ← c26801] at e_413_9
  have hr := e_413_9
  simp only [e_413_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 860)) := by
  have c26802 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26803 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26804 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26805 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26806 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26807 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_10
  simp only [← c26802, ← c26803, ← c26804] at e_413_10
  have e_413_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_11
  simp only [← c26805, ← c26806, k2, ← c26807] at e_413_11
  have hr := e_413_11
  simp only [e_413_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 861)) := by
  have c26808 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26809 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26810 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26811 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26812 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26813 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_12
  simp only [← c26808, ← c26809, ← c26810] at e_413_12
  have e_413_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_13
  simp only [← c26811, ← c26812, k2, ← c26813] at e_413_13
  have hr := e_413_13
  simp only [e_413_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 862)) := by
  have c26814 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26815 := (publicBatchWrapper89_copies837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26816 := (publicBatchWrapper89_copies838 a h).1
  have c26817 := (publicBatchWrapper89_copies838 a h).2.1
  have c26818 := (publicBatchWrapper89_copies838 a h).2.2.1
  have c26819 := (publicBatchWrapper89_copies838 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_14
  simp only [← c26814, ← c26815, ← c26816] at e_413_14
  have e_413_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_15
  simp only [← c26817, ← c26818, k2, ← c26819] at e_413_15
  have hr := e_413_15
  simp only [e_413_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 863)) := by
  have c26820 := (publicBatchWrapper89_copies838 a h).2.2.2.2.1
  have c26821 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.1
  have c26822 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.1
  have c26823 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.1
  have c26824 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.1
  have c26825 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_16
  simp only [← c26820, ← c26821, ← c26822] at e_413_16
  have e_413_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_17
  simp only [← c26823, ← c26824, k2, ← c26825] at e_413_17
  have hr := e_413_17
  simp only [e_413_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 413 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 864)) := by
  have c26826 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26827 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26828 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26829 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26830 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26831 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_413_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_18
  simp only [← c26826, ← c26827, ← c26828] at e_413_18
  have e_413_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_413_19
  simp only [← c26829, ← c26830, k2, ← c26831] at e_413_19
  have hr := e_413_19
  simp only [e_413_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 865)) := by
  have c26832 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26833 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26834 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26835 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26836 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26837 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_0
  simp only [← c26832, ← c26833, ← c26834] at e_414_0
  have e_414_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_1
  simp only [← c26835, ← c26836, k2, ← c26837] at e_414_1
  have hr := e_414_1
  simp only [e_414_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 866)) := by
  have c26838 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26839 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26840 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26841 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26842 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26843 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_2
  simp only [← c26838, ← c26839, ← c26840] at e_414_2
  have e_414_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_3
  simp only [← c26841, ← c26842, k2, ← c26843] at e_414_3
  have hr := e_414_3
  simp only [e_414_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 867)) := by
  have c26844 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26845 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26846 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26847 := (publicBatchWrapper89_copies838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26848 := (publicBatchWrapper89_copies839 a h).1
  have c26849 := (publicBatchWrapper89_copies839 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_4
  simp only [← c26844, ← c26845, ← c26846] at e_414_4
  have e_414_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_5
  simp only [← c26847, ← c26848, k2, ← c26849] at e_414_5
  have hr := e_414_5
  simp only [e_414_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 868)) := by
  have c26850 := (publicBatchWrapper89_copies839 a h).2.2.1
  have c26851 := (publicBatchWrapper89_copies839 a h).2.2.2.1
  have c26852 := (publicBatchWrapper89_copies839 a h).2.2.2.2.1
  have c26853 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.1
  have c26854 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.1
  have c26855 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_6
  simp only [← c26850, ← c26851, ← c26852] at e_414_6
  have e_414_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_7
  simp only [← c26853, ← c26854, k2, ← c26855] at e_414_7
  have hr := e_414_7
  simp only [e_414_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 869)) := by
  have c26856 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.1
  have c26857 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.1
  have c26858 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26859 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26860 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26861 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_8
  simp only [← c26856, ← c26857, ← c26858] at e_414_8
  have e_414_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_9
  simp only [← c26859, ← c26860, k2, ← c26861] at e_414_9
  have hr := e_414_9
  simp only [e_414_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 870)) := by
  have c26862 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26863 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26864 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26865 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26866 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26867 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_10
  simp only [← c26862, ← c26863, ← c26864] at e_414_10
  have e_414_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_11
  simp only [← c26865, ← c26866, k2, ← c26867] at e_414_11
  have hr := e_414_11
  simp only [e_414_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 871)) := by
  have c26868 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26869 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26870 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26871 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26872 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26873 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_12
  simp only [← c26868, ← c26869, ← c26870] at e_414_12
  have e_414_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_13
  simp only [← c26871, ← c26872, k2, ← c26873] at e_414_13
  have hr := e_414_13
  simp only [e_414_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 872)) := by
  have c26874 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26875 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26876 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26877 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26878 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26879 := (publicBatchWrapper89_copies839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_14
  simp only [← c26874, ← c26875, ← c26876] at e_414_14
  have e_414_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_15
  simp only [← c26877, ← c26878, k2, ← c26879] at e_414_15
  have hr := e_414_15
  simp only [e_414_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 873)) := by
  have c26880 := (publicBatchWrapper89_copies840 a h).1
  have c26881 := (publicBatchWrapper89_copies840 a h).2.1
  have c26882 := (publicBatchWrapper89_copies840 a h).2.2.1
  have c26883 := (publicBatchWrapper89_copies840 a h).2.2.2.1
  have c26884 := (publicBatchWrapper89_copies840 a h).2.2.2.2.1
  have c26885 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_16
  simp only [← c26880, ← c26881, ← c26882] at e_414_16
  have e_414_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_17
  simp only [← c26883, ← c26884, k2, ← c26885] at e_414_17
  have hr := e_414_17
  simp only [e_414_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 414 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 874)) := by
  have c26886 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.1
  have c26887 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.1
  have c26888 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.1
  have c26889 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.1
  have c26890 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26891 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_414_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_18
  simp only [← c26886, ← c26887, ← c26888] at e_414_18
  have e_414_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_414_19
  simp only [← c26889, ← c26890, k2, ← c26891] at e_414_19
  have hr := e_414_19
  simp only [e_414_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 875)) := by
  have c26892 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26893 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26894 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26895 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26896 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26897 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_0
  simp only [← c26892, ← c26893, ← c26894] at e_415_0
  have e_415_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_1
  simp only [← c26895, ← c26896, k2, ← c26897] at e_415_1
  have hr := e_415_1
  simp only [e_415_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 876)) := by
  have c26898 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26899 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26900 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26901 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26902 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26903 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_2
  simp only [← c26898, ← c26899, ← c26900] at e_415_2
  have e_415_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_3
  simp only [← c26901, ← c26902, k2, ← c26903] at e_415_3
  have hr := e_415_3
  simp only [e_415_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 877)) := by
  have c26904 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26905 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26906 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26907 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26908 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26909 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_4
  simp only [← c26904, ← c26905, ← c26906] at e_415_4
  have e_415_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_5
  simp only [← c26907, ← c26908, k2, ← c26909] at e_415_5
  have hr := e_415_5
  simp only [e_415_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 878)) := by
  have c26910 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26911 := (publicBatchWrapper89_copies840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26912 := (publicBatchWrapper89_copies841 a h).1
  have c26913 := (publicBatchWrapper89_copies841 a h).2.1
  have c26914 := (publicBatchWrapper89_copies841 a h).2.2.1
  have c26915 := (publicBatchWrapper89_copies841 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_6
  simp only [← c26910, ← c26911, ← c26912] at e_415_6
  have e_415_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_7
  simp only [← c26913, ← c26914, k2, ← c26915] at e_415_7
  have hr := e_415_7
  simp only [e_415_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 879)) := by
  have c26916 := (publicBatchWrapper89_copies841 a h).2.2.2.2.1
  have c26917 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.1
  have c26918 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.1
  have c26919 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.1
  have c26920 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.1
  have c26921 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_8
  simp only [← c26916, ← c26917, ← c26918] at e_415_8
  have e_415_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_9
  simp only [← c26919, ← c26920, k2, ← c26921] at e_415_9
  have hr := e_415_9
  simp only [e_415_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 880)) := by
  have c26922 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26923 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26924 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26925 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26926 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26927 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_10
  simp only [← c26922, ← c26923, ← c26924] at e_415_10
  have e_415_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_11
  simp only [← c26925, ← c26926, k2, ← c26927] at e_415_11
  have hr := e_415_11
  simp only [e_415_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 881)) := by
  have c26928 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26929 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26930 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26931 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26932 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26933 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_12
  simp only [← c26928, ← c26929, ← c26930] at e_415_12
  have e_415_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_13
  simp only [← c26931, ← c26932, k2, ← c26933] at e_415_13
  have hr := e_415_13
  simp only [e_415_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 882)) := by
  have c26934 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26935 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26936 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26937 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26938 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26939 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_14
  simp only [← c26934, ← c26935, ← c26936] at e_415_14
  have e_415_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_15
  simp only [← c26937, ← c26938, k2, ← c26939] at e_415_15
  have hr := e_415_15
  simp only [e_415_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 883)) := by
  have c26940 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26941 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26942 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26943 := (publicBatchWrapper89_copies841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26944 := (publicBatchWrapper89_copies842 a h).1
  have c26945 := (publicBatchWrapper89_copies842 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_16
  simp only [← c26940, ← c26941, ← c26942] at e_415_16
  have e_415_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_17
  simp only [← c26943, ← c26944, k2, ← c26945] at e_415_17
  have hr := e_415_17
  simp only [e_415_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 415 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 884)) := by
  have c26946 := (publicBatchWrapper89_copies842 a h).2.2.1
  have c26947 := (publicBatchWrapper89_copies842 a h).2.2.2.1
  have c26948 := (publicBatchWrapper89_copies842 a h).2.2.2.2.1
  have c26949 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.1
  have c26950 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.1
  have c26951 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_415_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_18
  simp only [← c26946, ← c26947, ← c26948] at e_415_18
  have e_415_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_415_19
  simp only [← c26949, ← c26950, k2, ← c26951] at e_415_19
  have hr := e_415_19
  simp only [e_415_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 885)) := by
  have c26952 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.1
  have c26953 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.1
  have c26954 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26955 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26956 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26957 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_0
  simp only [← c26952, ← c26953, ← c26954] at e_416_0
  have e_416_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_1
  simp only [← c26955, ← c26956, k2, ← c26957] at e_416_1
  have hr := e_416_1
  simp only [e_416_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 886)) := by
  have c26958 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26959 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26960 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26961 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26962 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26963 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_2
  simp only [← c26958, ← c26959, ← c26960] at e_416_2
  have e_416_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_3
  simp only [← c26961, ← c26962, k2, ← c26963] at e_416_3
  have hr := e_416_3
  simp only [e_416_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 887)) := by
  have c26964 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26965 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26966 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26967 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26968 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26969 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_4
  simp only [← c26964, ← c26965, ← c26966] at e_416_4
  have e_416_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_5
  simp only [← c26967, ← c26968, k2, ← c26969] at e_416_5
  have hr := e_416_5
  simp only [e_416_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 888)) := by
  have c26970 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26971 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26972 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26973 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26974 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26975 := (publicBatchWrapper89_copies842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_6
  simp only [← c26970, ← c26971, ← c26972] at e_416_6
  have e_416_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_7
  simp only [← c26973, ← c26974, k2, ← c26975] at e_416_7
  have hr := e_416_7
  simp only [e_416_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 889)) := by
  have c26976 := (publicBatchWrapper89_copies843 a h).1
  have c26977 := (publicBatchWrapper89_copies843 a h).2.1
  have c26978 := (publicBatchWrapper89_copies843 a h).2.2.1
  have c26979 := (publicBatchWrapper89_copies843 a h).2.2.2.1
  have c26980 := (publicBatchWrapper89_copies843 a h).2.2.2.2.1
  have c26981 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_8
  simp only [← c26976, ← c26977, ← c26978] at e_416_8
  have e_416_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_9
  simp only [← c26979, ← c26980, k2, ← c26981] at e_416_9
  have hr := e_416_9
  simp only [e_416_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 890)) := by
  have c26982 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.1
  have c26983 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.1
  have c26984 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.1
  have c26985 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.1
  have c26986 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26987 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_10
  simp only [← c26982, ← c26983, ← c26984] at e_416_10
  have e_416_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_11
  simp only [← c26985, ← c26986, k2, ← c26987] at e_416_11
  have hr := e_416_11
  simp only [e_416_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 891)) := by
  have c26988 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26989 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26990 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26991 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26992 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26993 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_12
  simp only [← c26988, ← c26989, ← c26990] at e_416_12
  have e_416_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_13
  simp only [← c26991, ← c26992, k2, ← c26993] at e_416_13
  have hr := e_416_13
  simp only [e_416_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 892)) := by
  have c26994 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26995 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26996 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26997 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26998 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26999 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_14
  simp only [← c26994, ← c26995, ← c26996] at e_416_14
  have e_416_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_15
  simp only [← c26997, ← c26998, k2, ← c26999] at e_416_15
  have hr := e_416_15
  simp only [e_416_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 893)) := by
  have c27000 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27001 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27002 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27003 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27004 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27005 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_16
  simp only [← c27000, ← c27001, ← c27002] at e_416_16
  have e_416_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_17
  simp only [← c27003, ← c27004, k2, ← c27005] at e_416_17
  have hr := e_416_17
  simp only [e_416_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 416 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 894)) := by
  have c27006 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27007 := (publicBatchWrapper89_copies843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27008 := (publicBatchWrapper89_copies844 a h).1
  have c27009 := (publicBatchWrapper89_copies844 a h).2.1
  have c27010 := (publicBatchWrapper89_copies844 a h).2.2.1
  have c27011 := (publicBatchWrapper89_copies844 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_416_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_18
  simp only [← c27006, ← c27007, ← c27008] at e_416_18
  have e_416_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_416_19
  simp only [← c27009, ← c27010, k2, ← c27011] at e_416_19
  have hr := e_416_19
  simp only [e_416_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 895)) := by
  have c27012 := (publicBatchWrapper89_copies844 a h).2.2.2.2.1
  have c27013 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.1
  have c27014 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.1
  have c27015 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.1
  have c27016 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.1
  have c27017 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_0
  simp only [← c27012, ← c27013, ← c27014] at e_417_0
  have e_417_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_1
  simp only [← c27015, ← c27016, k2, ← c27017] at e_417_1
  have hr := e_417_1
  simp only [e_417_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 896)) := by
  have c27018 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27019 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27020 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27021 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27022 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27023 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_2
  simp only [← c27018, ← c27019, ← c27020] at e_417_2
  have e_417_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_3
  simp only [← c27021, ← c27022, k2, ← c27023] at e_417_3
  have hr := e_417_3
  simp only [e_417_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 897)) := by
  have c27024 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27025 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27026 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27027 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27028 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27029 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_4
  simp only [← c27024, ← c27025, ← c27026] at e_417_4
  have e_417_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_5
  simp only [← c27027, ← c27028, k2, ← c27029] at e_417_5
  have hr := e_417_5
  simp only [e_417_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 898)) := by
  have c27030 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27031 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27032 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27033 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27034 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27035 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_6
  simp only [← c27030, ← c27031, ← c27032] at e_417_6
  have e_417_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_7
  simp only [← c27033, ← c27034, k2, ← c27035] at e_417_7
  have hr := e_417_7
  simp only [e_417_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 899)) := by
  have c27036 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27037 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27038 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27039 := (publicBatchWrapper89_copies844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27040 := (publicBatchWrapper89_copies845 a h).1
  have c27041 := (publicBatchWrapper89_copies845 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_8
  simp only [← c27036, ← c27037, ← c27038] at e_417_8
  have e_417_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_9
  simp only [← c27039, ← c27040, k2, ← c27041] at e_417_9
  have hr := e_417_9
  simp only [e_417_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 900)) := by
  have c27042 := (publicBatchWrapper89_copies845 a h).2.2.1
  have c27043 := (publicBatchWrapper89_copies845 a h).2.2.2.1
  have c27044 := (publicBatchWrapper89_copies845 a h).2.2.2.2.1
  have c27045 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.1
  have c27046 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.1
  have c27047 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_10
  simp only [← c27042, ← c27043, ← c27044] at e_417_10
  have e_417_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_11
  simp only [← c27045, ← c27046, k2, ← c27047] at e_417_11
  have hr := e_417_11
  simp only [e_417_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 901)) := by
  have c27048 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.1
  have c27049 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.1
  have c27050 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27051 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27052 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27053 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_12
  simp only [← c27048, ← c27049, ← c27050] at e_417_12
  have e_417_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_13
  simp only [← c27051, ← c27052, k2, ← c27053] at e_417_13
  have hr := e_417_13
  simp only [e_417_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 902)) := by
  have c27054 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27055 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27056 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27057 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27058 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27059 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_14
  simp only [← c27054, ← c27055, ← c27056] at e_417_14
  have e_417_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_15
  simp only [← c27057, ← c27058, k2, ← c27059] at e_417_15
  have hr := e_417_15
  simp only [e_417_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 903)) := by
  have c27060 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27061 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27062 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27063 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27064 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27065 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_16
  simp only [← c27060, ← c27061, ← c27062] at e_417_16
  have e_417_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_17
  simp only [← c27063, ← c27064, k2, ← c27065] at e_417_17
  have hr := e_417_17
  simp only [e_417_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 417 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 904)) := by
  have c27066 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27067 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27068 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27069 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27070 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27071 := (publicBatchWrapper89_copies845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_417_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_18
  simp only [← c27066, ← c27067, ← c27068] at e_417_18
  have e_417_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_417_19
  simp only [← c27069, ← c27070, k2, ← c27071] at e_417_19
  have hr := e_417_19
  simp only [e_417_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 905)) := by
  have c27072 := (publicBatchWrapper89_copies846 a h).1
  have c27073 := (publicBatchWrapper89_copies846 a h).2.1
  have c27074 := (publicBatchWrapper89_copies846 a h).2.2.1
  have c27075 := (publicBatchWrapper89_copies846 a h).2.2.2.1
  have c27076 := (publicBatchWrapper89_copies846 a h).2.2.2.2.1
  have c27077 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_0
  simp only [← c27072, ← c27073, ← c27074] at e_418_0
  have e_418_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_1
  simp only [← c27075, ← c27076, k2, ← c27077] at e_418_1
  have hr := e_418_1
  simp only [e_418_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 906)) := by
  have c27078 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.1
  have c27079 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.1
  have c27080 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.1
  have c27081 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.1
  have c27082 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27083 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_2
  simp only [← c27078, ← c27079, ← c27080] at e_418_2
  have e_418_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_3
  simp only [← c27081, ← c27082, k2, ← c27083] at e_418_3
  have hr := e_418_3
  simp only [e_418_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 907)) := by
  have c27084 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27085 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27086 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27087 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27088 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27089 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_4
  simp only [← c27084, ← c27085, ← c27086] at e_418_4
  have e_418_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_5
  simp only [← c27087, ← c27088, k2, ← c27089] at e_418_5
  have hr := e_418_5
  simp only [e_418_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 908)) := by
  have c27090 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27091 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27092 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27093 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27094 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27095 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_6
  simp only [← c27090, ← c27091, ← c27092] at e_418_6
  have e_418_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_7
  simp only [← c27093, ← c27094, k2, ← c27095] at e_418_7
  have hr := e_418_7
  simp only [e_418_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 909)) := by
  have c27096 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27097 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27098 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27099 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27100 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27101 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_8
  simp only [← c27096, ← c27097, ← c27098] at e_418_8
  have e_418_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_9
  simp only [← c27099, ← c27100, k2, ← c27101] at e_418_9
  have hr := e_418_9
  simp only [e_418_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 910)) := by
  have c27102 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27103 := (publicBatchWrapper89_copies846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27104 := (publicBatchWrapper89_copies847 a h).1
  have c27105 := (publicBatchWrapper89_copies847 a h).2.1
  have c27106 := (publicBatchWrapper89_copies847 a h).2.2.1
  have c27107 := (publicBatchWrapper89_copies847 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_10
  simp only [← c27102, ← c27103, ← c27104] at e_418_10
  have e_418_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_11
  simp only [← c27105, ← c27106, k2, ← c27107] at e_418_11
  have hr := e_418_11
  simp only [e_418_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 911)) := by
  have c27108 := (publicBatchWrapper89_copies847 a h).2.2.2.2.1
  have c27109 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.1
  have c27110 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.1
  have c27111 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.1
  have c27112 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.1
  have c27113 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_12
  simp only [← c27108, ← c27109, ← c27110] at e_418_12
  have e_418_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_13
  simp only [← c27111, ← c27112, k2, ← c27113] at e_418_13
  have hr := e_418_13
  simp only [e_418_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 912)) := by
  have c27114 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27115 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27116 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27117 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27118 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27119 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_14
  simp only [← c27114, ← c27115, ← c27116] at e_418_14
  have e_418_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_15
  simp only [← c27117, ← c27118, k2, ← c27119] at e_418_15
  have hr := e_418_15
  simp only [e_418_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 913)) := by
  have c27120 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27121 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27122 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27123 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27124 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27125 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_16
  simp only [← c27120, ← c27121, ← c27122] at e_418_16
  have e_418_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_17
  simp only [← c27123, ← c27124, k2, ← c27125] at e_418_17
  have hr := e_418_17
  simp only [e_418_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 418 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 914)) := by
  have c27126 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27127 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27128 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27129 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27130 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27131 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_418_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_18
  simp only [← c27126, ← c27127, ← c27128] at e_418_18
  have e_418_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_418_19
  simp only [← c27129, ← c27130, k2, ← c27131] at e_418_19
  have hr := e_418_19
  simp only [e_418_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 915)) := by
  have c27132 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27133 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27134 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27135 := (publicBatchWrapper89_copies847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27136 := (publicBatchWrapper89_copies848 a h).1
  have c27137 := (publicBatchWrapper89_copies848 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_0
  simp only [← c27132, ← c27133, ← c27134] at e_419_0
  have e_419_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_1
  simp only [← c27135, ← c27136, k2, ← c27137] at e_419_1
  have hr := e_419_1
  simp only [e_419_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 916)) := by
  have c27138 := (publicBatchWrapper89_copies848 a h).2.2.1
  have c27139 := (publicBatchWrapper89_copies848 a h).2.2.2.1
  have c27140 := (publicBatchWrapper89_copies848 a h).2.2.2.2.1
  have c27141 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.1
  have c27142 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.1
  have c27143 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_2
  simp only [← c27138, ← c27139, ← c27140] at e_419_2
  have e_419_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_3
  simp only [← c27141, ← c27142, k2, ← c27143] at e_419_3
  have hr := e_419_3
  simp only [e_419_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 917)) := by
  have c27144 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.1
  have c27145 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.1
  have c27146 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27147 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27148 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27149 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_4
  simp only [← c27144, ← c27145, ← c27146] at e_419_4
  have e_419_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_5
  simp only [← c27147, ← c27148, k2, ← c27149] at e_419_5
  have hr := e_419_5
  simp only [e_419_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 918)) := by
  have c27150 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27151 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27152 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27153 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27154 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27155 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_6
  simp only [← c27150, ← c27151, ← c27152] at e_419_6
  have e_419_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_7
  simp only [← c27153, ← c27154, k2, ← c27155] at e_419_7
  have hr := e_419_7
  simp only [e_419_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 919)) := by
  have c27156 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27157 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27158 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27159 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27160 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27161 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_8
  simp only [← c27156, ← c27157, ← c27158] at e_419_8
  have e_419_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_9
  simp only [← c27159, ← c27160, k2, ← c27161] at e_419_9
  have hr := e_419_9
  simp only [e_419_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 920)) := by
  have c27162 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27163 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27164 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27165 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27166 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27167 := (publicBatchWrapper89_copies848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_10
  simp only [← c27162, ← c27163, ← c27164] at e_419_10
  have e_419_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_11
  simp only [← c27165, ← c27166, k2, ← c27167] at e_419_11
  have hr := e_419_11
  simp only [e_419_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 921)) := by
  have c27168 := (publicBatchWrapper89_copies849 a h).1
  have c27169 := (publicBatchWrapper89_copies849 a h).2.1
  have c27170 := (publicBatchWrapper89_copies849 a h).2.2.1
  have c27171 := (publicBatchWrapper89_copies849 a h).2.2.2.1
  have c27172 := (publicBatchWrapper89_copies849 a h).2.2.2.2.1
  have c27173 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_12
  simp only [← c27168, ← c27169, ← c27170] at e_419_12
  have e_419_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_13
  simp only [← c27171, ← c27172, k2, ← c27173] at e_419_13
  have hr := e_419_13
  simp only [e_419_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 922)) := by
  have c27174 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.1
  have c27175 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.1
  have c27176 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.1
  have c27177 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.1
  have c27178 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27179 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_14
  simp only [← c27174, ← c27175, ← c27176] at e_419_14
  have e_419_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_15
  simp only [← c27177, ← c27178, k2, ← c27179] at e_419_15
  have hr := e_419_15
  simp only [e_419_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 923)) := by
  have c27180 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27181 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27182 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27183 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27184 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27185 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_16
  simp only [← c27180, ← c27181, ← c27182] at e_419_16
  have e_419_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_17
  simp only [← c27183, ← c27184, k2, ← c27185] at e_419_17
  have hr := e_419_17
  simp only [e_419_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 419 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 924)) := by
  have c27186 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27187 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27188 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27189 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27190 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27191 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_419_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_18
  simp only [← c27186, ← c27187, ← c27188] at e_419_18
  have e_419_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_419_19
  simp only [← c27189, ← c27190, k2, ← c27191] at e_419_19
  have hr := e_419_19
  simp only [e_419_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 925)) := by
  have c27192 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27193 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27194 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27195 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27196 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27197 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_0
  simp only [← c27192, ← c27193, ← c27194] at e_420_0
  have e_420_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_1
  simp only [← c27195, ← c27196, k2, ← c27197] at e_420_1
  have hr := e_420_1
  simp only [e_420_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 926)) := by
  have c27198 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27199 := (publicBatchWrapper89_copies849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27200 := (publicBatchWrapper89_copies850 a h).1
  have c27201 := (publicBatchWrapper89_copies850 a h).2.1
  have c27202 := (publicBatchWrapper89_copies850 a h).2.2.1
  have c27203 := (publicBatchWrapper89_copies850 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_2
  simp only [← c27198, ← c27199, ← c27200] at e_420_2
  have e_420_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_3
  simp only [← c27201, ← c27202, k2, ← c27203] at e_420_3
  have hr := e_420_3
  simp only [e_420_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 927)) := by
  have c27204 := (publicBatchWrapper89_copies850 a h).2.2.2.2.1
  have c27205 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.1
  have c27206 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.1
  have c27207 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.1
  have c27208 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.1
  have c27209 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_4
  simp only [← c27204, ← c27205, ← c27206] at e_420_4
  have e_420_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_5
  simp only [← c27207, ← c27208, k2, ← c27209] at e_420_5
  have hr := e_420_5
  simp only [e_420_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 928)) := by
  have c27210 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27211 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27212 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27213 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27214 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27215 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_6
  simp only [← c27210, ← c27211, ← c27212] at e_420_6
  have e_420_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_7
  simp only [← c27213, ← c27214, k2, ← c27215] at e_420_7
  have hr := e_420_7
  simp only [e_420_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 929)) := by
  have c27216 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27217 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27218 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27219 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27220 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27221 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_8
  simp only [← c27216, ← c27217, ← c27218] at e_420_8
  have e_420_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_9
  simp only [← c27219, ← c27220, k2, ← c27221] at e_420_9
  have hr := e_420_9
  simp only [e_420_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 930)) := by
  have c27222 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27223 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27224 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27225 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27226 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27227 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_10
  simp only [← c27222, ← c27223, ← c27224] at e_420_10
  have e_420_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_11
  simp only [← c27225, ← c27226, k2, ← c27227] at e_420_11
  have hr := e_420_11
  simp only [e_420_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 931)) := by
  have c27228 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27229 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27230 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27231 := (publicBatchWrapper89_copies850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27232 := (publicBatchWrapper89_copies851 a h).1
  have c27233 := (publicBatchWrapper89_copies851 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_12
  simp only [← c27228, ← c27229, ← c27230] at e_420_12
  have e_420_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_13
  simp only [← c27231, ← c27232, k2, ← c27233] at e_420_13
  have hr := e_420_13
  simp only [e_420_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 932)) := by
  have c27234 := (publicBatchWrapper89_copies851 a h).2.2.1
  have c27235 := (publicBatchWrapper89_copies851 a h).2.2.2.1
  have c27236 := (publicBatchWrapper89_copies851 a h).2.2.2.2.1
  have c27237 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.1
  have c27238 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.1
  have c27239 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_14
  simp only [← c27234, ← c27235, ← c27236] at e_420_14
  have e_420_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_15
  simp only [← c27237, ← c27238, k2, ← c27239] at e_420_15
  have hr := e_420_15
  simp only [e_420_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 933)) := by
  have c27240 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.1
  have c27241 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.1
  have c27242 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27243 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27244 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27245 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_16
  simp only [← c27240, ← c27241, ← c27242] at e_420_16
  have e_420_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_17
  simp only [← c27243, ← c27244, k2, ← c27245] at e_420_17
  have hr := e_420_17
  simp only [e_420_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 420 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 934)) := by
  have c27246 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27247 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27248 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27249 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27250 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27251 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_420_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_18
  simp only [← c27246, ← c27247, ← c27248] at e_420_18
  have e_420_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_420_19
  simp only [← c27249, ← c27250, k2, ← c27251] at e_420_19
  have hr := e_420_19
  simp only [e_420_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 935)) := by
  have c27252 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27253 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27254 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27255 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27256 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27257 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_0
  simp only [← c27252, ← c27253, ← c27254] at e_421_0
  have e_421_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_1
  simp only [← c27255, ← c27256, k2, ← c27257] at e_421_1
  have hr := e_421_1
  simp only [e_421_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 936)) := by
  have c27258 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27259 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27260 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27261 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27262 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27263 := (publicBatchWrapper89_copies851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_2
  simp only [← c27258, ← c27259, ← c27260] at e_421_2
  have e_421_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_3
  simp only [← c27261, ← c27262, k2, ← c27263] at e_421_3
  have hr := e_421_3
  simp only [e_421_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 937)) := by
  have c27264 := (publicBatchWrapper89_copies852 a h).1
  have c27265 := (publicBatchWrapper89_copies852 a h).2.1
  have c27266 := (publicBatchWrapper89_copies852 a h).2.2.1
  have c27267 := (publicBatchWrapper89_copies852 a h).2.2.2.1
  have c27268 := (publicBatchWrapper89_copies852 a h).2.2.2.2.1
  have c27269 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_4
  simp only [← c27264, ← c27265, ← c27266] at e_421_4
  have e_421_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_5
  simp only [← c27267, ← c27268, k2, ← c27269] at e_421_5
  have hr := e_421_5
  simp only [e_421_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 938)) := by
  have c27270 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.1
  have c27271 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.1
  have c27272 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.1
  have c27273 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.1
  have c27274 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27275 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_6
  simp only [← c27270, ← c27271, ← c27272] at e_421_6
  have e_421_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_7
  simp only [← c27273, ← c27274, k2, ← c27275] at e_421_7
  have hr := e_421_7
  simp only [e_421_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 939)) := by
  have c27276 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27277 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27278 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27279 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27280 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27281 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_8
  simp only [← c27276, ← c27277, ← c27278] at e_421_8
  have e_421_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_9
  simp only [← c27279, ← c27280, k2, ← c27281] at e_421_9
  have hr := e_421_9
  simp only [e_421_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 940)) := by
  have c27282 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27283 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27284 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27285 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27286 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27287 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_10
  simp only [← c27282, ← c27283, ← c27284] at e_421_10
  have e_421_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_11
  simp only [← c27285, ← c27286, k2, ← c27287] at e_421_11
  have hr := e_421_11
  simp only [e_421_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 941)) := by
  have c27288 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27289 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27290 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27291 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27292 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27293 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_12
  simp only [← c27288, ← c27289, ← c27290] at e_421_12
  have e_421_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_13
  simp only [← c27291, ← c27292, k2, ← c27293] at e_421_13
  have hr := e_421_13
  simp only [e_421_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 942)) := by
  have c27294 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27295 := (publicBatchWrapper89_copies852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27296 := (publicBatchWrapper89_copies853 a h).1
  have c27297 := (publicBatchWrapper89_copies853 a h).2.1
  have c27298 := (publicBatchWrapper89_copies853 a h).2.2.1
  have c27299 := (publicBatchWrapper89_copies853 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_14
  simp only [← c27294, ← c27295, ← c27296] at e_421_14
  have e_421_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_15
  simp only [← c27297, ← c27298, k2, ← c27299] at e_421_15
  have hr := e_421_15
  simp only [e_421_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 943)) := by
  have c27300 := (publicBatchWrapper89_copies853 a h).2.2.2.2.1
  have c27301 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.1
  have c27302 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.1
  have c27303 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.1
  have c27304 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.1
  have c27305 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_16
  simp only [← c27300, ← c27301, ← c27302] at e_421_16
  have e_421_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_17
  simp only [← c27303, ← c27304, k2, ← c27305] at e_421_17
  have hr := e_421_17
  simp only [e_421_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 421 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 944)) := by
  have c27306 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27307 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27308 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27309 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27310 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27311 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_421_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_18
  simp only [← c27306, ← c27307, ← c27308] at e_421_18
  have e_421_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_421_19
  simp only [← c27309, ← c27310, k2, ← c27311] at e_421_19
  have hr := e_421_19
  simp only [e_421_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 945)) := by
  have c27312 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27313 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27314 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27315 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27316 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27317 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_0
  simp only [← c27312, ← c27313, ← c27314] at e_422_0
  have e_422_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_1
  simp only [← c27315, ← c27316, k2, ← c27317] at e_422_1
  have hr := e_422_1
  simp only [e_422_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 946)) := by
  have c27318 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27319 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27320 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27321 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27322 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27323 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_2
  simp only [← c27318, ← c27319, ← c27320] at e_422_2
  have e_422_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_3
  simp only [← c27321, ← c27322, k2, ← c27323] at e_422_3
  have hr := e_422_3
  simp only [e_422_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 947)) := by
  have c27324 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27325 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27326 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27327 := (publicBatchWrapper89_copies853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27328 := (publicBatchWrapper89_copies854 a h).1
  have c27329 := (publicBatchWrapper89_copies854 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_4
  simp only [← c27324, ← c27325, ← c27326] at e_422_4
  have e_422_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_5
  simp only [← c27327, ← c27328, k2, ← c27329] at e_422_5
  have hr := e_422_5
  simp only [e_422_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 948)) := by
  have c27330 := (publicBatchWrapper89_copies854 a h).2.2.1
  have c27331 := (publicBatchWrapper89_copies854 a h).2.2.2.1
  have c27332 := (publicBatchWrapper89_copies854 a h).2.2.2.2.1
  have c27333 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.1
  have c27334 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.1
  have c27335 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_6
  simp only [← c27330, ← c27331, ← c27332] at e_422_6
  have e_422_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_7
  simp only [← c27333, ← c27334, k2, ← c27335] at e_422_7
  have hr := e_422_7
  simp only [e_422_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 949)) := by
  have c27336 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.1
  have c27337 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.1
  have c27338 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27339 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27340 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27341 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_8
  simp only [← c27336, ← c27337, ← c27338] at e_422_8
  have e_422_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_9
  simp only [← c27339, ← c27340, k2, ← c27341] at e_422_9
  have hr := e_422_9
  simp only [e_422_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 950)) := by
  have c27342 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27343 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27344 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27345 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27346 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27347 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_10
  simp only [← c27342, ← c27343, ← c27344] at e_422_10
  have e_422_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_11
  simp only [← c27345, ← c27346, k2, ← c27347] at e_422_11
  have hr := e_422_11
  simp only [e_422_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 951)) := by
  have c27348 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27349 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27350 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27351 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27352 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27353 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_12
  simp only [← c27348, ← c27349, ← c27350] at e_422_12
  have e_422_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_13
  simp only [← c27351, ← c27352, k2, ← c27353] at e_422_13
  have hr := e_422_13
  simp only [e_422_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 952)) := by
  have c27354 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27355 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27356 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27357 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27358 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27359 := (publicBatchWrapper89_copies854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_14
  simp only [← c27354, ← c27355, ← c27356] at e_422_14
  have e_422_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_15
  simp only [← c27357, ← c27358, k2, ← c27359] at e_422_15
  have hr := e_422_15
  simp only [e_422_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 953)) := by
  have c27360 := (publicBatchWrapper89_copies855 a h).1
  have c27361 := (publicBatchWrapper89_copies855 a h).2.1
  have c27362 := (publicBatchWrapper89_copies855 a h).2.2.1
  have c27363 := (publicBatchWrapper89_copies855 a h).2.2.2.1
  have c27364 := (publicBatchWrapper89_copies855 a h).2.2.2.2.1
  have c27365 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_16
  simp only [← c27360, ← c27361, ← c27362] at e_422_16
  have e_422_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_17
  simp only [← c27363, ← c27364, k2, ← c27365] at e_422_17
  have hr := e_422_17
  simp only [e_422_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 422 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 954)) := by
  have c27366 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.1
  have c27367 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.1
  have c27368 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.1
  have c27369 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.1
  have c27370 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27371 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_422_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_18
  simp only [← c27366, ← c27367, ← c27368] at e_422_18
  have e_422_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_422_19
  simp only [← c27369, ← c27370, k2, ← c27371] at e_422_19
  have hr := e_422_19
  simp only [e_422_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 955)) := by
  have c27372 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27373 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27374 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27375 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27376 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27377 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_0
  simp only [← c27372, ← c27373, ← c27374] at e_423_0
  have e_423_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_1
  simp only [← c27375, ← c27376, k2, ← c27377] at e_423_1
  have hr := e_423_1
  simp only [e_423_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 956)) := by
  have c27378 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27379 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27380 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27381 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27382 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27383 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_2
  simp only [← c27378, ← c27379, ← c27380] at e_423_2
  have e_423_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_3
  simp only [← c27381, ← c27382, k2, ← c27383] at e_423_3
  have hr := e_423_3
  simp only [e_423_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 957)) := by
  have c27384 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27385 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27386 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27387 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27388 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27389 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_4
  simp only [← c27384, ← c27385, ← c27386] at e_423_4
  have e_423_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_5
  simp only [← c27387, ← c27388, k2, ← c27389] at e_423_5
  have hr := e_423_5
  simp only [e_423_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 958)) := by
  have c27390 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27391 := (publicBatchWrapper89_copies855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27392 := (publicBatchWrapper89_copies856 a h).1
  have c27393 := (publicBatchWrapper89_copies856 a h).2.1
  have c27394 := (publicBatchWrapper89_copies856 a h).2.2.1
  have c27395 := (publicBatchWrapper89_copies856 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_6
  simp only [← c27390, ← c27391, ← c27392] at e_423_6
  have e_423_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_7
  simp only [← c27393, ← c27394, k2, ← c27395] at e_423_7
  have hr := e_423_7
  simp only [e_423_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 959)) := by
  have c27396 := (publicBatchWrapper89_copies856 a h).2.2.2.2.1
  have c27397 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.1
  have c27398 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.1
  have c27399 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.1
  have c27400 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.1
  have c27401 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_8
  simp only [← c27396, ← c27397, ← c27398] at e_423_8
  have e_423_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_9
  simp only [← c27399, ← c27400, k2, ← c27401] at e_423_9
  have hr := e_423_9
  simp only [e_423_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 960)) := by
  have c27402 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27403 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27404 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27405 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27406 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27407 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_10
  simp only [← c27402, ← c27403, ← c27404] at e_423_10
  have e_423_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_11
  simp only [← c27405, ← c27406, k2, ← c27407] at e_423_11
  have hr := e_423_11
  simp only [e_423_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 961)) := by
  have c27408 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27409 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27410 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27411 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27412 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27413 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_12
  simp only [← c27408, ← c27409, ← c27410] at e_423_12
  have e_423_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_13
  simp only [← c27411, ← c27412, k2, ← c27413] at e_423_13
  have hr := e_423_13
  simp only [e_423_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 962)) := by
  have c27414 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27415 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27416 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27417 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27418 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27419 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_14
  simp only [← c27414, ← c27415, ← c27416] at e_423_14
  have e_423_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_15
  simp only [← c27417, ← c27418, k2, ← c27419] at e_423_15
  have hr := e_423_15
  simp only [e_423_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 963)) := by
  have c27420 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27421 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27422 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27423 := (publicBatchWrapper89_copies856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27424 := (publicBatchWrapper89_copies857 a h).1
  have c27425 := (publicBatchWrapper89_copies857 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_16
  simp only [← c27420, ← c27421, ← c27422] at e_423_16
  have e_423_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_17
  simp only [← c27423, ← c27424, k2, ← c27425] at e_423_17
  have hr := e_423_17
  simp only [e_423_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 423 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 964)) := by
  have c27426 := (publicBatchWrapper89_copies857 a h).2.2.1
  have c27427 := (publicBatchWrapper89_copies857 a h).2.2.2.1
  have c27428 := (publicBatchWrapper89_copies857 a h).2.2.2.2.1
  have c27429 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.1
  have c27430 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.1
  have c27431 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_423_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_18
  simp only [← c27426, ← c27427, ← c27428] at e_423_18
  have e_423_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_423_19
  simp only [← c27429, ← c27430, k2, ← c27431] at e_423_19
  have hr := e_423_19
  simp only [e_423_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 965)) := by
  have c27432 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.1
  have c27433 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.1
  have c27434 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27435 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27436 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27437 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_0
  simp only [← c27432, ← c27433, ← c27434] at e_424_0
  have e_424_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_1
  simp only [← c27435, ← c27436, k2, ← c27437] at e_424_1
  have hr := e_424_1
  simp only [e_424_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 966)) := by
  have c27438 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27439 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27440 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27441 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27442 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27443 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_2
  simp only [← c27438, ← c27439, ← c27440] at e_424_2
  have e_424_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_3
  simp only [← c27441, ← c27442, k2, ← c27443] at e_424_3
  have hr := e_424_3
  simp only [e_424_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 967)) := by
  have c27444 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27445 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27446 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27447 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27448 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27449 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_4
  simp only [← c27444, ← c27445, ← c27446] at e_424_4
  have e_424_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_5
  simp only [← c27447, ← c27448, k2, ← c27449] at e_424_5
  have hr := e_424_5
  simp only [e_424_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 968)) := by
  have c27450 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27451 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27452 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27453 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27454 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27455 := (publicBatchWrapper89_copies857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_6
  simp only [← c27450, ← c27451, ← c27452] at e_424_6
  have e_424_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_7
  simp only [← c27453, ← c27454, k2, ← c27455] at e_424_7
  have hr := e_424_7
  simp only [e_424_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 969)) := by
  have c27456 := (publicBatchWrapper89_copies858 a h).1
  have c27457 := (publicBatchWrapper89_copies858 a h).2.1
  have c27458 := (publicBatchWrapper89_copies858 a h).2.2.1
  have c27459 := (publicBatchWrapper89_copies858 a h).2.2.2.1
  have c27460 := (publicBatchWrapper89_copies858 a h).2.2.2.2.1
  have c27461 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_8
  simp only [← c27456, ← c27457, ← c27458] at e_424_8
  have e_424_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_9
  simp only [← c27459, ← c27460, k2, ← c27461] at e_424_9
  have hr := e_424_9
  simp only [e_424_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1010)) := by
  have c27462 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.1
  have c27463 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.1
  have c27464 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.1
  have c27465 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.1
  have c27466 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27467 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_10
  simp only [← c27462, ← c27463, ← c27464] at e_424_10
  have e_424_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_11
  simp only [← c27465, ← c27466, k2, ← c27467] at e_424_11
  have hr := e_424_11
  simp only [e_424_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1011)) := by
  have c27468 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27469 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27470 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27471 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27472 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27473 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_12
  simp only [← c27468, ← c27469, ← c27470] at e_424_12
  have e_424_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_13
  simp only [← c27471, ← c27472, k2, ← c27473] at e_424_13
  have hr := e_424_13
  simp only [e_424_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1012)) := by
  have c27474 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27475 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27476 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27477 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27478 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27479 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_14
  simp only [← c27474, ← c27475, ← c27476] at e_424_14
  have e_424_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_15
  simp only [← c27477, ← c27478, k2, ← c27479] at e_424_15
  have hr := e_424_15
  simp only [e_424_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1013)) := by
  have c27480 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27481 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27482 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27483 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27484 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27485 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_16
  simp only [← c27480, ← c27481, ← c27482] at e_424_16
  have e_424_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_17
  simp only [← c27483, ← c27484, k2, ← c27485] at e_424_17
  have hr := e_424_17
  simp only [e_424_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 424 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1014)) := by
  have c27486 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27487 := (publicBatchWrapper89_copies858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27488 := (publicBatchWrapper89_copies859 a h).1
  have c27489 := (publicBatchWrapper89_copies859 a h).2.1
  have c27490 := (publicBatchWrapper89_copies859 a h).2.2.1
  have c27491 := (publicBatchWrapper89_copies859 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_424_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_18
  simp only [← c27486, ← c27487, ← c27488] at e_424_18
  have e_424_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_424_19
  simp only [← c27489, ← c27490, k2, ← c27491] at e_424_19
  have hr := e_424_19
  simp only [e_424_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1015)) := by
  have c27492 := (publicBatchWrapper89_copies859 a h).2.2.2.2.1
  have c27493 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.1
  have c27494 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.1
  have c27495 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.1
  have c27496 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.1
  have c27497 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_0
  simp only [← c27492, ← c27493, ← c27494] at e_425_0
  have e_425_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_1
  simp only [← c27495, ← c27496, k2, ← c27497] at e_425_1
  have hr := e_425_1
  simp only [e_425_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1016)) := by
  have c27498 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27499 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27500 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27501 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27502 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27503 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_2
  simp only [← c27498, ← c27499, ← c27500] at e_425_2
  have e_425_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_3
  simp only [← c27501, ← c27502, k2, ← c27503] at e_425_3
  have hr := e_425_3
  simp only [e_425_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1017)) := by
  have c27504 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27505 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27506 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27507 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27508 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27509 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_4
  simp only [← c27504, ← c27505, ← c27506] at e_425_4
  have e_425_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_5
  simp only [← c27507, ← c27508, k2, ← c27509] at e_425_5
  have hr := e_425_5
  simp only [e_425_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1018)) := by
  have c27510 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27511 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27512 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27513 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27514 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27515 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_6
  simp only [← c27510, ← c27511, ← c27512] at e_425_6
  have e_425_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_7
  simp only [← c27513, ← c27514, k2, ← c27515] at e_425_7
  have hr := e_425_7
  simp only [e_425_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1019)) := by
  have c27516 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27517 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27518 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27519 := (publicBatchWrapper89_copies859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27520 := (publicBatchWrapper89_copies860 a h).1
  have c27521 := (publicBatchWrapper89_copies860 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_8
  simp only [← c27516, ← c27517, ← c27518] at e_425_8
  have e_425_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_9
  simp only [← c27519, ← c27520, k2, ← c27521] at e_425_9
  have hr := e_425_9
  simp only [e_425_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1020)) := by
  have c27522 := (publicBatchWrapper89_copies860 a h).2.2.1
  have c27523 := (publicBatchWrapper89_copies860 a h).2.2.2.1
  have c27524 := (publicBatchWrapper89_copies860 a h).2.2.2.2.1
  have c27525 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.1
  have c27526 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.1
  have c27527 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_10
  simp only [← c27522, ← c27523, ← c27524] at e_425_10
  have e_425_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_11
  simp only [← c27525, ← c27526, k2, ← c27527] at e_425_11
  have hr := e_425_11
  simp only [e_425_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1021)) := by
  have c27528 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.1
  have c27529 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.1
  have c27530 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27531 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27532 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27533 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_12
  simp only [← c27528, ← c27529, ← c27530] at e_425_12
  have e_425_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_13
  simp only [← c27531, ← c27532, k2, ← c27533] at e_425_13
  have hr := e_425_13
  simp only [e_425_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1022)) := by
  have c27534 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27535 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27536 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27537 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27538 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27539 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_14
  simp only [← c27534, ← c27535, ← c27536] at e_425_14
  have e_425_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_15
  simp only [← c27537, ← c27538, k2, ← c27539] at e_425_15
  have hr := e_425_15
  simp only [e_425_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1023)) := by
  have c27540 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27541 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27542 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27543 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27544 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27545 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_16
  simp only [← c27540, ← c27541, ← c27542] at e_425_16
  have e_425_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_17
  simp only [← c27543, ← c27544, k2, ← c27545] at e_425_17
  have hr := e_425_17
  simp only [e_425_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 425 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1024)) := by
  have c27546 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27547 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27548 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27549 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27550 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27551 := (publicBatchWrapper89_copies860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_425_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_18
  simp only [← c27546, ← c27547, ← c27548] at e_425_18
  have e_425_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_425_19
  simp only [← c27549, ← c27550, k2, ← c27551] at e_425_19
  have hr := e_425_19
  simp only [e_425_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1025)) := by
  have c27552 := (publicBatchWrapper89_copies861 a h).1
  have c27553 := (publicBatchWrapper89_copies861 a h).2.1
  have c27554 := (publicBatchWrapper89_copies861 a h).2.2.1
  have c27555 := (publicBatchWrapper89_copies861 a h).2.2.2.1
  have c27556 := (publicBatchWrapper89_copies861 a h).2.2.2.2.1
  have c27557 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_0
  simp only [← c27552, ← c27553, ← c27554] at e_426_0
  have e_426_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_1
  simp only [← c27555, ← c27556, k2, ← c27557] at e_426_1
  have hr := e_426_1
  simp only [e_426_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1026)) := by
  have c27558 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.1
  have c27559 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.1
  have c27560 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.1
  have c27561 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.1
  have c27562 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27563 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_2
  simp only [← c27558, ← c27559, ← c27560] at e_426_2
  have e_426_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_3
  simp only [← c27561, ← c27562, k2, ← c27563] at e_426_3
  have hr := e_426_3
  simp only [e_426_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1027)) := by
  have c27564 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27565 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27566 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27567 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27568 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27569 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_4
  simp only [← c27564, ← c27565, ← c27566] at e_426_4
  have e_426_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_5
  simp only [← c27567, ← c27568, k2, ← c27569] at e_426_5
  have hr := e_426_5
  simp only [e_426_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1028)) := by
  have c27570 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27571 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27572 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27573 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27574 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27575 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_6
  simp only [← c27570, ← c27571, ← c27572] at e_426_6
  have e_426_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_7
  simp only [← c27573, ← c27574, k2, ← c27575] at e_426_7
  have hr := e_426_7
  simp only [e_426_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1029)) := by
  have c27576 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27577 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27578 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27579 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27580 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27581 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_8
  simp only [← c27576, ← c27577, ← c27578] at e_426_8
  have e_426_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_9
  simp only [← c27579, ← c27580, k2, ← c27581] at e_426_9
  have hr := e_426_9
  simp only [e_426_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1030)) := by
  have c27582 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27583 := (publicBatchWrapper89_copies861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27584 := (publicBatchWrapper89_copies862 a h).1
  have c27585 := (publicBatchWrapper89_copies862 a h).2.1
  have c27586 := (publicBatchWrapper89_copies862 a h).2.2.1
  have c27587 := (publicBatchWrapper89_copies862 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_10
  simp only [← c27582, ← c27583, ← c27584] at e_426_10
  have e_426_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_11
  simp only [← c27585, ← c27586, k2, ← c27587] at e_426_11
  have hr := e_426_11
  simp only [e_426_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1031)) := by
  have c27588 := (publicBatchWrapper89_copies862 a h).2.2.2.2.1
  have c27589 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.1
  have c27590 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.1
  have c27591 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.1
  have c27592 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.1
  have c27593 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_12
  simp only [← c27588, ← c27589, ← c27590] at e_426_12
  have e_426_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_13
  simp only [← c27591, ← c27592, k2, ← c27593] at e_426_13
  have hr := e_426_13
  simp only [e_426_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1032)) := by
  have c27594 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27595 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27596 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27597 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27598 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27599 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_14
  simp only [← c27594, ← c27595, ← c27596] at e_426_14
  have e_426_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_15
  simp only [← c27597, ← c27598, k2, ← c27599] at e_426_15
  have hr := e_426_15
  simp only [e_426_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1033)) := by
  have c27600 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27601 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27602 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27603 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27604 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27605 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_16
  simp only [← c27600, ← c27601, ← c27602] at e_426_16
  have e_426_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_17
  simp only [← c27603, ← c27604, k2, ← c27605] at e_426_17
  have hr := e_426_17
  simp only [e_426_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 426 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1034)) := by
  have c27606 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27607 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27608 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27609 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27610 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27611 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_426_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_18
  simp only [← c27606, ← c27607, ← c27608] at e_426_18
  have e_426_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_426_19
  simp only [← c27609, ← c27610, k2, ← c27611] at e_426_19
  have hr := e_426_19
  simp only [e_426_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1035)) := by
  have c27612 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27613 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27614 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27615 := (publicBatchWrapper89_copies862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27616 := (publicBatchWrapper89_copies863 a h).1
  have c27617 := (publicBatchWrapper89_copies863 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_0
  simp only [← c27612, ← c27613, ← c27614] at e_427_0
  have e_427_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_1
  simp only [← c27615, ← c27616, k2, ← c27617] at e_427_1
  have hr := e_427_1
  simp only [e_427_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1036)) := by
  have c27618 := (publicBatchWrapper89_copies863 a h).2.2.1
  have c27619 := (publicBatchWrapper89_copies863 a h).2.2.2.1
  have c27620 := (publicBatchWrapper89_copies863 a h).2.2.2.2.1
  have c27621 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.1
  have c27622 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.1
  have c27623 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_2
  simp only [← c27618, ← c27619, ← c27620] at e_427_2
  have e_427_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_3
  simp only [← c27621, ← c27622, k2, ← c27623] at e_427_3
  have hr := e_427_3
  simp only [e_427_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1037)) := by
  have c27624 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.1
  have c27625 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.1
  have c27626 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27627 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27628 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27629 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_4
  simp only [← c27624, ← c27625, ← c27626] at e_427_4
  have e_427_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_5
  simp only [← c27627, ← c27628, k2, ← c27629] at e_427_5
  have hr := e_427_5
  simp only [e_427_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1038)) := by
  have c27630 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27631 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27632 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27633 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27634 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27635 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_6
  simp only [← c27630, ← c27631, ← c27632] at e_427_6
  have e_427_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_7
  simp only [← c27633, ← c27634, k2, ← c27635] at e_427_7
  have hr := e_427_7
  simp only [e_427_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1039)) := by
  have c27636 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27637 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27638 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27639 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27640 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27641 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_8
  simp only [← c27636, ← c27637, ← c27638] at e_427_8
  have e_427_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_9
  simp only [← c27639, ← c27640, k2, ← c27641] at e_427_9
  have hr := e_427_9
  simp only [e_427_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1040)) := by
  have c27642 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27643 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27644 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27645 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27646 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27647 := (publicBatchWrapper89_copies863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_10
  simp only [← c27642, ← c27643, ← c27644] at e_427_10
  have e_427_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_11
  simp only [← c27645, ← c27646, k2, ← c27647] at e_427_11
  have hr := e_427_11
  simp only [e_427_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1041)) := by
  have c27648 := (publicBatchWrapper89_copies864 a h).1
  have c27649 := (publicBatchWrapper89_copies864 a h).2.1
  have c27650 := (publicBatchWrapper89_copies864 a h).2.2.1
  have c27651 := (publicBatchWrapper89_copies864 a h).2.2.2.1
  have c27652 := (publicBatchWrapper89_copies864 a h).2.2.2.2.1
  have c27653 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_12
  simp only [← c27648, ← c27649, ← c27650] at e_427_12
  have e_427_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_13
  simp only [← c27651, ← c27652, k2, ← c27653] at e_427_13
  have hr := e_427_13
  simp only [e_427_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1042)) := by
  have c27654 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.1
  have c27655 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.1
  have c27656 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.1
  have c27657 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.1
  have c27658 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27659 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_14
  simp only [← c27654, ← c27655, ← c27656] at e_427_14
  have e_427_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_15
  simp only [← c27657, ← c27658, k2, ← c27659] at e_427_15
  have hr := e_427_15
  simp only [e_427_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1043)) := by
  have c27660 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27661 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27662 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27663 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27664 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27665 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_16
  simp only [← c27660, ← c27661, ← c27662] at e_427_16
  have e_427_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_17
  simp only [← c27663, ← c27664, k2, ← c27665] at e_427_17
  have hr := e_427_17
  simp only [e_427_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 427 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1044)) := by
  have c27666 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27667 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27668 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27669 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27670 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27671 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_427_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_18
  simp only [← c27666, ← c27667, ← c27668] at e_427_18
  have e_427_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_427_19
  simp only [← c27669, ← c27670, k2, ← c27671] at e_427_19
  have hr := e_427_19
  simp only [e_427_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1045)) := by
  have c27672 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27673 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27674 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27675 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27676 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27677 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_0
  simp only [← c27672, ← c27673, ← c27674] at e_428_0
  have e_428_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_1
  simp only [← c27675, ← c27676, k2, ← c27677] at e_428_1
  have hr := e_428_1
  simp only [e_428_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1046)) := by
  have c27678 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27679 := (publicBatchWrapper89_copies864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27680 := (publicBatchWrapper89_copies865 a h).1
  have c27681 := (publicBatchWrapper89_copies865 a h).2.1
  have c27682 := (publicBatchWrapper89_copies865 a h).2.2.1
  have c27683 := (publicBatchWrapper89_copies865 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_2
  simp only [← c27678, ← c27679, ← c27680] at e_428_2
  have e_428_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_3
  simp only [← c27681, ← c27682, k2, ← c27683] at e_428_3
  have hr := e_428_3
  simp only [e_428_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1047)) := by
  have c27684 := (publicBatchWrapper89_copies865 a h).2.2.2.2.1
  have c27685 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.1
  have c27686 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.1
  have c27687 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.1
  have c27688 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.1
  have c27689 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_4
  simp only [← c27684, ← c27685, ← c27686] at e_428_4
  have e_428_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_5
  simp only [← c27687, ← c27688, k2, ← c27689] at e_428_5
  have hr := e_428_5
  simp only [e_428_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1048)) := by
  have c27690 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27691 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27692 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27693 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27694 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27695 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_6
  simp only [← c27690, ← c27691, ← c27692] at e_428_6
  have e_428_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_7
  simp only [← c27693, ← c27694, k2, ← c27695] at e_428_7
  have hr := e_428_7
  simp only [e_428_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1049)) := by
  have c27696 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27697 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27698 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27699 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27700 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27701 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_8
  simp only [← c27696, ← c27697, ← c27698] at e_428_8
  have e_428_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_9
  simp only [← c27699, ← c27700, k2, ← c27701] at e_428_9
  have hr := e_428_9
  simp only [e_428_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1050)) := by
  have c27702 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27703 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27704 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27705 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27706 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27707 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_10
  simp only [← c27702, ← c27703, ← c27704] at e_428_10
  have e_428_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_11
  simp only [← c27705, ← c27706, k2, ← c27707] at e_428_11
  have hr := e_428_11
  simp only [e_428_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1051)) := by
  have c27708 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27709 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27710 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27711 := (publicBatchWrapper89_copies865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27712 := (publicBatchWrapper89_copies866 a h).1
  have c27713 := (publicBatchWrapper89_copies866 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_12
  simp only [← c27708, ← c27709, ← c27710] at e_428_12
  have e_428_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_13
  simp only [← c27711, ← c27712, k2, ← c27713] at e_428_13
  have hr := e_428_13
  simp only [e_428_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1052)) := by
  have c27714 := (publicBatchWrapper89_copies866 a h).2.2.1
  have c27715 := (publicBatchWrapper89_copies866 a h).2.2.2.1
  have c27716 := (publicBatchWrapper89_copies866 a h).2.2.2.2.1
  have c27717 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.1
  have c27718 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.1
  have c27719 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_14
  simp only [← c27714, ← c27715, ← c27716] at e_428_14
  have e_428_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_15
  simp only [← c27717, ← c27718, k2, ← c27719] at e_428_15
  have hr := e_428_15
  simp only [e_428_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1053)) := by
  have c27720 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.1
  have c27721 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.1
  have c27722 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27723 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27724 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27725 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_16
  simp only [← c27720, ← c27721, ← c27722] at e_428_16
  have e_428_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_17
  simp only [← c27723, ← c27724, k2, ← c27725] at e_428_17
  have hr := e_428_17
  simp only [e_428_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 428 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1054)) := by
  have c27726 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27727 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27728 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27729 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27730 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27731 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_428_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_18
  simp only [← c27726, ← c27727, ← c27728] at e_428_18
  have e_428_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_428_19
  simp only [← c27729, ← c27730, k2, ← c27731] at e_428_19
  have hr := e_428_19
  simp only [e_428_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1055)) := by
  have c27732 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27733 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27734 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27735 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27736 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27737 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_0
  simp only [← c27732, ← c27733, ← c27734] at e_429_0
  have e_429_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_1
  simp only [← c27735, ← c27736, k2, ← c27737] at e_429_1
  have hr := e_429_1
  simp only [e_429_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1056)) := by
  have c27738 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27739 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27740 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27741 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27742 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27743 := (publicBatchWrapper89_copies866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_2
  simp only [← c27738, ← c27739, ← c27740] at e_429_2
  have e_429_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_3
  simp only [← c27741, ← c27742, k2, ← c27743] at e_429_3
  have hr := e_429_3
  simp only [e_429_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1057)) := by
  have c27744 := (publicBatchWrapper89_copies867 a h).1
  have c27745 := (publicBatchWrapper89_copies867 a h).2.1
  have c27746 := (publicBatchWrapper89_copies867 a h).2.2.1
  have c27747 := (publicBatchWrapper89_copies867 a h).2.2.2.1
  have c27748 := (publicBatchWrapper89_copies867 a h).2.2.2.2.1
  have c27749 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_4
  simp only [← c27744, ← c27745, ← c27746] at e_429_4
  have e_429_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_5
  simp only [← c27747, ← c27748, k2, ← c27749] at e_429_5
  have hr := e_429_5
  simp only [e_429_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1058)) := by
  have c27750 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.1
  have c27751 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.1
  have c27752 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.1
  have c27753 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.1
  have c27754 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27755 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_6
  simp only [← c27750, ← c27751, ← c27752] at e_429_6
  have e_429_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_7
  simp only [← c27753, ← c27754, k2, ← c27755] at e_429_7
  have hr := e_429_7
  simp only [e_429_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1059)) := by
  have c27756 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27757 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27758 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27759 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27760 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27761 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_8
  simp only [← c27756, ← c27757, ← c27758] at e_429_8
  have e_429_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_9
  simp only [← c27759, ← c27760, k2, ← c27761] at e_429_9
  have hr := e_429_9
  simp only [e_429_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1060)) := by
  have c27762 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27763 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27764 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27765 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27766 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27767 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_10
  simp only [← c27762, ← c27763, ← c27764] at e_429_10
  have e_429_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_11
  simp only [← c27765, ← c27766, k2, ← c27767] at e_429_11
  have hr := e_429_11
  simp only [e_429_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1061)) := by
  have c27768 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27769 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27770 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27771 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27772 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27773 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_12
  simp only [← c27768, ← c27769, ← c27770] at e_429_12
  have e_429_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_13
  simp only [← c27771, ← c27772, k2, ← c27773] at e_429_13
  have hr := e_429_13
  simp only [e_429_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1062)) := by
  have c27774 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27775 := (publicBatchWrapper89_copies867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27776 := (publicBatchWrapper89_copies868 a h).1
  have c27777 := (publicBatchWrapper89_copies868 a h).2.1
  have c27778 := (publicBatchWrapper89_copies868 a h).2.2.1
  have c27779 := (publicBatchWrapper89_copies868 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_14
  simp only [← c27774, ← c27775, ← c27776] at e_429_14
  have e_429_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_15
  simp only [← c27777, ← c27778, k2, ← c27779] at e_429_15
  have hr := e_429_15
  simp only [e_429_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1063)) := by
  have c27780 := (publicBatchWrapper89_copies868 a h).2.2.2.2.1
  have c27781 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.1
  have c27782 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.1
  have c27783 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.1
  have c27784 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.1
  have c27785 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_16
  simp only [← c27780, ← c27781, ← c27782] at e_429_16
  have e_429_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_17
  simp only [← c27783, ← c27784, k2, ← c27785] at e_429_17
  have hr := e_429_17
  simp only [e_429_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 429 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1064)) := by
  have c27786 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27787 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27788 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27789 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27790 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27791 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_429_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_18
  simp only [← c27786, ← c27787, ← c27788] at e_429_18
  have e_429_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_429_19
  simp only [← c27789, ← c27790, k2, ← c27791] at e_429_19
  have hr := e_429_19
  simp only [e_429_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1065)) := by
  have c27792 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27793 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27794 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27795 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27796 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27797 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_0
  simp only [← c27792, ← c27793, ← c27794] at e_430_0
  have e_430_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_1
  simp only [← c27795, ← c27796, k2, ← c27797] at e_430_1
  have hr := e_430_1
  simp only [e_430_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1066)) := by
  have c27798 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27799 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27800 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27801 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27802 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27803 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_2
  simp only [← c27798, ← c27799, ← c27800] at e_430_2
  have e_430_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_3
  simp only [← c27801, ← c27802, k2, ← c27803] at e_430_3
  have hr := e_430_3
  simp only [e_430_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1067)) := by
  have c27804 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27805 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27806 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27807 := (publicBatchWrapper89_copies868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27808 := (publicBatchWrapper89_copies869 a h).1
  have c27809 := (publicBatchWrapper89_copies869 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_4
  simp only [← c27804, ← c27805, ← c27806] at e_430_4
  have e_430_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_5
  simp only [← c27807, ← c27808, k2, ← c27809] at e_430_5
  have hr := e_430_5
  simp only [e_430_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1068)) := by
  have c27810 := (publicBatchWrapper89_copies869 a h).2.2.1
  have c27811 := (publicBatchWrapper89_copies869 a h).2.2.2.1
  have c27812 := (publicBatchWrapper89_copies869 a h).2.2.2.2.1
  have c27813 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.1
  have c27814 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.1
  have c27815 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_6
  simp only [← c27810, ← c27811, ← c27812] at e_430_6
  have e_430_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_7
  simp only [← c27813, ← c27814, k2, ← c27815] at e_430_7
  have hr := e_430_7
  simp only [e_430_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1069)) := by
  have c27816 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.1
  have c27817 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.1
  have c27818 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27819 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27820 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27821 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_8
  simp only [← c27816, ← c27817, ← c27818] at e_430_8
  have e_430_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_9
  simp only [← c27819, ← c27820, k2, ← c27821] at e_430_9
  have hr := e_430_9
  simp only [e_430_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1070)) := by
  have c27822 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27823 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27824 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27825 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27826 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27827 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_10
  simp only [← c27822, ← c27823, ← c27824] at e_430_10
  have e_430_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_11
  simp only [← c27825, ← c27826, k2, ← c27827] at e_430_11
  have hr := e_430_11
  simp only [e_430_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1071)) := by
  have c27828 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27829 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27830 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27831 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27832 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27833 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_12
  simp only [← c27828, ← c27829, ← c27830] at e_430_12
  have e_430_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_13
  simp only [← c27831, ← c27832, k2, ← c27833] at e_430_13
  have hr := e_430_13
  simp only [e_430_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1072)) := by
  have c27834 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27835 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27836 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27837 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27838 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27839 := (publicBatchWrapper89_copies869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_14
  simp only [← c27834, ← c27835, ← c27836] at e_430_14
  have e_430_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_15
  simp only [← c27837, ← c27838, k2, ← c27839] at e_430_15
  have hr := e_430_15
  simp only [e_430_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1073)) := by
  have c27840 := (publicBatchWrapper89_copies870 a h).1
  have c27841 := (publicBatchWrapper89_copies870 a h).2.1
  have c27842 := (publicBatchWrapper89_copies870 a h).2.2.1
  have c27843 := (publicBatchWrapper89_copies870 a h).2.2.2.1
  have c27844 := (publicBatchWrapper89_copies870 a h).2.2.2.2.1
  have c27845 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_16
  simp only [← c27840, ← c27841, ← c27842] at e_430_16
  have e_430_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_17
  simp only [← c27843, ← c27844, k2, ← c27845] at e_430_17
  have hr := e_430_17
  simp only [e_430_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 430 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1074)) := by
  have c27846 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.1
  have c27847 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.1
  have c27848 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.1
  have c27849 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.1
  have c27850 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27851 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_430_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_18
  simp only [← c27846, ← c27847, ← c27848] at e_430_18
  have e_430_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_430_19
  simp only [← c27849, ← c27850, k2, ← c27851] at e_430_19
  have hr := e_430_19
  simp only [e_430_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1075)) := by
  have c27852 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27853 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27854 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27855 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27856 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27857 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_0
  simp only [← c27852, ← c27853, ← c27854] at e_431_0
  have e_431_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_1
  simp only [← c27855, ← c27856, k2, ← c27857] at e_431_1
  have hr := e_431_1
  simp only [e_431_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1076)) := by
  have c27858 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27859 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27860 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27861 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27862 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27863 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_2
  simp only [← c27858, ← c27859, ← c27860] at e_431_2
  have e_431_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_3
  simp only [← c27861, ← c27862, k2, ← c27863] at e_431_3
  have hr := e_431_3
  simp only [e_431_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1077)) := by
  have c27864 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27865 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27866 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27867 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27868 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27869 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_4
  simp only [← c27864, ← c27865, ← c27866] at e_431_4
  have e_431_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_5
  simp only [← c27867, ← c27868, k2, ← c27869] at e_431_5
  have hr := e_431_5
  simp only [e_431_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1078)) := by
  have c27870 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27871 := (publicBatchWrapper89_copies870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27872 := (publicBatchWrapper89_copies871 a h).1
  have c27873 := (publicBatchWrapper89_copies871 a h).2.1
  have c27874 := (publicBatchWrapper89_copies871 a h).2.2.1
  have c27875 := (publicBatchWrapper89_copies871 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_6
  simp only [← c27870, ← c27871, ← c27872] at e_431_6
  have e_431_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_7
  simp only [← c27873, ← c27874, k2, ← c27875] at e_431_7
  have hr := e_431_7
  simp only [e_431_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1079)) := by
  have c27876 := (publicBatchWrapper89_copies871 a h).2.2.2.2.1
  have c27877 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.1
  have c27878 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.1
  have c27879 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.1
  have c27880 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.1
  have c27881 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_8
  simp only [← c27876, ← c27877, ← c27878] at e_431_8
  have e_431_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_9
  simp only [← c27879, ← c27880, k2, ← c27881] at e_431_9
  have hr := e_431_9
  simp only [e_431_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1080)) := by
  have c27882 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27883 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27884 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27885 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27886 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27887 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_10
  simp only [← c27882, ← c27883, ← c27884] at e_431_10
  have e_431_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_11
  simp only [← c27885, ← c27886, k2, ← c27887] at e_431_11
  have hr := e_431_11
  simp only [e_431_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1081)) := by
  have c27888 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27889 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27890 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27891 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27892 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27893 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_12
  simp only [← c27888, ← c27889, ← c27890] at e_431_12
  have e_431_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_13
  simp only [← c27891, ← c27892, k2, ← c27893] at e_431_13
  have hr := e_431_13
  simp only [e_431_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1082)) := by
  have c27894 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27895 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27896 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27897 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27898 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27899 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_14
  simp only [← c27894, ← c27895, ← c27896] at e_431_14
  have e_431_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_15
  simp only [← c27897, ← c27898, k2, ← c27899] at e_431_15
  have hr := e_431_15
  simp only [e_431_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1083)) := by
  have c27900 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27901 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27902 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27903 := (publicBatchWrapper89_copies871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27904 := (publicBatchWrapper89_copies872 a h).1
  have c27905 := (publicBatchWrapper89_copies872 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_16
  simp only [← c27900, ← c27901, ← c27902] at e_431_16
  have e_431_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_17
  simp only [← c27903, ← c27904, k2, ← c27905] at e_431_17
  have hr := e_431_17
  simp only [e_431_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 431 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1084)) := by
  have c27906 := (publicBatchWrapper89_copies872 a h).2.2.1
  have c27907 := (publicBatchWrapper89_copies872 a h).2.2.2.1
  have c27908 := (publicBatchWrapper89_copies872 a h).2.2.2.2.1
  have c27909 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.1
  have c27910 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.1
  have c27911 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_431_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_18
  simp only [← c27906, ← c27907, ← c27908] at e_431_18
  have e_431_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_431_19
  simp only [← c27909, ← c27910, k2, ← c27911] at e_431_19
  have hr := e_431_19
  simp only [e_431_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1085)) := by
  have c27912 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.1
  have c27913 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.1
  have c27914 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27915 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27916 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27917 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_0
  simp only [← c27912, ← c27913, ← c27914] at e_432_0
  have e_432_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_1
  simp only [← c27915, ← c27916, k2, ← c27917] at e_432_1
  have hr := e_432_1
  simp only [e_432_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1086)) := by
  have c27918 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27919 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27920 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27921 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27922 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27923 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_2
  simp only [← c27918, ← c27919, ← c27920] at e_432_2
  have e_432_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_3
  simp only [← c27921, ← c27922, k2, ← c27923] at e_432_3
  have hr := e_432_3
  simp only [e_432_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1087)) := by
  have c27924 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27925 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27926 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27927 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27928 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27929 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_4
  simp only [← c27924, ← c27925, ← c27926] at e_432_4
  have e_432_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_5
  simp only [← c27927, ← c27928, k2, ← c27929] at e_432_5
  have hr := e_432_5
  simp only [e_432_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1088)) := by
  have c27930 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27931 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27932 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27933 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27934 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27935 := (publicBatchWrapper89_copies872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_6
  simp only [← c27930, ← c27931, ← c27932] at e_432_6
  have e_432_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_7
  simp only [← c27933, ← c27934, k2, ← c27935] at e_432_7
  have hr := e_432_7
  simp only [e_432_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1089)) := by
  have c27936 := (publicBatchWrapper89_copies873 a h).1
  have c27937 := (publicBatchWrapper89_copies873 a h).2.1
  have c27938 := (publicBatchWrapper89_copies873 a h).2.2.1
  have c27939 := (publicBatchWrapper89_copies873 a h).2.2.2.1
  have c27940 := (publicBatchWrapper89_copies873 a h).2.2.2.2.1
  have c27941 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_8
  simp only [← c27936, ← c27937, ← c27938] at e_432_8
  have e_432_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_9
  simp only [← c27939, ← c27940, k2, ← c27941] at e_432_9
  have hr := e_432_9
  simp only [e_432_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1090)) := by
  have c27942 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.1
  have c27943 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.1
  have c27944 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.1
  have c27945 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.1
  have c27946 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27947 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_10
  simp only [← c27942, ← c27943, ← c27944] at e_432_10
  have e_432_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_11
  simp only [← c27945, ← c27946, k2, ← c27947] at e_432_11
  have hr := e_432_11
  simp only [e_432_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1091)) := by
  have c27948 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27949 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27950 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27951 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27952 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27953 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_12
  simp only [← c27948, ← c27949, ← c27950] at e_432_12
  have e_432_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_13
  simp only [← c27951, ← c27952, k2, ← c27953] at e_432_13
  have hr := e_432_13
  simp only [e_432_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1092)) := by
  have c27954 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27955 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27956 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27957 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27958 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27959 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_14
  simp only [← c27954, ← c27955, ← c27956] at e_432_14
  have e_432_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_15
  simp only [← c27957, ← c27958, k2, ← c27959] at e_432_15
  have hr := e_432_15
  simp only [e_432_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1093)) := by
  have c27960 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27961 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27962 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27963 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27964 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27965 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_16
  simp only [← c27960, ← c27961, ← c27962] at e_432_16
  have e_432_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_17
  simp only [← c27963, ← c27964, k2, ← c27965] at e_432_17
  have hr := e_432_17
  simp only [e_432_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 432 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1094)) := by
  have c27966 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27967 := (publicBatchWrapper89_copies873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c27968 := (publicBatchWrapper89_copies874 a h).1
  have c27969 := (publicBatchWrapper89_copies874 a h).2.1
  have c27970 := (publicBatchWrapper89_copies874 a h).2.2.1
  have c27971 := (publicBatchWrapper89_copies874 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_432_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_18
  simp only [← c27966, ← c27967, ← c27968] at e_432_18
  have e_432_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_432_19
  simp only [← c27969, ← c27970, k2, ← c27971] at e_432_19
  have hr := e_432_19
  simp only [e_432_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1095)) := by
  have c27972 := (publicBatchWrapper89_copies874 a h).2.2.2.2.1
  have c27973 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.1
  have c27974 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.1
  have c27975 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.1
  have c27976 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.1
  have c27977 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_0
  simp only [← c27972, ← c27973, ← c27974] at e_433_0
  have e_433_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_1
  simp only [← c27975, ← c27976, k2, ← c27977] at e_433_1
  have hr := e_433_1
  simp only [e_433_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1096)) := by
  have c27978 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c27979 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c27980 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27981 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27982 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27983 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_2
  simp only [← c27978, ← c27979, ← c27980] at e_433_2
  have e_433_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_3
  simp only [← c27981, ← c27982, k2, ← c27983] at e_433_3
  have hr := e_433_3
  simp only [e_433_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1097)) := by
  have c27984 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27985 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27986 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27987 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27988 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27989 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_4
  simp only [← c27984, ← c27985, ← c27986] at e_433_4
  have e_433_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_5
  simp only [← c27987, ← c27988, k2, ← c27989] at e_433_5
  have hr := e_433_5
  simp only [e_433_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1098)) := by
  have c27990 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27991 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27992 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27993 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27994 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27995 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_6
  simp only [← c27990, ← c27991, ← c27992] at e_433_6
  have e_433_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_7
  simp only [← c27993, ← c27994, k2, ← c27995] at e_433_7
  have hr := e_433_7
  simp only [e_433_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1099)) := by
  have c27996 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27997 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27998 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c27999 := (publicBatchWrapper89_copies874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28000 := (publicBatchWrapper89_copies875 a h).1
  have c28001 := (publicBatchWrapper89_copies875 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_8
  simp only [← c27996, ← c27997, ← c27998] at e_433_8
  have e_433_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_9
  simp only [← c27999, ← c28000, k2, ← c28001] at e_433_9
  have hr := e_433_9
  simp only [e_433_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1100)) := by
  have c28002 := (publicBatchWrapper89_copies875 a h).2.2.1
  have c28003 := (publicBatchWrapper89_copies875 a h).2.2.2.1
  have c28004 := (publicBatchWrapper89_copies875 a h).2.2.2.2.1
  have c28005 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.1
  have c28006 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.1
  have c28007 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_10
  simp only [← c28002, ← c28003, ← c28004] at e_433_10
  have e_433_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_11
  simp only [← c28005, ← c28006, k2, ← c28007] at e_433_11
  have hr := e_433_11
  simp only [e_433_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1101)) := by
  have c28008 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.1
  have c28009 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.1
  have c28010 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28011 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28012 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28013 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_12
  simp only [← c28008, ← c28009, ← c28010] at e_433_12
  have e_433_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_13
  simp only [← c28011, ← c28012, k2, ← c28013] at e_433_13
  have hr := e_433_13
  simp only [e_433_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1102)) := by
  have c28014 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28015 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28016 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28017 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28018 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28019 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_14
  simp only [← c28014, ← c28015, ← c28016] at e_433_14
  have e_433_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_15
  simp only [← c28017, ← c28018, k2, ← c28019] at e_433_15
  have hr := e_433_15
  simp only [e_433_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1103)) := by
  have c28020 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28021 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28022 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28023 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28024 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28025 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_16
  simp only [← c28020, ← c28021, ← c28022] at e_433_16
  have e_433_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_17
  simp only [← c28023, ← c28024, k2, ← c28025] at e_433_17
  have hr := e_433_17
  simp only [e_433_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 433 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1104)) := by
  have c28026 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28027 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28028 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28029 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28030 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28031 := (publicBatchWrapper89_copies875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_433_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_18
  simp only [← c28026, ← c28027, ← c28028] at e_433_18
  have e_433_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_433_19
  simp only [← c28029, ← c28030, k2, ← c28031] at e_433_19
  have hr := e_433_19
  simp only [e_433_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1105)) := by
  have c28032 := (publicBatchWrapper89_copies876 a h).1
  have c28033 := (publicBatchWrapper89_copies876 a h).2.1
  have c28034 := (publicBatchWrapper89_copies876 a h).2.2.1
  have c28035 := (publicBatchWrapper89_copies876 a h).2.2.2.1
  have c28036 := (publicBatchWrapper89_copies876 a h).2.2.2.2.1
  have c28037 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_0
  simp only [← c28032, ← c28033, ← c28034] at e_434_0
  have e_434_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_1
  simp only [← c28035, ← c28036, k2, ← c28037] at e_434_1
  have hr := e_434_1
  simp only [e_434_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1106)) := by
  have c28038 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.1
  have c28039 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.1
  have c28040 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.1
  have c28041 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.1
  have c28042 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28043 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_2
  simp only [← c28038, ← c28039, ← c28040] at e_434_2
  have e_434_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_3
  simp only [← c28041, ← c28042, k2, ← c28043] at e_434_3
  have hr := e_434_3
  simp only [e_434_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1107)) := by
  have c28044 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28045 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28046 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28047 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28048 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28049 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_4
  simp only [← c28044, ← c28045, ← c28046] at e_434_4
  have e_434_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_5
  simp only [← c28047, ← c28048, k2, ← c28049] at e_434_5
  have hr := e_434_5
  simp only [e_434_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1108)) := by
  have c28050 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28051 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28052 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28053 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28054 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28055 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_6
  simp only [← c28050, ← c28051, ← c28052] at e_434_6
  have e_434_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_7
  simp only [← c28053, ← c28054, k2, ← c28055] at e_434_7
  have hr := e_434_7
  simp only [e_434_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1109)) := by
  have c28056 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28057 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28058 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28059 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28060 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28061 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_8
  simp only [← c28056, ← c28057, ← c28058] at e_434_8
  have e_434_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_9
  simp only [← c28059, ← c28060, k2, ← c28061] at e_434_9
  have hr := e_434_9
  simp only [e_434_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1110)) := by
  have c28062 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28063 := (publicBatchWrapper89_copies876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28064 := (publicBatchWrapper89_copies877 a h).1
  have c28065 := (publicBatchWrapper89_copies877 a h).2.1
  have c28066 := (publicBatchWrapper89_copies877 a h).2.2.1
  have c28067 := (publicBatchWrapper89_copies877 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_10
  simp only [← c28062, ← c28063, ← c28064] at e_434_10
  have e_434_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_11
  simp only [← c28065, ← c28066, k2, ← c28067] at e_434_11
  have hr := e_434_11
  simp only [e_434_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1111)) := by
  have c28068 := (publicBatchWrapper89_copies877 a h).2.2.2.2.1
  have c28069 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.1
  have c28070 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.1
  have c28071 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.1
  have c28072 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.1
  have c28073 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_12
  simp only [← c28068, ← c28069, ← c28070] at e_434_12
  have e_434_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_13
  simp only [← c28071, ← c28072, k2, ← c28073] at e_434_13
  have hr := e_434_13
  simp only [e_434_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1112)) := by
  have c28074 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28075 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28076 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28077 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28078 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28079 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_14
  simp only [← c28074, ← c28075, ← c28076] at e_434_14
  have e_434_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_15
  simp only [← c28077, ← c28078, k2, ← c28079] at e_434_15
  have hr := e_434_15
  simp only [e_434_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1113)) := by
  have c28080 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28081 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28082 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28083 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28084 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28085 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_16
  simp only [← c28080, ← c28081, ← c28082] at e_434_16
  have e_434_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_17
  simp only [← c28083, ← c28084, k2, ← c28085] at e_434_17
  have hr := e_434_17
  simp only [e_434_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 434 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1114)) := by
  have c28086 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28087 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28088 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28089 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28090 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28091 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_434_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_18
  simp only [← c28086, ← c28087, ← c28088] at e_434_18
  have e_434_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_434_19
  simp only [← c28089, ← c28090, k2, ← c28091] at e_434_19
  have hr := e_434_19
  simp only [e_434_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1115)) := by
  have c28092 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28093 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28094 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28095 := (publicBatchWrapper89_copies877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28096 := (publicBatchWrapper89_copies878 a h).1
  have c28097 := (publicBatchWrapper89_copies878 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_0
  simp only [← c28092, ← c28093, ← c28094] at e_435_0
  have e_435_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_1
  simp only [← c28095, ← c28096, k2, ← c28097] at e_435_1
  have hr := e_435_1
  simp only [e_435_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1116)) := by
  have c28098 := (publicBatchWrapper89_copies878 a h).2.2.1
  have c28099 := (publicBatchWrapper89_copies878 a h).2.2.2.1
  have c28100 := (publicBatchWrapper89_copies878 a h).2.2.2.2.1
  have c28101 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.1
  have c28102 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.1
  have c28103 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_2
  simp only [← c28098, ← c28099, ← c28100] at e_435_2
  have e_435_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_3
  simp only [← c28101, ← c28102, k2, ← c28103] at e_435_3
  have hr := e_435_3
  simp only [e_435_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1117)) := by
  have c28104 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.1
  have c28105 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.1
  have c28106 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28107 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28108 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28109 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_4
  simp only [← c28104, ← c28105, ← c28106] at e_435_4
  have e_435_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_5
  simp only [← c28107, ← c28108, k2, ← c28109] at e_435_5
  have hr := e_435_5
  simp only [e_435_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1118)) := by
  have c28110 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28111 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28112 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28113 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28114 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28115 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_6
  simp only [← c28110, ← c28111, ← c28112] at e_435_6
  have e_435_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_7
  simp only [← c28113, ← c28114, k2, ← c28115] at e_435_7
  have hr := e_435_7
  simp only [e_435_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1119)) := by
  have c28116 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28117 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28118 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28119 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28120 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28121 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_8
  simp only [← c28116, ← c28117, ← c28118] at e_435_8
  have e_435_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_9
  simp only [← c28119, ← c28120, k2, ← c28121] at e_435_9
  have hr := e_435_9
  simp only [e_435_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1120)) := by
  have c28122 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28123 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28124 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28125 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28126 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28127 := (publicBatchWrapper89_copies878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_10
  simp only [← c28122, ← c28123, ← c28124] at e_435_10
  have e_435_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_11
  simp only [← c28125, ← c28126, k2, ← c28127] at e_435_11
  have hr := e_435_11
  simp only [e_435_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1121)) := by
  have c28128 := (publicBatchWrapper89_copies879 a h).1
  have c28129 := (publicBatchWrapper89_copies879 a h).2.1
  have c28130 := (publicBatchWrapper89_copies879 a h).2.2.1
  have c28131 := (publicBatchWrapper89_copies879 a h).2.2.2.1
  have c28132 := (publicBatchWrapper89_copies879 a h).2.2.2.2.1
  have c28133 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_12
  simp only [← c28128, ← c28129, ← c28130] at e_435_12
  have e_435_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_13
  simp only [← c28131, ← c28132, k2, ← c28133] at e_435_13
  have hr := e_435_13
  simp only [e_435_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1122)) := by
  have c28134 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.1
  have c28135 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.1
  have c28136 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.1
  have c28137 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.1
  have c28138 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28139 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_14
  simp only [← c28134, ← c28135, ← c28136] at e_435_14
  have e_435_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_15
  simp only [← c28137, ← c28138, k2, ← c28139] at e_435_15
  have hr := e_435_15
  simp only [e_435_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1123)) := by
  have c28140 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28141 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28142 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28143 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28144 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28145 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_16
  simp only [← c28140, ← c28141, ← c28142] at e_435_16
  have e_435_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_17
  simp only [← c28143, ← c28144, k2, ← c28145] at e_435_17
  have hr := e_435_17
  simp only [e_435_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 435 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1124)) := by
  have c28146 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28147 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28148 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28149 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28150 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28151 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_435_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_18
  simp only [← c28146, ← c28147, ← c28148] at e_435_18
  have e_435_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_435_19
  simp only [← c28149, ← c28150, k2, ← c28151] at e_435_19
  have hr := e_435_19
  simp only [e_435_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1125)) := by
  have c28152 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28153 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28154 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28155 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28156 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28157 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_0
  simp only [← c28152, ← c28153, ← c28154] at e_436_0
  have e_436_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_1
  simp only [← c28155, ← c28156, k2, ← c28157] at e_436_1
  have hr := e_436_1
  simp only [e_436_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1126)) := by
  have c28158 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28159 := (publicBatchWrapper89_copies879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28160 := (publicBatchWrapper89_copies880 a h).1
  have c28161 := (publicBatchWrapper89_copies880 a h).2.1
  have c28162 := (publicBatchWrapper89_copies880 a h).2.2.1
  have c28163 := (publicBatchWrapper89_copies880 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_2
  simp only [← c28158, ← c28159, ← c28160] at e_436_2
  have e_436_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_3
  simp only [← c28161, ← c28162, k2, ← c28163] at e_436_3
  have hr := e_436_3
  simp only [e_436_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1127)) := by
  have c28164 := (publicBatchWrapper89_copies880 a h).2.2.2.2.1
  have c28165 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.1
  have c28166 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.1
  have c28167 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.1
  have c28168 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.1
  have c28169 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_4
  simp only [← c28164, ← c28165, ← c28166] at e_436_4
  have e_436_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_5
  simp only [← c28167, ← c28168, k2, ← c28169] at e_436_5
  have hr := e_436_5
  simp only [e_436_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1128)) := by
  have c28170 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28171 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28172 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28173 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28174 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28175 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_6
  simp only [← c28170, ← c28171, ← c28172] at e_436_6
  have e_436_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_7
  simp only [← c28173, ← c28174, k2, ← c28175] at e_436_7
  have hr := e_436_7
  simp only [e_436_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1129)) := by
  have c28176 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28177 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28178 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28179 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28180 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28181 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_8
  simp only [← c28176, ← c28177, ← c28178] at e_436_8
  have e_436_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_9
  simp only [← c28179, ← c28180, k2, ← c28181] at e_436_9
  have hr := e_436_9
  simp only [e_436_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1130)) := by
  have c28182 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28183 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28184 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28185 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28186 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28187 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_10
  simp only [← c28182, ← c28183, ← c28184] at e_436_10
  have e_436_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_11
  simp only [← c28185, ← c28186, k2, ← c28187] at e_436_11
  have hr := e_436_11
  simp only [e_436_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1131)) := by
  have c28188 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28189 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28190 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28191 := (publicBatchWrapper89_copies880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28192 := (publicBatchWrapper89_copies881 a h).1
  have c28193 := (publicBatchWrapper89_copies881 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_12
  simp only [← c28188, ← c28189, ← c28190] at e_436_12
  have e_436_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_13
  simp only [← c28191, ← c28192, k2, ← c28193] at e_436_13
  have hr := e_436_13
  simp only [e_436_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1132)) := by
  have c28194 := (publicBatchWrapper89_copies881 a h).2.2.1
  have c28195 := (publicBatchWrapper89_copies881 a h).2.2.2.1
  have c28196 := (publicBatchWrapper89_copies881 a h).2.2.2.2.1
  have c28197 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.1
  have c28198 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.1
  have c28199 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_14
  simp only [← c28194, ← c28195, ← c28196] at e_436_14
  have e_436_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_15
  simp only [← c28197, ← c28198, k2, ← c28199] at e_436_15
  have hr := e_436_15
  simp only [e_436_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1133)) := by
  have c28200 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.1
  have c28201 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.1
  have c28202 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28203 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28204 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28205 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_16
  simp only [← c28200, ← c28201, ← c28202] at e_436_16
  have e_436_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_17
  simp only [← c28203, ← c28204, k2, ← c28205] at e_436_17
  have hr := e_436_17
  simp only [e_436_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 436 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1134)) := by
  have c28206 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28207 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28208 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28209 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28210 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28211 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_436_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_18
  simp only [← c28206, ← c28207, ← c28208] at e_436_18
  have e_436_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_436_19
  simp only [← c28209, ← c28210, k2, ← c28211] at e_436_19
  have hr := e_436_19
  simp only [e_436_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1135)) := by
  have c28212 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28213 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28214 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28215 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28216 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28217 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_0
  simp only [← c28212, ← c28213, ← c28214] at e_437_0
  have e_437_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_1
  simp only [← c28215, ← c28216, k2, ← c28217] at e_437_1
  have hr := e_437_1
  simp only [e_437_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1136)) := by
  have c28218 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28219 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28220 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28221 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28222 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28223 := (publicBatchWrapper89_copies881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_2
  simp only [← c28218, ← c28219, ← c28220] at e_437_2
  have e_437_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_3
  simp only [← c28221, ← c28222, k2, ← c28223] at e_437_3
  have hr := e_437_3
  simp only [e_437_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1137)) := by
  have c28224 := (publicBatchWrapper89_copies882 a h).1
  have c28225 := (publicBatchWrapper89_copies882 a h).2.1
  have c28226 := (publicBatchWrapper89_copies882 a h).2.2.1
  have c28227 := (publicBatchWrapper89_copies882 a h).2.2.2.1
  have c28228 := (publicBatchWrapper89_copies882 a h).2.2.2.2.1
  have c28229 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_4
  simp only [← c28224, ← c28225, ← c28226] at e_437_4
  have e_437_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_5
  simp only [← c28227, ← c28228, k2, ← c28229] at e_437_5
  have hr := e_437_5
  simp only [e_437_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1138)) := by
  have c28230 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.1
  have c28231 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.1
  have c28232 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.1
  have c28233 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.1
  have c28234 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28235 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_6
  simp only [← c28230, ← c28231, ← c28232] at e_437_6
  have e_437_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_7
  simp only [← c28233, ← c28234, k2, ← c28235] at e_437_7
  have hr := e_437_7
  simp only [e_437_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1139)) := by
  have c28236 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28237 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28238 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28239 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28240 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28241 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_8
  simp only [← c28236, ← c28237, ← c28238] at e_437_8
  have e_437_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_9
  simp only [← c28239, ← c28240, k2, ← c28241] at e_437_9
  have hr := e_437_9
  simp only [e_437_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1140)) := by
  have c28242 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28243 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28244 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28245 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28246 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28247 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_10
  simp only [← c28242, ← c28243, ← c28244] at e_437_10
  have e_437_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_11
  simp only [← c28245, ← c28246, k2, ← c28247] at e_437_11
  have hr := e_437_11
  simp only [e_437_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1141)) := by
  have c28248 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28249 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28250 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28251 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28252 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28253 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_12
  simp only [← c28248, ← c28249, ← c28250] at e_437_12
  have e_437_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_13
  simp only [← c28251, ← c28252, k2, ← c28253] at e_437_13
  have hr := e_437_13
  simp only [e_437_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1142)) := by
  have c28254 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28255 := (publicBatchWrapper89_copies882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28256 := (publicBatchWrapper89_copies883 a h).1
  have c28257 := (publicBatchWrapper89_copies883 a h).2.1
  have c28258 := (publicBatchWrapper89_copies883 a h).2.2.1
  have c28259 := (publicBatchWrapper89_copies883 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_14
  simp only [← c28254, ← c28255, ← c28256] at e_437_14
  have e_437_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_15
  simp only [← c28257, ← c28258, k2, ← c28259] at e_437_15
  have hr := e_437_15
  simp only [e_437_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1143)) := by
  have c28260 := (publicBatchWrapper89_copies883 a h).2.2.2.2.1
  have c28261 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.1
  have c28262 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.1
  have c28263 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.1
  have c28264 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.1
  have c28265 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_16
  simp only [← c28260, ← c28261, ← c28262] at e_437_16
  have e_437_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_17
  simp only [← c28263, ← c28264, k2, ← c28265] at e_437_17
  have hr := e_437_17
  simp only [e_437_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 437 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1144)) := by
  have c28266 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28267 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28268 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28269 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28270 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28271 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_437_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_18
  simp only [← c28266, ← c28267, ← c28268] at e_437_18
  have e_437_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_437_19
  simp only [← c28269, ← c28270, k2, ← c28271] at e_437_19
  have hr := e_437_19
  simp only [e_437_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1145)) := by
  have c28272 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28273 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28274 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28275 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28276 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28277 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_0
  simp only [← c28272, ← c28273, ← c28274] at e_438_0
  have e_438_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_1
  simp only [← c28275, ← c28276, k2, ← c28277] at e_438_1
  have hr := e_438_1
  simp only [e_438_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1146)) := by
  have c28278 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28279 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28280 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28281 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28282 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28283 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_2
  simp only [← c28278, ← c28279, ← c28280] at e_438_2
  have e_438_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_3
  simp only [← c28281, ← c28282, k2, ← c28283] at e_438_3
  have hr := e_438_3
  simp only [e_438_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1147)) := by
  have c28284 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28285 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28286 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28287 := (publicBatchWrapper89_copies883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28288 := (publicBatchWrapper89_copies884 a h).1
  have c28289 := (publicBatchWrapper89_copies884 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_4
  simp only [← c28284, ← c28285, ← c28286] at e_438_4
  have e_438_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_5
  simp only [← c28287, ← c28288, k2, ← c28289] at e_438_5
  have hr := e_438_5
  simp only [e_438_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1148)) := by
  have c28290 := (publicBatchWrapper89_copies884 a h).2.2.1
  have c28291 := (publicBatchWrapper89_copies884 a h).2.2.2.1
  have c28292 := (publicBatchWrapper89_copies884 a h).2.2.2.2.1
  have c28293 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.1
  have c28294 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.1
  have c28295 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_6
  simp only [← c28290, ← c28291, ← c28292] at e_438_6
  have e_438_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_7
  simp only [← c28293, ← c28294, k2, ← c28295] at e_438_7
  have hr := e_438_7
  simp only [e_438_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1149)) := by
  have c28296 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.1
  have c28297 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.1
  have c28298 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28299 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28300 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28301 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_8
  simp only [← c28296, ← c28297, ← c28298] at e_438_8
  have e_438_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_9
  simp only [← c28299, ← c28300, k2, ← c28301] at e_438_9
  have hr := e_438_9
  simp only [e_438_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1150)) := by
  have c28302 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28303 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28304 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28305 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28306 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28307 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_10
  simp only [← c28302, ← c28303, ← c28304] at e_438_10
  have e_438_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_11
  simp only [← c28305, ← c28306, k2, ← c28307] at e_438_11
  have hr := e_438_11
  simp only [e_438_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1151)) := by
  have c28308 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28309 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28310 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28311 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28312 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28313 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_12
  simp only [← c28308, ← c28309, ← c28310] at e_438_12
  have e_438_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_13
  simp only [← c28311, ← c28312, k2, ← c28313] at e_438_13
  have hr := e_438_13
  simp only [e_438_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1152)) := by
  have c28314 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28315 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28316 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28317 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28318 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28319 := (publicBatchWrapper89_copies884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_14
  simp only [← c28314, ← c28315, ← c28316] at e_438_14
  have e_438_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_15
  simp only [← c28317, ← c28318, k2, ← c28319] at e_438_15
  have hr := e_438_15
  simp only [e_438_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1153)) := by
  have c28320 := (publicBatchWrapper89_copies885 a h).1
  have c28321 := (publicBatchWrapper89_copies885 a h).2.1
  have c28322 := (publicBatchWrapper89_copies885 a h).2.2.1
  have c28323 := (publicBatchWrapper89_copies885 a h).2.2.2.1
  have c28324 := (publicBatchWrapper89_copies885 a h).2.2.2.2.1
  have c28325 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_16
  simp only [← c28320, ← c28321, ← c28322] at e_438_16
  have e_438_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_17
  simp only [← c28323, ← c28324, k2, ← c28325] at e_438_17
  have hr := e_438_17
  simp only [e_438_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 438 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1154)) := by
  have c28326 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.1
  have c28327 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.1
  have c28328 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.1
  have c28329 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.1
  have c28330 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28331 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_438_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_18
  simp only [← c28326, ← c28327, ← c28328] at e_438_18
  have e_438_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_438_19
  simp only [← c28329, ← c28330, k2, ← c28331] at e_438_19
  have hr := e_438_19
  simp only [e_438_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1155)) := by
  have c28332 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28333 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28334 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28335 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28336 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28337 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_0
  simp only [← c28332, ← c28333, ← c28334] at e_439_0
  have e_439_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_1
  simp only [← c28335, ← c28336, k2, ← c28337] at e_439_1
  have hr := e_439_1
  simp only [e_439_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1156)) := by
  have c28338 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28339 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28340 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28341 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28342 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28343 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_2
  simp only [← c28338, ← c28339, ← c28340] at e_439_2
  have e_439_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_3
  simp only [← c28341, ← c28342, k2, ← c28343] at e_439_3
  have hr := e_439_3
  simp only [e_439_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1157)) := by
  have c28344 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28345 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28346 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28347 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28348 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28349 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_4
  simp only [← c28344, ← c28345, ← c28346] at e_439_4
  have e_439_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_5
  simp only [← c28347, ← c28348, k2, ← c28349] at e_439_5
  have hr := e_439_5
  simp only [e_439_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1158)) := by
  have c28350 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28351 := (publicBatchWrapper89_copies885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28352 := (publicBatchWrapper89_copies886 a h).1
  have c28353 := (publicBatchWrapper89_copies886 a h).2.1
  have c28354 := (publicBatchWrapper89_copies886 a h).2.2.1
  have c28355 := (publicBatchWrapper89_copies886 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_6
  simp only [← c28350, ← c28351, ← c28352] at e_439_6
  have e_439_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_7
  simp only [← c28353, ← c28354, k2, ← c28355] at e_439_7
  have hr := e_439_7
  simp only [e_439_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1159)) := by
  have c28356 := (publicBatchWrapper89_copies886 a h).2.2.2.2.1
  have c28357 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.1
  have c28358 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.1
  have c28359 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.1
  have c28360 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.1
  have c28361 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_8
  simp only [← c28356, ← c28357, ← c28358] at e_439_8
  have e_439_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_9
  simp only [← c28359, ← c28360, k2, ← c28361] at e_439_9
  have hr := e_439_9
  simp only [e_439_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1160)) := by
  have c28362 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28363 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28364 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28365 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28366 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28367 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_10
  simp only [← c28362, ← c28363, ← c28364] at e_439_10
  have e_439_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_11
  simp only [← c28365, ← c28366, k2, ← c28367] at e_439_11
  have hr := e_439_11
  simp only [e_439_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1161)) := by
  have c28368 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28369 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28370 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28371 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28372 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28373 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_12
  simp only [← c28368, ← c28369, ← c28370] at e_439_12
  have e_439_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_13
  simp only [← c28371, ← c28372, k2, ← c28373] at e_439_13
  have hr := e_439_13
  simp only [e_439_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1162)) := by
  have c28374 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28375 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28376 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28377 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28378 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28379 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_14
  simp only [← c28374, ← c28375, ← c28376] at e_439_14
  have e_439_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_15
  simp only [← c28377, ← c28378, k2, ← c28379] at e_439_15
  have hr := e_439_15
  simp only [e_439_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1163)) := by
  have c28380 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28381 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28382 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28383 := (publicBatchWrapper89_copies886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28384 := (publicBatchWrapper89_copies887 a h).1
  have c28385 := (publicBatchWrapper89_copies887 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_16
  simp only [← c28380, ← c28381, ← c28382] at e_439_16
  have e_439_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_17
  simp only [← c28383, ← c28384, k2, ← c28385] at e_439_17
  have hr := e_439_17
  simp only [e_439_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 439 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1164)) := by
  have c28386 := (publicBatchWrapper89_copies887 a h).2.2.1
  have c28387 := (publicBatchWrapper89_copies887 a h).2.2.2.1
  have c28388 := (publicBatchWrapper89_copies887 a h).2.2.2.2.1
  have c28389 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.1
  have c28390 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.1
  have c28391 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_439_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_18
  simp only [← c28386, ← c28387, ← c28388] at e_439_18
  have e_439_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_439_19
  simp only [← c28389, ← c28390, k2, ← c28391] at e_439_19
  have hr := e_439_19
  simp only [e_439_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1165)) := by
  have c28392 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.1
  have c28393 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.1
  have c28394 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28395 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28396 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28397 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_0
  simp only [← c28392, ← c28393, ← c28394] at e_440_0
  have e_440_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_1
  simp only [← c28395, ← c28396, k2, ← c28397] at e_440_1
  have hr := e_440_1
  simp only [e_440_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1166)) := by
  have c28398 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28399 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28400 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28401 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28402 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28403 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_2
  simp only [← c28398, ← c28399, ← c28400] at e_440_2
  have e_440_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_3
  simp only [← c28401, ← c28402, k2, ← c28403] at e_440_3
  have hr := e_440_3
  simp only [e_440_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1167)) := by
  have c28404 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28405 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28406 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28407 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28408 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28409 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_4
  simp only [← c28404, ← c28405, ← c28406] at e_440_4
  have e_440_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_5
  simp only [← c28407, ← c28408, k2, ← c28409] at e_440_5
  have hr := e_440_5
  simp only [e_440_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1168)) := by
  have c28410 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28411 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28412 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28413 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28414 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28415 := (publicBatchWrapper89_copies887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_6
  simp only [← c28410, ← c28411, ← c28412] at e_440_6
  have e_440_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_7
  simp only [← c28413, ← c28414, k2, ← c28415] at e_440_7
  have hr := e_440_7
  simp only [e_440_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1169)) := by
  have c28416 := (publicBatchWrapper89_copies888 a h).1
  have c28417 := (publicBatchWrapper89_copies888 a h).2.1
  have c28418 := (publicBatchWrapper89_copies888 a h).2.2.1
  have c28419 := (publicBatchWrapper89_copies888 a h).2.2.2.1
  have c28420 := (publicBatchWrapper89_copies888 a h).2.2.2.2.1
  have c28421 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_8
  simp only [← c28416, ← c28417, ← c28418] at e_440_8
  have e_440_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_9
  simp only [← c28419, ← c28420, k2, ← c28421] at e_440_9
  have hr := e_440_9
  simp only [e_440_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1210)) := by
  have c28422 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.1
  have c28423 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.1
  have c28424 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.1
  have c28425 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.1
  have c28426 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28427 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_10
  simp only [← c28422, ← c28423, ← c28424] at e_440_10
  have e_440_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_11
  simp only [← c28425, ← c28426, k2, ← c28427] at e_440_11
  have hr := e_440_11
  simp only [e_440_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1211)) := by
  have c28428 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28429 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28430 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28431 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28432 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28433 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_12
  simp only [← c28428, ← c28429, ← c28430] at e_440_12
  have e_440_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_13
  simp only [← c28431, ← c28432, k2, ← c28433] at e_440_13
  have hr := e_440_13
  simp only [e_440_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1212)) := by
  have c28434 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28435 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28436 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28437 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28438 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28439 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_14
  simp only [← c28434, ← c28435, ← c28436] at e_440_14
  have e_440_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_15
  simp only [← c28437, ← c28438, k2, ← c28439] at e_440_15
  have hr := e_440_15
  simp only [e_440_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1213)) := by
  have c28440 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28441 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28442 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28443 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28444 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28445 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_16
  simp only [← c28440, ← c28441, ← c28442] at e_440_16
  have e_440_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_17
  simp only [← c28443, ← c28444, k2, ← c28445] at e_440_17
  have hr := e_440_17
  simp only [e_440_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 440 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1214)) := by
  have c28446 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28447 := (publicBatchWrapper89_copies888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28448 := (publicBatchWrapper89_copies889 a h).1
  have c28449 := (publicBatchWrapper89_copies889 a h).2.1
  have c28450 := (publicBatchWrapper89_copies889 a h).2.2.1
  have c28451 := (publicBatchWrapper89_copies889 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_440_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_18
  simp only [← c28446, ← c28447, ← c28448] at e_440_18
  have e_440_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_440_19
  simp only [← c28449, ← c28450, k2, ← c28451] at e_440_19
  have hr := e_440_19
  simp only [e_440_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1215)) := by
  have c28452 := (publicBatchWrapper89_copies889 a h).2.2.2.2.1
  have c28453 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.1
  have c28454 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.1
  have c28455 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.1
  have c28456 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.1
  have c28457 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_0
  simp only [← c28452, ← c28453, ← c28454] at e_441_0
  have e_441_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_1
  simp only [← c28455, ← c28456, k2, ← c28457] at e_441_1
  have hr := e_441_1
  simp only [e_441_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1216)) := by
  have c28458 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28459 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28460 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28461 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28462 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28463 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_2
  simp only [← c28458, ← c28459, ← c28460] at e_441_2
  have e_441_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_3
  simp only [← c28461, ← c28462, k2, ← c28463] at e_441_3
  have hr := e_441_3
  simp only [e_441_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1217)) := by
  have c28464 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28465 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28466 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28467 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28468 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28469 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_4
  simp only [← c28464, ← c28465, ← c28466] at e_441_4
  have e_441_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_5
  simp only [← c28467, ← c28468, k2, ← c28469] at e_441_5
  have hr := e_441_5
  simp only [e_441_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1218)) := by
  have c28470 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28471 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28472 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28473 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28474 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28475 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_6
  simp only [← c28470, ← c28471, ← c28472] at e_441_6
  have e_441_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_7
  simp only [← c28473, ← c28474, k2, ← c28475] at e_441_7
  have hr := e_441_7
  simp only [e_441_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1219)) := by
  have c28476 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28477 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28478 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28479 := (publicBatchWrapper89_copies889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28480 := (publicBatchWrapper89_copies890 a h).1
  have c28481 := (publicBatchWrapper89_copies890 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_8
  simp only [← c28476, ← c28477, ← c28478] at e_441_8
  have e_441_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_9
  simp only [← c28479, ← c28480, k2, ← c28481] at e_441_9
  have hr := e_441_9
  simp only [e_441_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1220)) := by
  have c28482 := (publicBatchWrapper89_copies890 a h).2.2.1
  have c28483 := (publicBatchWrapper89_copies890 a h).2.2.2.1
  have c28484 := (publicBatchWrapper89_copies890 a h).2.2.2.2.1
  have c28485 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.1
  have c28486 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.1
  have c28487 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_10
  simp only [← c28482, ← c28483, ← c28484] at e_441_10
  have e_441_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_11
  simp only [← c28485, ← c28486, k2, ← c28487] at e_441_11
  have hr := e_441_11
  simp only [e_441_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1221)) := by
  have c28488 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.1
  have c28489 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.1
  have c28490 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28491 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28492 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28493 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_12
  simp only [← c28488, ← c28489, ← c28490] at e_441_12
  have e_441_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_13
  simp only [← c28491, ← c28492, k2, ← c28493] at e_441_13
  have hr := e_441_13
  simp only [e_441_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1222)) := by
  have c28494 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28495 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28496 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28497 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28498 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28499 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_14
  simp only [← c28494, ← c28495, ← c28496] at e_441_14
  have e_441_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_15
  simp only [← c28497, ← c28498, k2, ← c28499] at e_441_15
  have hr := e_441_15
  simp only [e_441_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f3999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1223)) := by
  have c28500 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28501 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28502 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28503 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28504 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28505 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_16
  simp only [← c28500, ← c28501, ← c28502] at e_441_16
  have e_441_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_17
  simp only [← c28503, ← c28504, k2, ← c28505] at e_441_17
  have hr := e_441_17
  simp only [e_441_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 441 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1224)) := by
  have c28506 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28507 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28508 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28509 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28510 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28511 := (publicBatchWrapper89_copies890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_441_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_18
  simp only [← c28506, ← c28507, ← c28508] at e_441_18
  have e_441_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_441_19
  simp only [← c28509, ← c28510, k2, ← c28511] at e_441_19
  have hr := e_441_19
  simp only [e_441_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1225)) := by
  have c28512 := (publicBatchWrapper89_copies891 a h).1
  have c28513 := (publicBatchWrapper89_copies891 a h).2.1
  have c28514 := (publicBatchWrapper89_copies891 a h).2.2.1
  have c28515 := (publicBatchWrapper89_copies891 a h).2.2.2.1
  have c28516 := (publicBatchWrapper89_copies891 a h).2.2.2.2.1
  have c28517 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_0
  simp only [← c28512, ← c28513, ← c28514] at e_442_0
  have e_442_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_1
  simp only [← c28515, ← c28516, k2, ← c28517] at e_442_1
  have hr := e_442_1
  simp only [e_442_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1226)) := by
  have c28518 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.1
  have c28519 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.1
  have c28520 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.1
  have c28521 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.1
  have c28522 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28523 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_2
  simp only [← c28518, ← c28519, ← c28520] at e_442_2
  have e_442_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_3
  simp only [← c28521, ← c28522, k2, ← c28523] at e_442_3
  have hr := e_442_3
  simp only [e_442_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1227)) := by
  have c28524 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28525 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28526 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28527 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28528 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28529 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_4
  simp only [← c28524, ← c28525, ← c28526] at e_442_4
  have e_442_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_5
  simp only [← c28527, ← c28528, k2, ← c28529] at e_442_5
  have hr := e_442_5
  simp only [e_442_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1228)) := by
  have c28530 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28531 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28532 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28533 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28534 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28535 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_6
  simp only [← c28530, ← c28531, ← c28532] at e_442_6
  have e_442_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_7
  simp only [← c28533, ← c28534, k2, ← c28535] at e_442_7
  have hr := e_442_7
  simp only [e_442_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1229)) := by
  have c28536 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28537 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28538 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28539 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28540 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28541 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_8
  simp only [← c28536, ← c28537, ← c28538] at e_442_8
  have e_442_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_9
  simp only [← c28539, ← c28540, k2, ← c28541] at e_442_9
  have hr := e_442_9
  simp only [e_442_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1230)) := by
  have c28542 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28543 := (publicBatchWrapper89_copies891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28544 := (publicBatchWrapper89_copies892 a h).1
  have c28545 := (publicBatchWrapper89_copies892 a h).2.1
  have c28546 := (publicBatchWrapper89_copies892 a h).2.2.1
  have c28547 := (publicBatchWrapper89_copies892 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_10
  simp only [← c28542, ← c28543, ← c28544] at e_442_10
  have e_442_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_11
  simp only [← c28545, ← c28546, k2, ← c28547] at e_442_11
  have hr := e_442_11
  simp only [e_442_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1231)) := by
  have c28548 := (publicBatchWrapper89_copies892 a h).2.2.2.2.1
  have c28549 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.1
  have c28550 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.1
  have c28551 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.1
  have c28552 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.1
  have c28553 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_12
  simp only [← c28548, ← c28549, ← c28550] at e_442_12
  have e_442_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_13
  simp only [← c28551, ← c28552, k2, ← c28553] at e_442_13
  have hr := e_442_13
  simp only [e_442_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1232)) := by
  have c28554 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28555 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28556 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28557 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28558 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28559 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_14
  simp only [← c28554, ← c28555, ← c28556] at e_442_14
  have e_442_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_15
  simp only [← c28557, ← c28558, k2, ← c28559] at e_442_15
  have hr := e_442_15
  simp only [e_442_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1233)) := by
  have c28560 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28561 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28562 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28563 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28564 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28565 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_16
  simp only [← c28560, ← c28561, ← c28562] at e_442_16
  have e_442_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_17
  simp only [← c28563, ← c28564, k2, ← c28565] at e_442_17
  have hr := e_442_17
  simp only [e_442_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 442 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1234)) := by
  have c28566 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28567 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28568 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28569 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28570 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28571 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_442_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_18
  simp only [← c28566, ← c28567, ← c28568] at e_442_18
  have e_442_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_442_19
  simp only [← c28569, ← c28570, k2, ← c28571] at e_442_19
  have hr := e_442_19
  simp only [e_442_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1235)) := by
  have c28572 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28573 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28574 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28575 := (publicBatchWrapper89_copies892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28576 := (publicBatchWrapper89_copies893 a h).1
  have c28577 := (publicBatchWrapper89_copies893 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_0
  simp only [← c28572, ← c28573, ← c28574] at e_443_0
  have e_443_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_1
  simp only [← c28575, ← c28576, k2, ← c28577] at e_443_1
  have hr := e_443_1
  simp only [e_443_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1236)) := by
  have c28578 := (publicBatchWrapper89_copies893 a h).2.2.1
  have c28579 := (publicBatchWrapper89_copies893 a h).2.2.2.1
  have c28580 := (publicBatchWrapper89_copies893 a h).2.2.2.2.1
  have c28581 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.1
  have c28582 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.1
  have c28583 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_2
  simp only [← c28578, ← c28579, ← c28580] at e_443_2
  have e_443_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_3
  simp only [← c28581, ← c28582, k2, ← c28583] at e_443_3
  have hr := e_443_3
  simp only [e_443_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1237)) := by
  have c28584 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.1
  have c28585 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.1
  have c28586 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28587 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28588 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28589 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_4
  simp only [← c28584, ← c28585, ← c28586] at e_443_4
  have e_443_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_5
  simp only [← c28587, ← c28588, k2, ← c28589] at e_443_5
  have hr := e_443_5
  simp only [e_443_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1238)) := by
  have c28590 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28591 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28592 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28593 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28594 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28595 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_6
  simp only [← c28590, ← c28591, ← c28592] at e_443_6
  have e_443_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_7
  simp only [← c28593, ← c28594, k2, ← c28595] at e_443_7
  have hr := e_443_7
  simp only [e_443_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1239)) := by
  have c28596 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28597 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28598 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28599 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28600 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28601 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_8
  simp only [← c28596, ← c28597, ← c28598] at e_443_8
  have e_443_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_9
  simp only [← c28599, ← c28600, k2, ← c28601] at e_443_9
  have hr := e_443_9
  simp only [e_443_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1240)) := by
  have c28602 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28603 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28604 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28605 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28606 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28607 := (publicBatchWrapper89_copies893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_10
  simp only [← c28602, ← c28603, ← c28604] at e_443_10
  have e_443_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_11
  simp only [← c28605, ← c28606, k2, ← c28607] at e_443_11
  have hr := e_443_11
  simp only [e_443_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1241)) := by
  have c28608 := (publicBatchWrapper89_copies894 a h).1
  have c28609 := (publicBatchWrapper89_copies894 a h).2.1
  have c28610 := (publicBatchWrapper89_copies894 a h).2.2.1
  have c28611 := (publicBatchWrapper89_copies894 a h).2.2.2.1
  have c28612 := (publicBatchWrapper89_copies894 a h).2.2.2.2.1
  have c28613 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_12
  simp only [← c28608, ← c28609, ← c28610] at e_443_12
  have e_443_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_13
  simp only [← c28611, ← c28612, k2, ← c28613] at e_443_13
  have hr := e_443_13
  simp only [e_443_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1242)) := by
  have c28614 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.1
  have c28615 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.1
  have c28616 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.1
  have c28617 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.1
  have c28618 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28619 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_14
  simp only [← c28614, ← c28615, ← c28616] at e_443_14
  have e_443_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_15
  simp only [← c28617, ← c28618, k2, ← c28619] at e_443_15
  have hr := e_443_15
  simp only [e_443_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1243)) := by
  have c28620 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28621 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28622 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28623 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28624 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28625 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_16
  simp only [← c28620, ← c28621, ← c28622] at e_443_16
  have e_443_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_17
  simp only [← c28623, ← c28624, k2, ← c28625] at e_443_17
  have hr := e_443_17
  simp only [e_443_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 443 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1244)) := by
  have c28626 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28627 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28628 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28629 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28630 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28631 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_443_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_18
  simp only [← c28626, ← c28627, ← c28628] at e_443_18
  have e_443_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_443_19
  simp only [← c28629, ← c28630, k2, ← c28631] at e_443_19
  have hr := e_443_19
  simp only [e_443_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1245)) := by
  have c28632 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28633 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28634 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28635 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28636 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28637 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_0
  simp only [← c28632, ← c28633, ← c28634] at e_444_0
  have e_444_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_1
  simp only [← c28635, ← c28636, k2, ← c28637] at e_444_1
  have hr := e_444_1
  simp only [e_444_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1246)) := by
  have c28638 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28639 := (publicBatchWrapper89_copies894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28640 := (publicBatchWrapper89_copies895 a h).1
  have c28641 := (publicBatchWrapper89_copies895 a h).2.1
  have c28642 := (publicBatchWrapper89_copies895 a h).2.2.1
  have c28643 := (publicBatchWrapper89_copies895 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_2
  simp only [← c28638, ← c28639, ← c28640] at e_444_2
  have e_444_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_3
  simp only [← c28641, ← c28642, k2, ← c28643] at e_444_3
  have hr := e_444_3
  simp only [e_444_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1247)) := by
  have c28644 := (publicBatchWrapper89_copies895 a h).2.2.2.2.1
  have c28645 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.1
  have c28646 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.1
  have c28647 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.1
  have c28648 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.1
  have c28649 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_4
  simp only [← c28644, ← c28645, ← c28646] at e_444_4
  have e_444_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_5
  simp only [← c28647, ← c28648, k2, ← c28649] at e_444_5
  have hr := e_444_5
  simp only [e_444_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1248)) := by
  have c28650 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28651 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28652 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28653 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28654 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28655 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_6
  simp only [← c28650, ← c28651, ← c28652] at e_444_6
  have e_444_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_7
  simp only [← c28653, ← c28654, k2, ← c28655] at e_444_7
  have hr := e_444_7
  simp only [e_444_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1249)) := by
  have c28656 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28657 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28658 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28659 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28660 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28661 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_8
  simp only [← c28656, ← c28657, ← c28658] at e_444_8
  have e_444_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_9
  simp only [← c28659, ← c28660, k2, ← c28661] at e_444_9
  have hr := e_444_9
  simp only [e_444_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1250)) := by
  have c28662 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28663 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28664 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28665 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28666 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28667 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_10
  simp only [← c28662, ← c28663, ← c28664] at e_444_10
  have e_444_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_11
  simp only [← c28665, ← c28666, k2, ← c28667] at e_444_11
  have hr := e_444_11
  simp only [e_444_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1251)) := by
  have c28668 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28669 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28670 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28671 := (publicBatchWrapper89_copies895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28672 := (publicBatchWrapper89_copies896 a h).1
  have c28673 := (publicBatchWrapper89_copies896 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_12
  simp only [← c28668, ← c28669, ← c28670] at e_444_12
  have e_444_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_13
  simp only [← c28671, ← c28672, k2, ← c28673] at e_444_13
  have hr := e_444_13
  simp only [e_444_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1252)) := by
  have c28674 := (publicBatchWrapper89_copies896 a h).2.2.1
  have c28675 := (publicBatchWrapper89_copies896 a h).2.2.2.1
  have c28676 := (publicBatchWrapper89_copies896 a h).2.2.2.2.1
  have c28677 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.1
  have c28678 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.1
  have c28679 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_14
  simp only [← c28674, ← c28675, ← c28676] at e_444_14
  have e_444_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_15
  simp only [← c28677, ← c28678, k2, ← c28679] at e_444_15
  have hr := e_444_15
  simp only [e_444_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1253)) := by
  have c28680 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.1
  have c28681 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.1
  have c28682 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28683 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28684 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28685 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_16
  simp only [← c28680, ← c28681, ← c28682] at e_444_16
  have e_444_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_17
  simp only [← c28683, ← c28684, k2, ← c28685] at e_444_17
  have hr := e_444_17
  simp only [e_444_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 444 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1254)) := by
  have c28686 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28687 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28688 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28689 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28690 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28691 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_444_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_18
  simp only [← c28686, ← c28687, ← c28688] at e_444_18
  have e_444_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_444_19
  simp only [← c28689, ← c28690, k2, ← c28691] at e_444_19
  have hr := e_444_19
  simp only [e_444_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1255)) := by
  have c28692 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28693 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28694 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28695 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28696 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28697 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_0
  simp only [← c28692, ← c28693, ← c28694] at e_445_0
  have e_445_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_1
  simp only [← c28695, ← c28696, k2, ← c28697] at e_445_1
  have hr := e_445_1
  simp only [e_445_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1256)) := by
  have c28698 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28699 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28700 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28701 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28702 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28703 := (publicBatchWrapper89_copies896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_2
  simp only [← c28698, ← c28699, ← c28700] at e_445_2
  have e_445_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_3
  simp only [← c28701, ← c28702, k2, ← c28703] at e_445_3
  have hr := e_445_3
  simp only [e_445_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1257)) := by
  have c28704 := (publicBatchWrapper89_copies897 a h).1
  have c28705 := (publicBatchWrapper89_copies897 a h).2.1
  have c28706 := (publicBatchWrapper89_copies897 a h).2.2.1
  have c28707 := (publicBatchWrapper89_copies897 a h).2.2.2.1
  have c28708 := (publicBatchWrapper89_copies897 a h).2.2.2.2.1
  have c28709 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_4
  simp only [← c28704, ← c28705, ← c28706] at e_445_4
  have e_445_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_5
  simp only [← c28707, ← c28708, k2, ← c28709] at e_445_5
  have hr := e_445_5
  simp only [e_445_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1258)) := by
  have c28710 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.1
  have c28711 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.1
  have c28712 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.1
  have c28713 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.1
  have c28714 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28715 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_6
  simp only [← c28710, ← c28711, ← c28712] at e_445_6
  have e_445_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_7
  simp only [← c28713, ← c28714, k2, ← c28715] at e_445_7
  have hr := e_445_7
  simp only [e_445_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1259)) := by
  have c28716 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28717 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28718 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28719 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28720 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28721 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_8
  simp only [← c28716, ← c28717, ← c28718] at e_445_8
  have e_445_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_9
  simp only [← c28719, ← c28720, k2, ← c28721] at e_445_9
  have hr := e_445_9
  simp only [e_445_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1260)) := by
  have c28722 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28723 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28724 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28725 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28726 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28727 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_10
  simp only [← c28722, ← c28723, ← c28724] at e_445_10
  have e_445_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_11
  simp only [← c28725, ← c28726, k2, ← c28727] at e_445_11
  have hr := e_445_11
  simp only [e_445_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1261)) := by
  have c28728 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28729 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28730 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28731 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28732 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28733 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_12
  simp only [← c28728, ← c28729, ← c28730] at e_445_12
  have e_445_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_13
  simp only [← c28731, ← c28732, k2, ← c28733] at e_445_13
  have hr := e_445_13
  simp only [e_445_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1262)) := by
  have c28734 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28735 := (publicBatchWrapper89_copies897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28736 := (publicBatchWrapper89_copies898 a h).1
  have c28737 := (publicBatchWrapper89_copies898 a h).2.1
  have c28738 := (publicBatchWrapper89_copies898 a h).2.2.1
  have c28739 := (publicBatchWrapper89_copies898 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_14
  simp only [← c28734, ← c28735, ← c28736] at e_445_14
  have e_445_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_15
  simp only [← c28737, ← c28738, k2, ← c28739] at e_445_15
  have hr := e_445_15
  simp only [e_445_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1263)) := by
  have c28740 := (publicBatchWrapper89_copies898 a h).2.2.2.2.1
  have c28741 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.1
  have c28742 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.1
  have c28743 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.1
  have c28744 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.1
  have c28745 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_16
  simp only [← c28740, ← c28741, ← c28742] at e_445_16
  have e_445_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_17
  simp only [← c28743, ← c28744, k2, ← c28745] at e_445_17
  have hr := e_445_17
  simp only [e_445_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 445 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1264)) := by
  have c28746 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28747 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28748 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28749 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28750 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28751 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_445_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_18
  simp only [← c28746, ← c28747, ← c28748] at e_445_18
  have e_445_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_445_19
  simp only [← c28749, ← c28750, k2, ← c28751] at e_445_19
  have hr := e_445_19
  simp only [e_445_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1265)) := by
  have c28752 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28753 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28754 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28755 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28756 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28757 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_0
  simp only [← c28752, ← c28753, ← c28754] at e_446_0
  have e_446_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_1
  simp only [← c28755, ← c28756, k2, ← c28757] at e_446_1
  have hr := e_446_1
  simp only [e_446_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1266)) := by
  have c28758 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28759 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28760 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28761 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28762 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28763 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_2
  simp only [← c28758, ← c28759, ← c28760] at e_446_2
  have e_446_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_3
  simp only [← c28761, ← c28762, k2, ← c28763] at e_446_3
  have hr := e_446_3
  simp only [e_446_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1267)) := by
  have c28764 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28765 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28766 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28767 := (publicBatchWrapper89_copies898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28768 := (publicBatchWrapper89_copies899 a h).1
  have c28769 := (publicBatchWrapper89_copies899 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_4
  simp only [← c28764, ← c28765, ← c28766] at e_446_4
  have e_446_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_5
  simp only [← c28767, ← c28768, k2, ← c28769] at e_446_5
  have hr := e_446_5
  simp only [e_446_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1268)) := by
  have c28770 := (publicBatchWrapper89_copies899 a h).2.2.1
  have c28771 := (publicBatchWrapper89_copies899 a h).2.2.2.1
  have c28772 := (publicBatchWrapper89_copies899 a h).2.2.2.2.1
  have c28773 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.1
  have c28774 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.1
  have c28775 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_6
  simp only [← c28770, ← c28771, ← c28772] at e_446_6
  have e_446_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_7
  simp only [← c28773, ← c28774, k2, ← c28775] at e_446_7
  have hr := e_446_7
  simp only [e_446_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1269)) := by
  have c28776 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.1
  have c28777 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.1
  have c28778 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28779 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28780 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28781 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_8
  simp only [← c28776, ← c28777, ← c28778] at e_446_8
  have e_446_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_9
  simp only [← c28779, ← c28780, k2, ← c28781] at e_446_9
  have hr := e_446_9
  simp only [e_446_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1270)) := by
  have c28782 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28783 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28784 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28785 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28786 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28787 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_10
  simp only [← c28782, ← c28783, ← c28784] at e_446_10
  have e_446_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_11
  simp only [← c28785, ← c28786, k2, ← c28787] at e_446_11
  have hr := e_446_11
  simp only [e_446_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1271)) := by
  have c28788 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28789 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28790 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28791 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28792 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28793 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_12
  simp only [← c28788, ← c28789, ← c28790] at e_446_12
  have e_446_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_13
  simp only [← c28791, ← c28792, k2, ← c28793] at e_446_13
  have hr := e_446_13
  simp only [e_446_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1272)) := by
  have c28794 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28795 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28796 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28797 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28798 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28799 := (publicBatchWrapper89_copies899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_14
  simp only [← c28794, ← c28795, ← c28796] at e_446_14
  have e_446_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_15
  simp only [← c28797, ← c28798, k2, ← c28799] at e_446_15
  have hr := e_446_15
  simp only [e_446_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1273)) := by
  have c28800 := (publicBatchWrapper89_copies900 a h).1
  have c28801 := (publicBatchWrapper89_copies900 a h).2.1
  have c28802 := (publicBatchWrapper89_copies900 a h).2.2.1
  have c28803 := (publicBatchWrapper89_copies900 a h).2.2.2.1
  have c28804 := (publicBatchWrapper89_copies900 a h).2.2.2.2.1
  have c28805 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_16
  simp only [← c28800, ← c28801, ← c28802] at e_446_16
  have e_446_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_17
  simp only [← c28803, ← c28804, k2, ← c28805] at e_446_17
  have hr := e_446_17
  simp only [e_446_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 446 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1274)) := by
  have c28806 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.1
  have c28807 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.1
  have c28808 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.1
  have c28809 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.1
  have c28810 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28811 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_446_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_18
  simp only [← c28806, ← c28807, ← c28808] at e_446_18
  have e_446_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_446_19
  simp only [← c28809, ← c28810, k2, ← c28811] at e_446_19
  have hr := e_446_19
  simp only [e_446_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1275)) := by
  have c28812 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28813 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28814 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28815 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28816 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28817 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_0
  simp only [← c28812, ← c28813, ← c28814] at e_447_0
  have e_447_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_1
  simp only [← c28815, ← c28816, k2, ← c28817] at e_447_1
  have hr := e_447_1
  simp only [e_447_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1276)) := by
  have c28818 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28819 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28820 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28821 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28822 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28823 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_2
  simp only [← c28818, ← c28819, ← c28820] at e_447_2
  have e_447_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_3
  simp only [← c28821, ← c28822, k2, ← c28823] at e_447_3
  have hr := e_447_3
  simp only [e_447_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1277)) := by
  have c28824 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28825 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28826 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28827 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28828 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28829 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_4
  simp only [← c28824, ← c28825, ← c28826] at e_447_4
  have e_447_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_5
  simp only [← c28827, ← c28828, k2, ← c28829] at e_447_5
  have hr := e_447_5
  simp only [e_447_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1278)) := by
  have c28830 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28831 := (publicBatchWrapper89_copies900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28832 := (publicBatchWrapper89_copies901 a h).1
  have c28833 := (publicBatchWrapper89_copies901 a h).2.1
  have c28834 := (publicBatchWrapper89_copies901 a h).2.2.1
  have c28835 := (publicBatchWrapper89_copies901 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_6
  simp only [← c28830, ← c28831, ← c28832] at e_447_6
  have e_447_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_7
  simp only [← c28833, ← c28834, k2, ← c28835] at e_447_7
  have hr := e_447_7
  simp only [e_447_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1279)) := by
  have c28836 := (publicBatchWrapper89_copies901 a h).2.2.2.2.1
  have c28837 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.1
  have c28838 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.1
  have c28839 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.1
  have c28840 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.1
  have c28841 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_8
  simp only [← c28836, ← c28837, ← c28838] at e_447_8
  have e_447_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_9
  simp only [← c28839, ← c28840, k2, ← c28841] at e_447_9
  have hr := e_447_9
  simp only [e_447_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1280)) := by
  have c28842 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28843 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28844 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28845 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28846 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28847 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_10
  simp only [← c28842, ← c28843, ← c28844] at e_447_10
  have e_447_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_11
  simp only [← c28845, ← c28846, k2, ← c28847] at e_447_11
  have hr := e_447_11
  simp only [e_447_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1281)) := by
  have c28848 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28849 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28850 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28851 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28852 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28853 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_12
  simp only [← c28848, ← c28849, ← c28850] at e_447_12
  have e_447_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_13
  simp only [← c28851, ← c28852, k2, ← c28853] at e_447_13
  have hr := e_447_13
  simp only [e_447_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1282)) := by
  have c28854 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28855 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28856 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28857 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28858 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28859 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_14
  simp only [← c28854, ← c28855, ← c28856] at e_447_14
  have e_447_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_15
  simp only [← c28857, ← c28858, k2, ← c28859] at e_447_15
  have hr := e_447_15
  simp only [e_447_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1283)) := by
  have c28860 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28861 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28862 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28863 := (publicBatchWrapper89_copies901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28864 := (publicBatchWrapper89_copies902 a h).1
  have c28865 := (publicBatchWrapper89_copies902 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_16
  simp only [← c28860, ← c28861, ← c28862] at e_447_16
  have e_447_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_17
  simp only [← c28863, ← c28864, k2, ← c28865] at e_447_17
  have hr := e_447_17
  simp only [e_447_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 447 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1284)) := by
  have c28866 := (publicBatchWrapper89_copies902 a h).2.2.1
  have c28867 := (publicBatchWrapper89_copies902 a h).2.2.2.1
  have c28868 := (publicBatchWrapper89_copies902 a h).2.2.2.2.1
  have c28869 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.1
  have c28870 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.1
  have c28871 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_447_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_18
  simp only [← c28866, ← c28867, ← c28868] at e_447_18
  have e_447_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_447_19
  simp only [← c28869, ← c28870, k2, ← c28871] at e_447_19
  have hr := e_447_19
  simp only [e_447_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1285)) := by
  have c28872 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.1
  have c28873 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.1
  have c28874 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28875 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28876 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28877 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_0
  simp only [← c28872, ← c28873, ← c28874] at e_448_0
  have e_448_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_1
  simp only [← c28875, ← c28876, k2, ← c28877] at e_448_1
  have hr := e_448_1
  simp only [e_448_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1286)) := by
  have c28878 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28879 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28880 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28881 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28882 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28883 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_2
  simp only [← c28878, ← c28879, ← c28880] at e_448_2
  have e_448_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_3
  simp only [← c28881, ← c28882, k2, ← c28883] at e_448_3
  have hr := e_448_3
  simp only [e_448_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1287)) := by
  have c28884 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28885 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28886 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28887 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28888 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28889 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_4
  simp only [← c28884, ← c28885, ← c28886] at e_448_4
  have e_448_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_5
  simp only [← c28887, ← c28888, k2, ← c28889] at e_448_5
  have hr := e_448_5
  simp only [e_448_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1288)) := by
  have c28890 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28891 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28892 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28893 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28894 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28895 := (publicBatchWrapper89_copies902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_6
  simp only [← c28890, ← c28891, ← c28892] at e_448_6
  have e_448_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_7
  simp only [← c28893, ← c28894, k2, ← c28895] at e_448_7
  have hr := e_448_7
  simp only [e_448_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1289)) := by
  have c28896 := (publicBatchWrapper89_copies903 a h).1
  have c28897 := (publicBatchWrapper89_copies903 a h).2.1
  have c28898 := (publicBatchWrapper89_copies903 a h).2.2.1
  have c28899 := (publicBatchWrapper89_copies903 a h).2.2.2.1
  have c28900 := (publicBatchWrapper89_copies903 a h).2.2.2.2.1
  have c28901 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_8
  simp only [← c28896, ← c28897, ← c28898] at e_448_8
  have e_448_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_9
  simp only [← c28899, ← c28900, k2, ← c28901] at e_448_9
  have hr := e_448_9
  simp only [e_448_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1290)) := by
  have c28902 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.1
  have c28903 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.1
  have c28904 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.1
  have c28905 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.1
  have c28906 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28907 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_10
  simp only [← c28902, ← c28903, ← c28904] at e_448_10
  have e_448_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_11
  simp only [← c28905, ← c28906, k2, ← c28907] at e_448_11
  have hr := e_448_11
  simp only [e_448_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1291)) := by
  have c28908 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28909 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28910 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28911 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28912 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28913 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_12
  simp only [← c28908, ← c28909, ← c28910] at e_448_12
  have e_448_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_13
  simp only [← c28911, ← c28912, k2, ← c28913] at e_448_13
  have hr := e_448_13
  simp only [e_448_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1292)) := by
  have c28914 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28915 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28916 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28917 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28918 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28919 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_14
  simp only [← c28914, ← c28915, ← c28916] at e_448_14
  have e_448_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_15
  simp only [← c28917, ← c28918, k2, ← c28919] at e_448_15
  have hr := e_448_15
  simp only [e_448_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1293)) := by
  have c28920 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28921 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28922 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28923 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28924 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28925 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_16
  simp only [← c28920, ← c28921, ← c28922] at e_448_16
  have e_448_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_17
  simp only [← c28923, ← c28924, k2, ← c28925] at e_448_17
  have hr := e_448_17
  simp only [e_448_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 448 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1294)) := by
  have c28926 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28927 := (publicBatchWrapper89_copies903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28928 := (publicBatchWrapper89_copies904 a h).1
  have c28929 := (publicBatchWrapper89_copies904 a h).2.1
  have c28930 := (publicBatchWrapper89_copies904 a h).2.2.1
  have c28931 := (publicBatchWrapper89_copies904 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_448_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_18
  simp only [← c28926, ← c28927, ← c28928] at e_448_18
  have e_448_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_448_19
  simp only [← c28929, ← c28930, k2, ← c28931] at e_448_19
  have hr := e_448_19
  simp only [e_448_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1295)) := by
  have c28932 := (publicBatchWrapper89_copies904 a h).2.2.2.2.1
  have c28933 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.1
  have c28934 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.1
  have c28935 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.1
  have c28936 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.1
  have c28937 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_0
  simp only [← c28932, ← c28933, ← c28934] at e_449_0
  have e_449_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_1
  simp only [← c28935, ← c28936, k2, ← c28937] at e_449_1
  have hr := e_449_1
  simp only [e_449_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1296)) := by
  have c28938 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28939 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28940 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28941 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28942 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28943 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_2
  simp only [← c28938, ← c28939, ← c28940] at e_449_2
  have e_449_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_3
  simp only [← c28941, ← c28942, k2, ← c28943] at e_449_3
  have hr := e_449_3
  simp only [e_449_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1297)) := by
  have c28944 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28945 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28946 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28947 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28948 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28949 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_4
  simp only [← c28944, ← c28945, ← c28946] at e_449_4
  have e_449_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_5
  simp only [← c28947, ← c28948, k2, ← c28949] at e_449_5
  have hr := e_449_5
  simp only [e_449_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1298)) := by
  have c28950 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28951 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28952 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28953 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28954 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28955 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_6
  simp only [← c28950, ← c28951, ← c28952] at e_449_6
  have e_449_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_7
  simp only [← c28953, ← c28954, k2, ← c28955] at e_449_7
  have hr := e_449_7
  simp only [e_449_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1299)) := by
  have c28956 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28957 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28958 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28959 := (publicBatchWrapper89_copies904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c28960 := (publicBatchWrapper89_copies905 a h).1
  have c28961 := (publicBatchWrapper89_copies905 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_8
  simp only [← c28956, ← c28957, ← c28958] at e_449_8
  have e_449_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_9
  simp only [← c28959, ← c28960, k2, ← c28961] at e_449_9
  have hr := e_449_9
  simp only [e_449_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1300)) := by
  have c28962 := (publicBatchWrapper89_copies905 a h).2.2.1
  have c28963 := (publicBatchWrapper89_copies905 a h).2.2.2.1
  have c28964 := (publicBatchWrapper89_copies905 a h).2.2.2.2.1
  have c28965 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.1
  have c28966 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.1
  have c28967 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_10
  simp only [← c28962, ← c28963, ← c28964] at e_449_10
  have e_449_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_11
  simp only [← c28965, ← c28966, k2, ← c28967] at e_449_11
  have hr := e_449_11
  simp only [e_449_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1301)) := by
  have c28968 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.1
  have c28969 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.1
  have c28970 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c28971 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c28972 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28973 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_12
  simp only [← c28968, ← c28969, ← c28970] at e_449_12
  have e_449_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_13
  simp only [← c28971, ← c28972, k2, ← c28973] at e_449_13
  have hr := e_449_13
  simp only [e_449_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1302)) := by
  have c28974 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28975 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28976 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28977 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28978 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28979 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_14
  simp only [← c28974, ← c28975, ← c28976] at e_449_14
  have e_449_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_15
  simp only [← c28977, ← c28978, k2, ← c28979] at e_449_15
  have hr := e_449_15
  simp only [e_449_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1303)) := by
  have c28980 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28981 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28982 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28983 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28984 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28985 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_16
  simp only [← c28980, ← c28981, ← c28982] at e_449_16
  have e_449_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_17
  simp only [← c28983, ← c28984, k2, ← c28985] at e_449_17
  have hr := e_449_17
  simp only [e_449_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 449 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1304)) := by
  have c28986 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28987 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28988 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28989 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28990 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c28991 := (publicBatchWrapper89_copies905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_449_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_18
  simp only [← c28986, ← c28987, ← c28988] at e_449_18
  have e_449_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_449_19
  simp only [← c28989, ← c28990, k2, ← c28991] at e_449_19
  have hr := e_449_19
  simp only [e_449_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1305)) := by
  have c28992 := (publicBatchWrapper89_copies906 a h).1
  have c28993 := (publicBatchWrapper89_copies906 a h).2.1
  have c28994 := (publicBatchWrapper89_copies906 a h).2.2.1
  have c28995 := (publicBatchWrapper89_copies906 a h).2.2.2.1
  have c28996 := (publicBatchWrapper89_copies906 a h).2.2.2.2.1
  have c28997 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_0
  simp only [← c28992, ← c28993, ← c28994] at e_450_0
  have e_450_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_1
  simp only [← c28995, ← c28996, k2, ← c28997] at e_450_1
  have hr := e_450_1
  simp only [e_450_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1306)) := by
  have c28998 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.1
  have c28999 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.1
  have c29000 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.1
  have c29001 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.1
  have c29002 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29003 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_2
  simp only [← c28998, ← c28999, ← c29000] at e_450_2
  have e_450_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_3
  simp only [← c29001, ← c29002, k2, ← c29003] at e_450_3
  have hr := e_450_3
  simp only [e_450_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1307)) := by
  have c29004 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29005 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29006 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29007 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29008 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29009 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_4
  simp only [← c29004, ← c29005, ← c29006] at e_450_4
  have e_450_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_5
  simp only [← c29007, ← c29008, k2, ← c29009] at e_450_5
  have hr := e_450_5
  simp only [e_450_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1308)) := by
  have c29010 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29011 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29012 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29013 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29014 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29015 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_6
  simp only [← c29010, ← c29011, ← c29012] at e_450_6
  have e_450_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_7
  simp only [← c29013, ← c29014, k2, ← c29015] at e_450_7
  have hr := e_450_7
  simp only [e_450_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1309)) := by
  have c29016 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29017 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29018 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29019 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29020 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29021 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_8
  simp only [← c29016, ← c29017, ← c29018] at e_450_8
  have e_450_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_9
  simp only [← c29019, ← c29020, k2, ← c29021] at e_450_9
  have hr := e_450_9
  simp only [e_450_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1310)) := by
  have c29022 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29023 := (publicBatchWrapper89_copies906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29024 := (publicBatchWrapper89_copies907 a h).1
  have c29025 := (publicBatchWrapper89_copies907 a h).2.1
  have c29026 := (publicBatchWrapper89_copies907 a h).2.2.1
  have c29027 := (publicBatchWrapper89_copies907 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_10
  simp only [← c29022, ← c29023, ← c29024] at e_450_10
  have e_450_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_11
  simp only [← c29025, ← c29026, k2, ← c29027] at e_450_11
  have hr := e_450_11
  simp only [e_450_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1311)) := by
  have c29028 := (publicBatchWrapper89_copies907 a h).2.2.2.2.1
  have c29029 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.1
  have c29030 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.1
  have c29031 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.1
  have c29032 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.1
  have c29033 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_12
  simp only [← c29028, ← c29029, ← c29030] at e_450_12
  have e_450_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_13
  simp only [← c29031, ← c29032, k2, ← c29033] at e_450_13
  have hr := e_450_13
  simp only [e_450_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1312)) := by
  have c29034 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29035 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29036 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29037 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29038 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29039 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_14
  simp only [← c29034, ← c29035, ← c29036] at e_450_14
  have e_450_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_15
  simp only [← c29037, ← c29038, k2, ← c29039] at e_450_15
  have hr := e_450_15
  simp only [e_450_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1313)) := by
  have c29040 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29041 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29042 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29043 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29044 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29045 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_16
  simp only [← c29040, ← c29041, ← c29042] at e_450_16
  have e_450_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_17
  simp only [← c29043, ← c29044, k2, ← c29045] at e_450_17
  have hr := e_450_17
  simp only [e_450_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 450 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1314)) := by
  have c29046 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29047 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29048 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29049 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29050 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29051 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_450_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_18
  simp only [← c29046, ← c29047, ← c29048] at e_450_18
  have e_450_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_450_19
  simp only [← c29049, ← c29050, k2, ← c29051] at e_450_19
  have hr := e_450_19
  simp only [e_450_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1315)) := by
  have c29052 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29053 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29054 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29055 := (publicBatchWrapper89_copies907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29056 := (publicBatchWrapper89_copies908 a h).1
  have c29057 := (publicBatchWrapper89_copies908 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_0
  simp only [← c29052, ← c29053, ← c29054] at e_451_0
  have e_451_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_1
  simp only [← c29055, ← c29056, k2, ← c29057] at e_451_1
  have hr := e_451_1
  simp only [e_451_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1316)) := by
  have c29058 := (publicBatchWrapper89_copies908 a h).2.2.1
  have c29059 := (publicBatchWrapper89_copies908 a h).2.2.2.1
  have c29060 := (publicBatchWrapper89_copies908 a h).2.2.2.2.1
  have c29061 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.1
  have c29062 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.1
  have c29063 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_2
  simp only [← c29058, ← c29059, ← c29060] at e_451_2
  have e_451_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_3
  simp only [← c29061, ← c29062, k2, ← c29063] at e_451_3
  have hr := e_451_3
  simp only [e_451_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1317)) := by
  have c29064 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.1
  have c29065 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.1
  have c29066 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29067 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29068 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29069 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_4
  simp only [← c29064, ← c29065, ← c29066] at e_451_4
  have e_451_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_5
  simp only [← c29067, ← c29068, k2, ← c29069] at e_451_5
  have hr := e_451_5
  simp only [e_451_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1318)) := by
  have c29070 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29071 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29072 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29073 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29074 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29075 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_6
  simp only [← c29070, ← c29071, ← c29072] at e_451_6
  have e_451_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_7
  simp only [← c29073, ← c29074, k2, ← c29075] at e_451_7
  have hr := e_451_7
  simp only [e_451_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1319)) := by
  have c29076 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29077 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29078 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29079 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29080 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29081 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_8
  simp only [← c29076, ← c29077, ← c29078] at e_451_8
  have e_451_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_9
  simp only [← c29079, ← c29080, k2, ← c29081] at e_451_9
  have hr := e_451_9
  simp only [e_451_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1320)) := by
  have c29082 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29083 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29084 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29085 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29086 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29087 := (publicBatchWrapper89_copies908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_10
  simp only [← c29082, ← c29083, ← c29084] at e_451_10
  have e_451_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_11
  simp only [← c29085, ← c29086, k2, ← c29087] at e_451_11
  have hr := e_451_11
  simp only [e_451_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1321)) := by
  have c29088 := (publicBatchWrapper89_copies909 a h).1
  have c29089 := (publicBatchWrapper89_copies909 a h).2.1
  have c29090 := (publicBatchWrapper89_copies909 a h).2.2.1
  have c29091 := (publicBatchWrapper89_copies909 a h).2.2.2.1
  have c29092 := (publicBatchWrapper89_copies909 a h).2.2.2.2.1
  have c29093 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_12
  simp only [← c29088, ← c29089, ← c29090] at e_451_12
  have e_451_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_13
  simp only [← c29091, ← c29092, k2, ← c29093] at e_451_13
  have hr := e_451_13
  simp only [e_451_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1322)) := by
  have c29094 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.1
  have c29095 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.1
  have c29096 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.1
  have c29097 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.1
  have c29098 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29099 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_14
  simp only [← c29094, ← c29095, ← c29096] at e_451_14
  have e_451_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_15
  simp only [← c29097, ← c29098, k2, ← c29099] at e_451_15
  have hr := e_451_15
  simp only [e_451_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1323)) := by
  have c29100 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29101 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29102 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29103 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29104 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29105 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_16
  simp only [← c29100, ← c29101, ← c29102] at e_451_16
  have e_451_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_17
  simp only [← c29103, ← c29104, k2, ← c29105] at e_451_17
  have hr := e_451_17
  simp only [e_451_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 451 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1324)) := by
  have c29106 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29107 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29108 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29109 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29110 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29111 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_451_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_18
  simp only [← c29106, ← c29107, ← c29108] at e_451_18
  have e_451_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_451_19
  simp only [← c29109, ← c29110, k2, ← c29111] at e_451_19
  have hr := e_451_19
  simp only [e_451_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1325)) := by
  have c29112 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29113 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29114 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29115 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29116 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29117 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_0
  simp only [← c29112, ← c29113, ← c29114] at e_452_0
  have e_452_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_1
  simp only [← c29115, ← c29116, k2, ← c29117] at e_452_1
  have hr := e_452_1
  simp only [e_452_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1326)) := by
  have c29118 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29119 := (publicBatchWrapper89_copies909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29120 := (publicBatchWrapper89_copies910 a h).1
  have c29121 := (publicBatchWrapper89_copies910 a h).2.1
  have c29122 := (publicBatchWrapper89_copies910 a h).2.2.1
  have c29123 := (publicBatchWrapper89_copies910 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_2
  simp only [← c29118, ← c29119, ← c29120] at e_452_2
  have e_452_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_3
  simp only [← c29121, ← c29122, k2, ← c29123] at e_452_3
  have hr := e_452_3
  simp only [e_452_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1327)) := by
  have c29124 := (publicBatchWrapper89_copies910 a h).2.2.2.2.1
  have c29125 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.1
  have c29126 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.1
  have c29127 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.1
  have c29128 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.1
  have c29129 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_4
  simp only [← c29124, ← c29125, ← c29126] at e_452_4
  have e_452_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_5
  simp only [← c29127, ← c29128, k2, ← c29129] at e_452_5
  have hr := e_452_5
  simp only [e_452_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1328)) := by
  have c29130 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29131 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29132 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29133 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29134 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29135 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_6
  simp only [← c29130, ← c29131, ← c29132] at e_452_6
  have e_452_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_7
  simp only [← c29133, ← c29134, k2, ← c29135] at e_452_7
  have hr := e_452_7
  simp only [e_452_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1329)) := by
  have c29136 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29137 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29138 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29139 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29140 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29141 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_8
  simp only [← c29136, ← c29137, ← c29138] at e_452_8
  have e_452_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_9
  simp only [← c29139, ← c29140, k2, ← c29141] at e_452_9
  have hr := e_452_9
  simp only [e_452_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1330)) := by
  have c29142 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29143 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29144 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29145 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29146 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29147 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_10
  simp only [← c29142, ← c29143, ← c29144] at e_452_10
  have e_452_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_11
  simp only [← c29145, ← c29146, k2, ← c29147] at e_452_11
  have hr := e_452_11
  simp only [e_452_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1331)) := by
  have c29148 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29149 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29150 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29151 := (publicBatchWrapper89_copies910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29152 := (publicBatchWrapper89_copies911 a h).1
  have c29153 := (publicBatchWrapper89_copies911 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_12
  simp only [← c29148, ← c29149, ← c29150] at e_452_12
  have e_452_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_13
  simp only [← c29151, ← c29152, k2, ← c29153] at e_452_13
  have hr := e_452_13
  simp only [e_452_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1332)) := by
  have c29154 := (publicBatchWrapper89_copies911 a h).2.2.1
  have c29155 := (publicBatchWrapper89_copies911 a h).2.2.2.1
  have c29156 := (publicBatchWrapper89_copies911 a h).2.2.2.2.1
  have c29157 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.1
  have c29158 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.1
  have c29159 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_14
  simp only [← c29154, ← c29155, ← c29156] at e_452_14
  have e_452_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_15
  simp only [← c29157, ← c29158, k2, ← c29159] at e_452_15
  have hr := e_452_15
  simp only [e_452_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1333)) := by
  have c29160 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.1
  have c29161 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.1
  have c29162 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29163 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29164 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29165 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_16
  simp only [← c29160, ← c29161, ← c29162] at e_452_16
  have e_452_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_17
  simp only [← c29163, ← c29164, k2, ← c29165] at e_452_17
  have hr := e_452_17
  simp only [e_452_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 452 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1334)) := by
  have c29166 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29167 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29168 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29169 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29170 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29171 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_452_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_18
  simp only [← c29166, ← c29167, ← c29168] at e_452_18
  have e_452_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_452_19
  simp only [← c29169, ← c29170, k2, ← c29171] at e_452_19
  have hr := e_452_19
  simp only [e_452_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1335)) := by
  have c29172 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29173 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29174 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29175 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29176 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29177 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_0
  simp only [← c29172, ← c29173, ← c29174] at e_453_0
  have e_453_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_1
  simp only [← c29175, ← c29176, k2, ← c29177] at e_453_1
  have hr := e_453_1
  simp only [e_453_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1336)) := by
  have c29178 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29179 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29180 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29181 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29182 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29183 := (publicBatchWrapper89_copies911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_2
  simp only [← c29178, ← c29179, ← c29180] at e_453_2
  have e_453_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_3
  simp only [← c29181, ← c29182, k2, ← c29183] at e_453_3
  have hr := e_453_3
  simp only [e_453_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1337)) := by
  have c29184 := (publicBatchWrapper89_copies912 a h).1
  have c29185 := (publicBatchWrapper89_copies912 a h).2.1
  have c29186 := (publicBatchWrapper89_copies912 a h).2.2.1
  have c29187 := (publicBatchWrapper89_copies912 a h).2.2.2.1
  have c29188 := (publicBatchWrapper89_copies912 a h).2.2.2.2.1
  have c29189 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_4
  simp only [← c29184, ← c29185, ← c29186] at e_453_4
  have e_453_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_5
  simp only [← c29187, ← c29188, k2, ← c29189] at e_453_5
  have hr := e_453_5
  simp only [e_453_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1338)) := by
  have c29190 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.1
  have c29191 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.1
  have c29192 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.1
  have c29193 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.1
  have c29194 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29195 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_6
  simp only [← c29190, ← c29191, ← c29192] at e_453_6
  have e_453_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_7
  simp only [← c29193, ← c29194, k2, ← c29195] at e_453_7
  have hr := e_453_7
  simp only [e_453_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1339)) := by
  have c29196 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29197 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29198 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29199 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29200 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29201 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_8
  simp only [← c29196, ← c29197, ← c29198] at e_453_8
  have e_453_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_9
  simp only [← c29199, ← c29200, k2, ← c29201] at e_453_9
  have hr := e_453_9
  simp only [e_453_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1340)) := by
  have c29202 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29203 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29204 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29205 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29206 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29207 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_10
  simp only [← c29202, ← c29203, ← c29204] at e_453_10
  have e_453_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_11
  simp only [← c29205, ← c29206, k2, ← c29207] at e_453_11
  have hr := e_453_11
  simp only [e_453_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1341)) := by
  have c29208 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29209 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29210 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29211 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29212 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29213 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_12
  simp only [← c29208, ← c29209, ← c29210] at e_453_12
  have e_453_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_13
  simp only [← c29211, ← c29212, k2, ← c29213] at e_453_13
  have hr := e_453_13
  simp only [e_453_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1342)) := by
  have c29214 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29215 := (publicBatchWrapper89_copies912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29216 := (publicBatchWrapper89_copies913 a h).1
  have c29217 := (publicBatchWrapper89_copies913 a h).2.1
  have c29218 := (publicBatchWrapper89_copies913 a h).2.2.1
  have c29219 := (publicBatchWrapper89_copies913 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_14
  simp only [← c29214, ← c29215, ← c29216] at e_453_14
  have e_453_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_15
  simp only [← c29217, ← c29218, k2, ← c29219] at e_453_15
  have hr := e_453_15
  simp only [e_453_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1343)) := by
  have c29220 := (publicBatchWrapper89_copies913 a h).2.2.2.2.1
  have c29221 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.1
  have c29222 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.1
  have c29223 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.1
  have c29224 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.1
  have c29225 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_16
  simp only [← c29220, ← c29221, ← c29222] at e_453_16
  have e_453_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_17
  simp only [← c29223, ← c29224, k2, ← c29225] at e_453_17
  have hr := e_453_17
  simp only [e_453_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 453 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1344)) := by
  have c29226 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29227 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29228 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29229 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29230 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29231 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_453_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_18
  simp only [← c29226, ← c29227, ← c29228] at e_453_18
  have e_453_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_453_19
  simp only [← c29229, ← c29230, k2, ← c29231] at e_453_19
  have hr := e_453_19
  simp only [e_453_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1345)) := by
  have c29232 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29233 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29234 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29235 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29236 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29237 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_0
  simp only [← c29232, ← c29233, ← c29234] at e_454_0
  have e_454_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_1
  simp only [← c29235, ← c29236, k2, ← c29237] at e_454_1
  have hr := e_454_1
  simp only [e_454_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1346)) := by
  have c29238 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29239 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29240 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29241 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29242 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29243 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_2
  simp only [← c29238, ← c29239, ← c29240] at e_454_2
  have e_454_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_3
  simp only [← c29241, ← c29242, k2, ← c29243] at e_454_3
  have hr := e_454_3
  simp only [e_454_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1347)) := by
  have c29244 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29245 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29246 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29247 := (publicBatchWrapper89_copies913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29248 := (publicBatchWrapper89_copies914 a h).1
  have c29249 := (publicBatchWrapper89_copies914 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_4
  simp only [← c29244, ← c29245, ← c29246] at e_454_4
  have e_454_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_5
  simp only [← c29247, ← c29248, k2, ← c29249] at e_454_5
  have hr := e_454_5
  simp only [e_454_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1348)) := by
  have c29250 := (publicBatchWrapper89_copies914 a h).2.2.1
  have c29251 := (publicBatchWrapper89_copies914 a h).2.2.2.1
  have c29252 := (publicBatchWrapper89_copies914 a h).2.2.2.2.1
  have c29253 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.1
  have c29254 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.1
  have c29255 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_6
  simp only [← c29250, ← c29251, ← c29252] at e_454_6
  have e_454_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_7
  simp only [← c29253, ← c29254, k2, ← c29255] at e_454_7
  have hr := e_454_7
  simp only [e_454_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1349)) := by
  have c29256 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.1
  have c29257 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.1
  have c29258 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29259 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29260 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29261 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_8
  simp only [← c29256, ← c29257, ← c29258] at e_454_8
  have e_454_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_9
  simp only [← c29259, ← c29260, k2, ← c29261] at e_454_9
  have hr := e_454_9
  simp only [e_454_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1350)) := by
  have c29262 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29263 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29264 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29265 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29266 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29267 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_10
  simp only [← c29262, ← c29263, ← c29264] at e_454_10
  have e_454_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_11
  simp only [← c29265, ← c29266, k2, ← c29267] at e_454_11
  have hr := e_454_11
  simp only [e_454_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1351)) := by
  have c29268 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29269 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29270 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29271 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29272 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29273 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_12
  simp only [← c29268, ← c29269, ← c29270] at e_454_12
  have e_454_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_13
  simp only [← c29271, ← c29272, k2, ← c29273] at e_454_13
  have hr := e_454_13
  simp only [e_454_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1352)) := by
  have c29274 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29275 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29276 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29277 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29278 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29279 := (publicBatchWrapper89_copies914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_14
  simp only [← c29274, ← c29275, ← c29276] at e_454_14
  have e_454_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_15
  simp only [← c29277, ← c29278, k2, ← c29279] at e_454_15
  have hr := e_454_15
  simp only [e_454_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1353)) := by
  have c29280 := (publicBatchWrapper89_copies915 a h).1
  have c29281 := (publicBatchWrapper89_copies915 a h).2.1
  have c29282 := (publicBatchWrapper89_copies915 a h).2.2.1
  have c29283 := (publicBatchWrapper89_copies915 a h).2.2.2.1
  have c29284 := (publicBatchWrapper89_copies915 a h).2.2.2.2.1
  have c29285 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_16
  simp only [← c29280, ← c29281, ← c29282] at e_454_16
  have e_454_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_17
  simp only [← c29283, ← c29284, k2, ← c29285] at e_454_17
  have hr := e_454_17
  simp only [e_454_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 454 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1354)) := by
  have c29286 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.1
  have c29287 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.1
  have c29288 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.1
  have c29289 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.1
  have c29290 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29291 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_454_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_18
  simp only [← c29286, ← c29287, ← c29288] at e_454_18
  have e_454_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_454_19
  simp only [← c29289, ← c29290, k2, ← c29291] at e_454_19
  have hr := e_454_19
  simp only [e_454_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1355)) := by
  have c29292 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29293 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29294 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29295 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29296 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29297 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_0
  simp only [← c29292, ← c29293, ← c29294] at e_455_0
  have e_455_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_1
  simp only [← c29295, ← c29296, k2, ← c29297] at e_455_1
  have hr := e_455_1
  simp only [e_455_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1356)) := by
  have c29298 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29299 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29300 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29301 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29302 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29303 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_2
  simp only [← c29298, ← c29299, ← c29300] at e_455_2
  have e_455_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_3
  simp only [← c29301, ← c29302, k2, ← c29303] at e_455_3
  have hr := e_455_3
  simp only [e_455_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1357)) := by
  have c29304 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29305 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29306 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29307 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29308 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29309 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_4
  simp only [← c29304, ← c29305, ← c29306] at e_455_4
  have e_455_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_5
  simp only [← c29307, ← c29308, k2, ← c29309] at e_455_5
  have hr := e_455_5
  simp only [e_455_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1358)) := by
  have c29310 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29311 := (publicBatchWrapper89_copies915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29312 := (publicBatchWrapper89_copies916 a h).1
  have c29313 := (publicBatchWrapper89_copies916 a h).2.1
  have c29314 := (publicBatchWrapper89_copies916 a h).2.2.1
  have c29315 := (publicBatchWrapper89_copies916 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_6
  simp only [← c29310, ← c29311, ← c29312] at e_455_6
  have e_455_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_7
  simp only [← c29313, ← c29314, k2, ← c29315] at e_455_7
  have hr := e_455_7
  simp only [e_455_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1359)) := by
  have c29316 := (publicBatchWrapper89_copies916 a h).2.2.2.2.1
  have c29317 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.1
  have c29318 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.1
  have c29319 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.1
  have c29320 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.1
  have c29321 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_8
  simp only [← c29316, ← c29317, ← c29318] at e_455_8
  have e_455_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_9
  simp only [← c29319, ← c29320, k2, ← c29321] at e_455_9
  have hr := e_455_9
  simp only [e_455_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1360)) := by
  have c29322 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29323 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29324 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29325 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29326 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29327 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_10
  simp only [← c29322, ← c29323, ← c29324] at e_455_10
  have e_455_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_11
  simp only [← c29325, ← c29326, k2, ← c29327] at e_455_11
  have hr := e_455_11
  simp only [e_455_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1361)) := by
  have c29328 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29329 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29330 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29331 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29332 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29333 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_12
  simp only [← c29328, ← c29329, ← c29330] at e_455_12
  have e_455_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_13
  simp only [← c29331, ← c29332, k2, ← c29333] at e_455_13
  have hr := e_455_13
  simp only [e_455_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1362)) := by
  have c29334 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29335 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29336 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29337 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29338 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29339 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_14
  simp only [← c29334, ← c29335, ← c29336] at e_455_14
  have e_455_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_15
  simp only [← c29337, ← c29338, k2, ← c29339] at e_455_15
  have hr := e_455_15
  simp only [e_455_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1363)) := by
  have c29340 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29341 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29342 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29343 := (publicBatchWrapper89_copies916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29344 := (publicBatchWrapper89_copies917 a h).1
  have c29345 := (publicBatchWrapper89_copies917 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_16
  simp only [← c29340, ← c29341, ← c29342] at e_455_16
  have e_455_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_17
  simp only [← c29343, ← c29344, k2, ← c29345] at e_455_17
  have hr := e_455_17
  simp only [e_455_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 455 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1364)) := by
  have c29346 := (publicBatchWrapper89_copies917 a h).2.2.1
  have c29347 := (publicBatchWrapper89_copies917 a h).2.2.2.1
  have c29348 := (publicBatchWrapper89_copies917 a h).2.2.2.2.1
  have c29349 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.1
  have c29350 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.1
  have c29351 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_455_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_18
  simp only [← c29346, ← c29347, ← c29348] at e_455_18
  have e_455_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_455_19
  simp only [← c29349, ← c29350, k2, ← c29351] at e_455_19
  have hr := e_455_19
  simp only [e_455_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1365)) := by
  have c29352 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.1
  have c29353 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.1
  have c29354 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29355 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29356 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29357 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_0
  simp only [← c29352, ← c29353, ← c29354] at e_456_0
  have e_456_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_1
  simp only [← c29355, ← c29356, k2, ← c29357] at e_456_1
  have hr := e_456_1
  simp only [e_456_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1366)) := by
  have c29358 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29359 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29360 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29361 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29362 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29363 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_2
  simp only [← c29358, ← c29359, ← c29360] at e_456_2
  have e_456_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_3
  simp only [← c29361, ← c29362, k2, ← c29363] at e_456_3
  have hr := e_456_3
  simp only [e_456_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1367)) := by
  have c29364 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29365 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29366 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29367 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29368 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29369 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_4
  simp only [← c29364, ← c29365, ← c29366] at e_456_4
  have e_456_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_5
  simp only [← c29367, ← c29368, k2, ← c29369] at e_456_5
  have hr := e_456_5
  simp only [e_456_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1368)) := by
  have c29370 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29371 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29372 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29373 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29374 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29375 := (publicBatchWrapper89_copies917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_6
  simp only [← c29370, ← c29371, ← c29372] at e_456_6
  have e_456_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_7
  simp only [← c29373, ← c29374, k2, ← c29375] at e_456_7
  have hr := e_456_7
  simp only [e_456_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1369)) := by
  have c29376 := (publicBatchWrapper89_copies918 a h).1
  have c29377 := (publicBatchWrapper89_copies918 a h).2.1
  have c29378 := (publicBatchWrapper89_copies918 a h).2.2.1
  have c29379 := (publicBatchWrapper89_copies918 a h).2.2.2.1
  have c29380 := (publicBatchWrapper89_copies918 a h).2.2.2.2.1
  have c29381 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_8
  simp only [← c29376, ← c29377, ← c29378] at e_456_8
  have e_456_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_9
  simp only [← c29379, ← c29380, k2, ← c29381] at e_456_9
  have hr := e_456_9
  simp only [e_456_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1410)) := by
  have c29382 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.1
  have c29383 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.1
  have c29384 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.1
  have c29385 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.1
  have c29386 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29387 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_10
  simp only [← c29382, ← c29383, ← c29384] at e_456_10
  have e_456_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_11
  simp only [← c29385, ← c29386, k2, ← c29387] at e_456_11
  have hr := e_456_11
  simp only [e_456_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1411)) := by
  have c29388 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29389 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29390 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29391 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29392 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29393 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_12
  simp only [← c29388, ← c29389, ← c29390] at e_456_12
  have e_456_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_13
  simp only [← c29391, ← c29392, k2, ← c29393] at e_456_13
  have hr := e_456_13
  simp only [e_456_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1412)) := by
  have c29394 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29395 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29396 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29397 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29398 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29399 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_14
  simp only [← c29394, ← c29395, ← c29396] at e_456_14
  have e_456_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_15
  simp only [← c29397, ← c29398, k2, ← c29399] at e_456_15
  have hr := e_456_15
  simp only [e_456_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1413)) := by
  have c29400 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29401 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29402 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29403 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29404 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29405 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_16
  simp only [← c29400, ← c29401, ← c29402] at e_456_16
  have e_456_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_17
  simp only [← c29403, ← c29404, k2, ← c29405] at e_456_17
  have hr := e_456_17
  simp only [e_456_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 456 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1414)) := by
  have c29406 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29407 := (publicBatchWrapper89_copies918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29408 := (publicBatchWrapper89_copies919 a h).1
  have c29409 := (publicBatchWrapper89_copies919 a h).2.1
  have c29410 := (publicBatchWrapper89_copies919 a h).2.2.1
  have c29411 := (publicBatchWrapper89_copies919 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_456_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_18
  simp only [← c29406, ← c29407, ← c29408] at e_456_18
  have e_456_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_456_19
  simp only [← c29409, ← c29410, k2, ← c29411] at e_456_19
  have hr := e_456_19
  simp only [e_456_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1415)) := by
  have c29412 := (publicBatchWrapper89_copies919 a h).2.2.2.2.1
  have c29413 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.1
  have c29414 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.1
  have c29415 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.1
  have c29416 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.1
  have c29417 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_0
  simp only [← c29412, ← c29413, ← c29414] at e_457_0
  have e_457_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_1
  simp only [← c29415, ← c29416, k2, ← c29417] at e_457_1
  have hr := e_457_1
  simp only [e_457_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1416)) := by
  have c29418 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29419 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29420 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29421 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29422 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29423 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_2
  simp only [← c29418, ← c29419, ← c29420] at e_457_2
  have e_457_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_3
  simp only [← c29421, ← c29422, k2, ← c29423] at e_457_3
  have hr := e_457_3
  simp only [e_457_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1417)) := by
  have c29424 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29425 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29426 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29427 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29428 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29429 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_4
  simp only [← c29424, ← c29425, ← c29426] at e_457_4
  have e_457_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_5
  simp only [← c29427, ← c29428, k2, ← c29429] at e_457_5
  have hr := e_457_5
  simp only [e_457_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1418)) := by
  have c29430 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29431 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29432 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29433 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29434 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29435 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_6
  simp only [← c29430, ← c29431, ← c29432] at e_457_6
  have e_457_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_7
  simp only [← c29433, ← c29434, k2, ← c29435] at e_457_7
  have hr := e_457_7
  simp only [e_457_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1419)) := by
  have c29436 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29437 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29438 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29439 := (publicBatchWrapper89_copies919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29440 := (publicBatchWrapper89_copies920 a h).1
  have c29441 := (publicBatchWrapper89_copies920 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_8
  simp only [← c29436, ← c29437, ← c29438] at e_457_8
  have e_457_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_9
  simp only [← c29439, ← c29440, k2, ← c29441] at e_457_9
  have hr := e_457_9
  simp only [e_457_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1420)) := by
  have c29442 := (publicBatchWrapper89_copies920 a h).2.2.1
  have c29443 := (publicBatchWrapper89_copies920 a h).2.2.2.1
  have c29444 := (publicBatchWrapper89_copies920 a h).2.2.2.2.1
  have c29445 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.1
  have c29446 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.1
  have c29447 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_10
  simp only [← c29442, ← c29443, ← c29444] at e_457_10
  have e_457_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_11
  simp only [← c29445, ← c29446, k2, ← c29447] at e_457_11
  have hr := e_457_11
  simp only [e_457_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1421)) := by
  have c29448 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.1
  have c29449 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.1
  have c29450 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29451 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29452 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29453 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_12
  simp only [← c29448, ← c29449, ← c29450] at e_457_12
  have e_457_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_13
  simp only [← c29451, ← c29452, k2, ← c29453] at e_457_13
  have hr := e_457_13
  simp only [e_457_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1422)) := by
  have c29454 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29455 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29456 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29457 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29458 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29459 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_14
  simp only [← c29454, ← c29455, ← c29456] at e_457_14
  have e_457_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_15
  simp only [← c29457, ← c29458, k2, ← c29459] at e_457_15
  have hr := e_457_15
  simp only [e_457_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1423)) := by
  have c29460 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29461 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29462 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29463 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29464 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29465 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_16
  simp only [← c29460, ← c29461, ← c29462] at e_457_16
  have e_457_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_17
  simp only [← c29463, ← c29464, k2, ← c29465] at e_457_17
  have hr := e_457_17
  simp only [e_457_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 457 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1424)) := by
  have c29466 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29467 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29468 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29469 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29470 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29471 := (publicBatchWrapper89_copies920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_457_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_18
  simp only [← c29466, ← c29467, ← c29468] at e_457_18
  have e_457_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_457_19
  simp only [← c29469, ← c29470, k2, ← c29471] at e_457_19
  have hr := e_457_19
  simp only [e_457_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1425)) := by
  have c29472 := (publicBatchWrapper89_copies921 a h).1
  have c29473 := (publicBatchWrapper89_copies921 a h).2.1
  have c29474 := (publicBatchWrapper89_copies921 a h).2.2.1
  have c29475 := (publicBatchWrapper89_copies921 a h).2.2.2.1
  have c29476 := (publicBatchWrapper89_copies921 a h).2.2.2.2.1
  have c29477 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_0
  simp only [← c29472, ← c29473, ← c29474] at e_458_0
  have e_458_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_1
  simp only [← c29475, ← c29476, k2, ← c29477] at e_458_1
  have hr := e_458_1
  simp only [e_458_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1426)) := by
  have c29478 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.1
  have c29479 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.1
  have c29480 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.1
  have c29481 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.1
  have c29482 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29483 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_2
  simp only [← c29478, ← c29479, ← c29480] at e_458_2
  have e_458_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_3
  simp only [← c29481, ← c29482, k2, ← c29483] at e_458_3
  have hr := e_458_3
  simp only [e_458_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1427)) := by
  have c29484 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29485 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29486 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29487 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29488 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29489 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_4
  simp only [← c29484, ← c29485, ← c29486] at e_458_4
  have e_458_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_5
  simp only [← c29487, ← c29488, k2, ← c29489] at e_458_5
  have hr := e_458_5
  simp only [e_458_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1428)) := by
  have c29490 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29491 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29492 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29493 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29494 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29495 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_6
  simp only [← c29490, ← c29491, ← c29492] at e_458_6
  have e_458_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_7
  simp only [← c29493, ← c29494, k2, ← c29495] at e_458_7
  have hr := e_458_7
  simp only [e_458_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1429)) := by
  have c29496 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29497 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29498 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29499 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29500 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29501 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_8
  simp only [← c29496, ← c29497, ← c29498] at e_458_8
  have e_458_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_9
  simp only [← c29499, ← c29500, k2, ← c29501] at e_458_9
  have hr := e_458_9
  simp only [e_458_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1430)) := by
  have c29502 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29503 := (publicBatchWrapper89_copies921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29504 := (publicBatchWrapper89_copies922 a h).1
  have c29505 := (publicBatchWrapper89_copies922 a h).2.1
  have c29506 := (publicBatchWrapper89_copies922 a h).2.2.1
  have c29507 := (publicBatchWrapper89_copies922 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_10
  simp only [← c29502, ← c29503, ← c29504] at e_458_10
  have e_458_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_11
  simp only [← c29505, ← c29506, k2, ← c29507] at e_458_11
  have hr := e_458_11
  simp only [e_458_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1431)) := by
  have c29508 := (publicBatchWrapper89_copies922 a h).2.2.2.2.1
  have c29509 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.1
  have c29510 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.1
  have c29511 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.1
  have c29512 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.1
  have c29513 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_12
  simp only [← c29508, ← c29509, ← c29510] at e_458_12
  have e_458_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_13
  simp only [← c29511, ← c29512, k2, ← c29513] at e_458_13
  have hr := e_458_13
  simp only [e_458_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1432)) := by
  have c29514 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29515 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29516 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29517 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29518 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29519 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_14
  simp only [← c29514, ← c29515, ← c29516] at e_458_14
  have e_458_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_15
  simp only [← c29517, ← c29518, k2, ← c29519] at e_458_15
  have hr := e_458_15
  simp only [e_458_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1433)) := by
  have c29520 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29521 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29522 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29523 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29524 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29525 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_16
  simp only [← c29520, ← c29521, ← c29522] at e_458_16
  have e_458_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_17
  simp only [← c29523, ← c29524, k2, ← c29525] at e_458_17
  have hr := e_458_17
  simp only [e_458_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 458 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1434)) := by
  have c29526 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29527 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29528 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29529 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29530 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29531 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_458_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_18
  simp only [← c29526, ← c29527, ← c29528] at e_458_18
  have e_458_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_458_19
  simp only [← c29529, ← c29530, k2, ← c29531] at e_458_19
  have hr := e_458_19
  simp only [e_458_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1435)) := by
  have c29532 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29533 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29534 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29535 := (publicBatchWrapper89_copies922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29536 := (publicBatchWrapper89_copies923 a h).1
  have c29537 := (publicBatchWrapper89_copies923 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_0
  simp only [← c29532, ← c29533, ← c29534] at e_459_0
  have e_459_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_1
  simp only [← c29535, ← c29536, k2, ← c29537] at e_459_1
  have hr := e_459_1
  simp only [e_459_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1436)) := by
  have c29538 := (publicBatchWrapper89_copies923 a h).2.2.1
  have c29539 := (publicBatchWrapper89_copies923 a h).2.2.2.1
  have c29540 := (publicBatchWrapper89_copies923 a h).2.2.2.2.1
  have c29541 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.1
  have c29542 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.1
  have c29543 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_2
  simp only [← c29538, ← c29539, ← c29540] at e_459_2
  have e_459_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_3
  simp only [← c29541, ← c29542, k2, ← c29543] at e_459_3
  have hr := e_459_3
  simp only [e_459_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1437)) := by
  have c29544 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.1
  have c29545 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.1
  have c29546 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29547 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29548 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29549 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_4
  simp only [← c29544, ← c29545, ← c29546] at e_459_4
  have e_459_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_5
  simp only [← c29547, ← c29548, k2, ← c29549] at e_459_5
  have hr := e_459_5
  simp only [e_459_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1438)) := by
  have c29550 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29551 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29552 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29553 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29554 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29555 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_6
  simp only [← c29550, ← c29551, ← c29552] at e_459_6
  have e_459_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_7
  simp only [← c29553, ← c29554, k2, ← c29555] at e_459_7
  have hr := e_459_7
  simp only [e_459_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1439)) := by
  have c29556 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29557 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29558 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29559 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29560 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29561 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_8
  simp only [← c29556, ← c29557, ← c29558] at e_459_8
  have e_459_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_9
  simp only [← c29559, ← c29560, k2, ← c29561] at e_459_9
  have hr := e_459_9
  simp only [e_459_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1440)) := by
  have c29562 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29563 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29564 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29565 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29566 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29567 := (publicBatchWrapper89_copies923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_10
  simp only [← c29562, ← c29563, ← c29564] at e_459_10
  have e_459_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_11
  simp only [← c29565, ← c29566, k2, ← c29567] at e_459_11
  have hr := e_459_11
  simp only [e_459_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1441)) := by
  have c29568 := (publicBatchWrapper89_copies924 a h).1
  have c29569 := (publicBatchWrapper89_copies924 a h).2.1
  have c29570 := (publicBatchWrapper89_copies924 a h).2.2.1
  have c29571 := (publicBatchWrapper89_copies924 a h).2.2.2.1
  have c29572 := (publicBatchWrapper89_copies924 a h).2.2.2.2.1
  have c29573 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_12
  simp only [← c29568, ← c29569, ← c29570] at e_459_12
  have e_459_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_13
  simp only [← c29571, ← c29572, k2, ← c29573] at e_459_13
  have hr := e_459_13
  simp only [e_459_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1442)) := by
  have c29574 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.1
  have c29575 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.1
  have c29576 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.1
  have c29577 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.1
  have c29578 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29579 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_14
  simp only [← c29574, ← c29575, ← c29576] at e_459_14
  have e_459_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_15
  simp only [← c29577, ← c29578, k2, ← c29579] at e_459_15
  have hr := e_459_15
  simp only [e_459_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1443)) := by
  have c29580 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29581 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29582 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29583 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29584 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29585 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_16
  simp only [← c29580, ← c29581, ← c29582] at e_459_16
  have e_459_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_17
  simp only [← c29583, ← c29584, k2, ← c29585] at e_459_17
  have hr := e_459_17
  simp only [e_459_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 459 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1444)) := by
  have c29586 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29587 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29588 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29589 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29590 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29591 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_459_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_18
  simp only [← c29586, ← c29587, ← c29588] at e_459_18
  have e_459_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_459_19
  simp only [← c29589, ← c29590, k2, ← c29591] at e_459_19
  have hr := e_459_19
  simp only [e_459_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1445)) := by
  have c29592 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29593 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29594 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29595 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29596 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29597 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_0
  simp only [← c29592, ← c29593, ← c29594] at e_460_0
  have e_460_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_1
  simp only [← c29595, ← c29596, k2, ← c29597] at e_460_1
  have hr := e_460_1
  simp only [e_460_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1446)) := by
  have c29598 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29599 := (publicBatchWrapper89_copies924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29600 := (publicBatchWrapper89_copies925 a h).1
  have c29601 := (publicBatchWrapper89_copies925 a h).2.1
  have c29602 := (publicBatchWrapper89_copies925 a h).2.2.1
  have c29603 := (publicBatchWrapper89_copies925 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_2
  simp only [← c29598, ← c29599, ← c29600] at e_460_2
  have e_460_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_3
  simp only [← c29601, ← c29602, k2, ← c29603] at e_460_3
  have hr := e_460_3
  simp only [e_460_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1447)) := by
  have c29604 := (publicBatchWrapper89_copies925 a h).2.2.2.2.1
  have c29605 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.1
  have c29606 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.1
  have c29607 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.1
  have c29608 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.1
  have c29609 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_4
  simp only [← c29604, ← c29605, ← c29606] at e_460_4
  have e_460_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_5
  simp only [← c29607, ← c29608, k2, ← c29609] at e_460_5
  have hr := e_460_5
  simp only [e_460_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1448)) := by
  have c29610 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29611 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29612 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29613 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29614 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29615 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_6
  simp only [← c29610, ← c29611, ← c29612] at e_460_6
  have e_460_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_7
  simp only [← c29613, ← c29614, k2, ← c29615] at e_460_7
  have hr := e_460_7
  simp only [e_460_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1449)) := by
  have c29616 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29617 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29618 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29619 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29620 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29621 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_8
  simp only [← c29616, ← c29617, ← c29618] at e_460_8
  have e_460_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_9
  simp only [← c29619, ← c29620, k2, ← c29621] at e_460_9
  have hr := e_460_9
  simp only [e_460_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1450)) := by
  have c29622 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29623 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29624 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29625 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29626 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29627 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_10
  simp only [← c29622, ← c29623, ← c29624] at e_460_10
  have e_460_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_11
  simp only [← c29625, ← c29626, k2, ← c29627] at e_460_11
  have hr := e_460_11
  simp only [e_460_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1451)) := by
  have c29628 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29629 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29630 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29631 := (publicBatchWrapper89_copies925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29632 := (publicBatchWrapper89_copies926 a h).1
  have c29633 := (publicBatchWrapper89_copies926 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_12
  simp only [← c29628, ← c29629, ← c29630] at e_460_12
  have e_460_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_13
  simp only [← c29631, ← c29632, k2, ← c29633] at e_460_13
  have hr := e_460_13
  simp only [e_460_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1452)) := by
  have c29634 := (publicBatchWrapper89_copies926 a h).2.2.1
  have c29635 := (publicBatchWrapper89_copies926 a h).2.2.2.1
  have c29636 := (publicBatchWrapper89_copies926 a h).2.2.2.2.1
  have c29637 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.1
  have c29638 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.1
  have c29639 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_14
  simp only [← c29634, ← c29635, ← c29636] at e_460_14
  have e_460_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_15
  simp only [← c29637, ← c29638, k2, ← c29639] at e_460_15
  have hr := e_460_15
  simp only [e_460_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1453)) := by
  have c29640 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.1
  have c29641 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.1
  have c29642 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29643 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29644 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29645 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_16
  simp only [← c29640, ← c29641, ← c29642] at e_460_16
  have e_460_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_17
  simp only [← c29643, ← c29644, k2, ← c29645] at e_460_17
  have hr := e_460_17
  simp only [e_460_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 460 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1454)) := by
  have c29646 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29647 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29648 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29649 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29650 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29651 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_460_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_18
  simp only [← c29646, ← c29647, ← c29648] at e_460_18
  have e_460_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_460_19
  simp only [← c29649, ← c29650, k2, ← c29651] at e_460_19
  have hr := e_460_19
  simp only [e_460_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1455)) := by
  have c29652 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29653 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29654 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29655 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29656 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29657 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_0
  simp only [← c29652, ← c29653, ← c29654] at e_461_0
  have e_461_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_1
  simp only [← c29655, ← c29656, k2, ← c29657] at e_461_1
  have hr := e_461_1
  simp only [e_461_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1456)) := by
  have c29658 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29659 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29660 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29661 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29662 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29663 := (publicBatchWrapper89_copies926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_2
  simp only [← c29658, ← c29659, ← c29660] at e_461_2
  have e_461_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_3
  simp only [← c29661, ← c29662, k2, ← c29663] at e_461_3
  have hr := e_461_3
  simp only [e_461_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1457)) := by
  have c29664 := (publicBatchWrapper89_copies927 a h).1
  have c29665 := (publicBatchWrapper89_copies927 a h).2.1
  have c29666 := (publicBatchWrapper89_copies927 a h).2.2.1
  have c29667 := (publicBatchWrapper89_copies927 a h).2.2.2.1
  have c29668 := (publicBatchWrapper89_copies927 a h).2.2.2.2.1
  have c29669 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_4
  simp only [← c29664, ← c29665, ← c29666] at e_461_4
  have e_461_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_5
  simp only [← c29667, ← c29668, k2, ← c29669] at e_461_5
  have hr := e_461_5
  simp only [e_461_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1458)) := by
  have c29670 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.1
  have c29671 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.1
  have c29672 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.1
  have c29673 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.1
  have c29674 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29675 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_6
  simp only [← c29670, ← c29671, ← c29672] at e_461_6
  have e_461_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_7
  simp only [← c29673, ← c29674, k2, ← c29675] at e_461_7
  have hr := e_461_7
  simp only [e_461_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1459)) := by
  have c29676 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29677 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29678 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29679 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29680 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29681 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_8
  simp only [← c29676, ← c29677, ← c29678] at e_461_8
  have e_461_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_9
  simp only [← c29679, ← c29680, k2, ← c29681] at e_461_9
  have hr := e_461_9
  simp only [e_461_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1460)) := by
  have c29682 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29683 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29684 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29685 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29686 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29687 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_10
  simp only [← c29682, ← c29683, ← c29684] at e_461_10
  have e_461_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_11
  simp only [← c29685, ← c29686, k2, ← c29687] at e_461_11
  have hr := e_461_11
  simp only [e_461_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1461)) := by
  have c29688 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29689 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29690 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29691 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29692 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29693 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_12
  simp only [← c29688, ← c29689, ← c29690] at e_461_12
  have e_461_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_13
  simp only [← c29691, ← c29692, k2, ← c29693] at e_461_13
  have hr := e_461_13
  simp only [e_461_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1462)) := by
  have c29694 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29695 := (publicBatchWrapper89_copies927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29696 := (publicBatchWrapper89_copies928 a h).1
  have c29697 := (publicBatchWrapper89_copies928 a h).2.1
  have c29698 := (publicBatchWrapper89_copies928 a h).2.2.1
  have c29699 := (publicBatchWrapper89_copies928 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_14
  simp only [← c29694, ← c29695, ← c29696] at e_461_14
  have e_461_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_15
  simp only [← c29697, ← c29698, k2, ← c29699] at e_461_15
  have hr := e_461_15
  simp only [e_461_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1463)) := by
  have c29700 := (publicBatchWrapper89_copies928 a h).2.2.2.2.1
  have c29701 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.1
  have c29702 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.1
  have c29703 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.1
  have c29704 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.1
  have c29705 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_16
  simp only [← c29700, ← c29701, ← c29702] at e_461_16
  have e_461_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_17
  simp only [← c29703, ← c29704, k2, ← c29705] at e_461_17
  have hr := e_461_17
  simp only [e_461_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 461 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1464)) := by
  have c29706 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29707 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29708 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29709 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29710 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29711 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_461_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_18
  simp only [← c29706, ← c29707, ← c29708] at e_461_18
  have e_461_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_461_19
  simp only [← c29709, ← c29710, k2, ← c29711] at e_461_19
  have hr := e_461_19
  simp only [e_461_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1465)) := by
  have c29712 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29713 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29714 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29715 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29716 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29717 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_0
  simp only [← c29712, ← c29713, ← c29714] at e_462_0
  have e_462_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_1
  simp only [← c29715, ← c29716, k2, ← c29717] at e_462_1
  have hr := e_462_1
  simp only [e_462_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1466)) := by
  have c29718 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29719 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29720 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29721 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29722 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29723 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_2
  simp only [← c29718, ← c29719, ← c29720] at e_462_2
  have e_462_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_3
  simp only [← c29721, ← c29722, k2, ← c29723] at e_462_3
  have hr := e_462_3
  simp only [e_462_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1467)) := by
  have c29724 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29725 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29726 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29727 := (publicBatchWrapper89_copies928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29728 := (publicBatchWrapper89_copies929 a h).1
  have c29729 := (publicBatchWrapper89_copies929 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_4
  simp only [← c29724, ← c29725, ← c29726] at e_462_4
  have e_462_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_5
  simp only [← c29727, ← c29728, k2, ← c29729] at e_462_5
  have hr := e_462_5
  simp only [e_462_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1468)) := by
  have c29730 := (publicBatchWrapper89_copies929 a h).2.2.1
  have c29731 := (publicBatchWrapper89_copies929 a h).2.2.2.1
  have c29732 := (publicBatchWrapper89_copies929 a h).2.2.2.2.1
  have c29733 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.1
  have c29734 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.1
  have c29735 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_6
  simp only [← c29730, ← c29731, ← c29732] at e_462_6
  have e_462_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_7
  simp only [← c29733, ← c29734, k2, ← c29735] at e_462_7
  have hr := e_462_7
  simp only [e_462_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1469)) := by
  have c29736 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.1
  have c29737 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.1
  have c29738 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29739 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29740 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29741 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_8
  simp only [← c29736, ← c29737, ← c29738] at e_462_8
  have e_462_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_9
  simp only [← c29739, ← c29740, k2, ← c29741] at e_462_9
  have hr := e_462_9
  simp only [e_462_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1470)) := by
  have c29742 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29743 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29744 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29745 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29746 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29747 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_10
  simp only [← c29742, ← c29743, ← c29744] at e_462_10
  have e_462_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_11
  simp only [← c29745, ← c29746, k2, ← c29747] at e_462_11
  have hr := e_462_11
  simp only [e_462_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1471)) := by
  have c29748 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29749 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29750 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29751 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29752 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29753 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_12
  simp only [← c29748, ← c29749, ← c29750] at e_462_12
  have e_462_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_13
  simp only [← c29751, ← c29752, k2, ← c29753] at e_462_13
  have hr := e_462_13
  simp only [e_462_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1472)) := by
  have c29754 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29755 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29756 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29757 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29758 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29759 := (publicBatchWrapper89_copies929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_14
  simp only [← c29754, ← c29755, ← c29756] at e_462_14
  have e_462_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_15
  simp only [← c29757, ← c29758, k2, ← c29759] at e_462_15
  have hr := e_462_15
  simp only [e_462_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1473)) := by
  have c29760 := (publicBatchWrapper89_copies930 a h).1
  have c29761 := (publicBatchWrapper89_copies930 a h).2.1
  have c29762 := (publicBatchWrapper89_copies930 a h).2.2.1
  have c29763 := (publicBatchWrapper89_copies930 a h).2.2.2.1
  have c29764 := (publicBatchWrapper89_copies930 a h).2.2.2.2.1
  have c29765 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_16
  simp only [← c29760, ← c29761, ← c29762] at e_462_16
  have e_462_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_17
  simp only [← c29763, ← c29764, k2, ← c29765] at e_462_17
  have hr := e_462_17
  simp only [e_462_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 462 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1474)) := by
  have c29766 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.1
  have c29767 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.1
  have c29768 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.1
  have c29769 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.1
  have c29770 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29771 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_462_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_18
  simp only [← c29766, ← c29767, ← c29768] at e_462_18
  have e_462_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_462_19
  simp only [← c29769, ← c29770, k2, ← c29771] at e_462_19
  have hr := e_462_19
  simp only [e_462_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1475)) := by
  have c29772 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29773 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29774 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29775 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29776 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29777 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_0
  simp only [← c29772, ← c29773, ← c29774] at e_463_0
  have e_463_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_1
  simp only [← c29775, ← c29776, k2, ← c29777] at e_463_1
  have hr := e_463_1
  simp only [e_463_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1476)) := by
  have c29778 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29779 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29780 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29781 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29782 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29783 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_2
  simp only [← c29778, ← c29779, ← c29780] at e_463_2
  have e_463_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_3
  simp only [← c29781, ← c29782, k2, ← c29783] at e_463_3
  have hr := e_463_3
  simp only [e_463_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1477)) := by
  have c29784 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29785 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29786 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29787 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29788 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29789 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_4
  simp only [← c29784, ← c29785, ← c29786] at e_463_4
  have e_463_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_5
  simp only [← c29787, ← c29788, k2, ← c29789] at e_463_5
  have hr := e_463_5
  simp only [e_463_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1478)) := by
  have c29790 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29791 := (publicBatchWrapper89_copies930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29792 := (publicBatchWrapper89_copies931 a h).1
  have c29793 := (publicBatchWrapper89_copies931 a h).2.1
  have c29794 := (publicBatchWrapper89_copies931 a h).2.2.1
  have c29795 := (publicBatchWrapper89_copies931 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_6
  simp only [← c29790, ← c29791, ← c29792] at e_463_6
  have e_463_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_7
  simp only [← c29793, ← c29794, k2, ← c29795] at e_463_7
  have hr := e_463_7
  simp only [e_463_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1479)) := by
  have c29796 := (publicBatchWrapper89_copies931 a h).2.2.2.2.1
  have c29797 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.1
  have c29798 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.1
  have c29799 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.1
  have c29800 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.1
  have c29801 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_8
  simp only [← c29796, ← c29797, ← c29798] at e_463_8
  have e_463_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_9
  simp only [← c29799, ← c29800, k2, ← c29801] at e_463_9
  have hr := e_463_9
  simp only [e_463_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1480)) := by
  have c29802 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29803 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29804 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29805 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29806 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29807 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_10
  simp only [← c29802, ← c29803, ← c29804] at e_463_10
  have e_463_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_11
  simp only [← c29805, ← c29806, k2, ← c29807] at e_463_11
  have hr := e_463_11
  simp only [e_463_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1481)) := by
  have c29808 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29809 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29810 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29811 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29812 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29813 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_12
  simp only [← c29808, ← c29809, ← c29810] at e_463_12
  have e_463_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_13
  simp only [← c29811, ← c29812, k2, ← c29813] at e_463_13
  have hr := e_463_13
  simp only [e_463_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1482)) := by
  have c29814 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29815 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29816 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29817 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29818 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29819 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_14
  simp only [← c29814, ← c29815, ← c29816] at e_463_14
  have e_463_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_15
  simp only [← c29817, ← c29818, k2, ← c29819] at e_463_15
  have hr := e_463_15
  simp only [e_463_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1483)) := by
  have c29820 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29821 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29822 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29823 := (publicBatchWrapper89_copies931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29824 := (publicBatchWrapper89_copies932 a h).1
  have c29825 := (publicBatchWrapper89_copies932 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_16
  simp only [← c29820, ← c29821, ← c29822] at e_463_16
  have e_463_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_17
  simp only [← c29823, ← c29824, k2, ← c29825] at e_463_17
  have hr := e_463_17
  simp only [e_463_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 463 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1484)) := by
  have c29826 := (publicBatchWrapper89_copies932 a h).2.2.1
  have c29827 := (publicBatchWrapper89_copies932 a h).2.2.2.1
  have c29828 := (publicBatchWrapper89_copies932 a h).2.2.2.2.1
  have c29829 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.1
  have c29830 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.1
  have c29831 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_463_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_18
  simp only [← c29826, ← c29827, ← c29828] at e_463_18
  have e_463_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_463_19
  simp only [← c29829, ← c29830, k2, ← c29831] at e_463_19
  have hr := e_463_19
  simp only [e_463_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1485)) := by
  have c29832 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.1
  have c29833 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.1
  have c29834 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29835 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29836 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29837 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_0
  simp only [← c29832, ← c29833, ← c29834] at e_464_0
  have e_464_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_1
  simp only [← c29835, ← c29836, k2, ← c29837] at e_464_1
  have hr := e_464_1
  simp only [e_464_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1486)) := by
  have c29838 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29839 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29840 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29841 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29842 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29843 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_2
  simp only [← c29838, ← c29839, ← c29840] at e_464_2
  have e_464_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_3
  simp only [← c29841, ← c29842, k2, ← c29843] at e_464_3
  have hr := e_464_3
  simp only [e_464_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1487)) := by
  have c29844 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29845 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29846 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29847 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29848 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29849 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_4
  simp only [← c29844, ← c29845, ← c29846] at e_464_4
  have e_464_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_5
  simp only [← c29847, ← c29848, k2, ← c29849] at e_464_5
  have hr := e_464_5
  simp only [e_464_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1488)) := by
  have c29850 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29851 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29852 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29853 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29854 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29855 := (publicBatchWrapper89_copies932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_6
  simp only [← c29850, ← c29851, ← c29852] at e_464_6
  have e_464_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_7
  simp only [← c29853, ← c29854, k2, ← c29855] at e_464_7
  have hr := e_464_7
  simp only [e_464_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1489)) := by
  have c29856 := (publicBatchWrapper89_copies933 a h).1
  have c29857 := (publicBatchWrapper89_copies933 a h).2.1
  have c29858 := (publicBatchWrapper89_copies933 a h).2.2.1
  have c29859 := (publicBatchWrapper89_copies933 a h).2.2.2.1
  have c29860 := (publicBatchWrapper89_copies933 a h).2.2.2.2.1
  have c29861 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_8
  simp only [← c29856, ← c29857, ← c29858] at e_464_8
  have e_464_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_9
  simp only [← c29859, ← c29860, k2, ← c29861] at e_464_9
  have hr := e_464_9
  simp only [e_464_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1490)) := by
  have c29862 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.1
  have c29863 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.1
  have c29864 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.1
  have c29865 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.1
  have c29866 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29867 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_10
  simp only [← c29862, ← c29863, ← c29864] at e_464_10
  have e_464_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_11
  simp only [← c29865, ← c29866, k2, ← c29867] at e_464_11
  have hr := e_464_11
  simp only [e_464_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1491)) := by
  have c29868 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29869 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29870 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29871 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29872 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29873 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_12
  simp only [← c29868, ← c29869, ← c29870] at e_464_12
  have e_464_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_13
  simp only [← c29871, ← c29872, k2, ← c29873] at e_464_13
  have hr := e_464_13
  simp only [e_464_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1492)) := by
  have c29874 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29875 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29876 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29877 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29878 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29879 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_14
  simp only [← c29874, ← c29875, ← c29876] at e_464_14
  have e_464_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_15
  simp only [← c29877, ← c29878, k2, ← c29879] at e_464_15
  have hr := e_464_15
  simp only [e_464_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1493)) := by
  have c29880 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29881 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29882 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29883 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29884 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29885 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_16
  simp only [← c29880, ← c29881, ← c29882] at e_464_16
  have e_464_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_17
  simp only [← c29883, ← c29884, k2, ← c29885] at e_464_17
  have hr := e_464_17
  simp only [e_464_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 464 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1494)) := by
  have c29886 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29887 := (publicBatchWrapper89_copies933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29888 := (publicBatchWrapper89_copies934 a h).1
  have c29889 := (publicBatchWrapper89_copies934 a h).2.1
  have c29890 := (publicBatchWrapper89_copies934 a h).2.2.1
  have c29891 := (publicBatchWrapper89_copies934 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_464_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_18
  simp only [← c29886, ← c29887, ← c29888] at e_464_18
  have e_464_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_464_19
  simp only [← c29889, ← c29890, k2, ← c29891] at e_464_19
  have hr := e_464_19
  simp only [e_464_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1495)) := by
  have c29892 := (publicBatchWrapper89_copies934 a h).2.2.2.2.1
  have c29893 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.1
  have c29894 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.1
  have c29895 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.1
  have c29896 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.1
  have c29897 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_0
  simp only [← c29892, ← c29893, ← c29894] at e_465_0
  have e_465_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_1
  simp only [← c29895, ← c29896, k2, ← c29897] at e_465_1
  have hr := e_465_1
  simp only [e_465_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1496)) := by
  have c29898 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29899 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29900 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29901 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29902 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29903 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_2
  simp only [← c29898, ← c29899, ← c29900] at e_465_2
  have e_465_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_3
  simp only [← c29901, ← c29902, k2, ← c29903] at e_465_3
  have hr := e_465_3
  simp only [e_465_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1497)) := by
  have c29904 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29905 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29906 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29907 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29908 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29909 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_4
  simp only [← c29904, ← c29905, ← c29906] at e_465_4
  have e_465_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_5
  simp only [← c29907, ← c29908, k2, ← c29909] at e_465_5
  have hr := e_465_5
  simp only [e_465_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1498)) := by
  have c29910 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29911 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29912 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29913 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29914 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29915 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_6
  simp only [← c29910, ← c29911, ← c29912] at e_465_6
  have e_465_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_7
  simp only [← c29913, ← c29914, k2, ← c29915] at e_465_7
  have hr := e_465_7
  simp only [e_465_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1499)) := by
  have c29916 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29917 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29918 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29919 := (publicBatchWrapper89_copies934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29920 := (publicBatchWrapper89_copies935 a h).1
  have c29921 := (publicBatchWrapper89_copies935 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_8
  simp only [← c29916, ← c29917, ← c29918] at e_465_8
  have e_465_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_9
  simp only [← c29919, ← c29920, k2, ← c29921] at e_465_9
  have hr := e_465_9
  simp only [e_465_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1500)) := by
  have c29922 := (publicBatchWrapper89_copies935 a h).2.2.1
  have c29923 := (publicBatchWrapper89_copies935 a h).2.2.2.1
  have c29924 := (publicBatchWrapper89_copies935 a h).2.2.2.2.1
  have c29925 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.1
  have c29926 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.1
  have c29927 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_10
  simp only [← c29922, ← c29923, ← c29924] at e_465_10
  have e_465_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_11
  simp only [← c29925, ← c29926, k2, ← c29927] at e_465_11
  have hr := e_465_11
  simp only [e_465_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1501)) := by
  have c29928 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.1
  have c29929 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.1
  have c29930 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29931 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29932 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29933 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_12
  simp only [← c29928, ← c29929, ← c29930] at e_465_12
  have e_465_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_13
  simp only [← c29931, ← c29932, k2, ← c29933] at e_465_13
  have hr := e_465_13
  simp only [e_465_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1502)) := by
  have c29934 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29935 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29936 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29937 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29938 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29939 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_14
  simp only [← c29934, ← c29935, ← c29936] at e_465_14
  have e_465_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_15
  simp only [← c29937, ← c29938, k2, ← c29939] at e_465_15
  have hr := e_465_15
  simp only [e_465_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1503)) := by
  have c29940 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29941 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29942 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29943 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29944 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29945 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_16
  simp only [← c29940, ← c29941, ← c29942] at e_465_16
  have e_465_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_17
  simp only [← c29943, ← c29944, k2, ← c29945] at e_465_17
  have hr := e_465_17
  simp only [e_465_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 465 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1504)) := by
  have c29946 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29947 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29948 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29949 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29950 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29951 := (publicBatchWrapper89_copies935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_465_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_18
  simp only [← c29946, ← c29947, ← c29948] at e_465_18
  have e_465_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_465_19
  simp only [← c29949, ← c29950, k2, ← c29951] at e_465_19
  have hr := e_465_19
  simp only [e_465_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1505)) := by
  have c29952 := (publicBatchWrapper89_copies936 a h).1
  have c29953 := (publicBatchWrapper89_copies936 a h).2.1
  have c29954 := (publicBatchWrapper89_copies936 a h).2.2.1
  have c29955 := (publicBatchWrapper89_copies936 a h).2.2.2.1
  have c29956 := (publicBatchWrapper89_copies936 a h).2.2.2.2.1
  have c29957 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_0
  simp only [← c29952, ← c29953, ← c29954] at e_466_0
  have e_466_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_1
  simp only [← c29955, ← c29956, k2, ← c29957] at e_466_1
  have hr := e_466_1
  simp only [e_466_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1506)) := by
  have c29958 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.1
  have c29959 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.1
  have c29960 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.1
  have c29961 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.1
  have c29962 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29963 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_2
  simp only [← c29958, ← c29959, ← c29960] at e_466_2
  have e_466_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_3
  simp only [← c29961, ← c29962, k2, ← c29963] at e_466_3
  have hr := e_466_3
  simp only [e_466_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1507)) := by
  have c29964 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29965 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29966 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29967 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29968 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29969 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_4
  simp only [← c29964, ← c29965, ← c29966] at e_466_4
  have e_466_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_5
  simp only [← c29967, ← c29968, k2, ← c29969] at e_466_5
  have hr := e_466_5
  simp only [e_466_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1508)) := by
  have c29970 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29971 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29972 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29973 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29974 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29975 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_6
  simp only [← c29970, ← c29971, ← c29972] at e_466_6
  have e_466_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_7
  simp only [← c29973, ← c29974, k2, ← c29975] at e_466_7
  have hr := e_466_7
  simp only [e_466_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1509)) := by
  have c29976 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29977 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29978 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29979 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29980 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29981 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_8
  simp only [← c29976, ← c29977, ← c29978] at e_466_8
  have e_466_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_9
  simp only [← c29979, ← c29980, k2, ← c29981] at e_466_9
  have hr := e_466_9
  simp only [e_466_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1510)) := by
  have c29982 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29983 := (publicBatchWrapper89_copies936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c29984 := (publicBatchWrapper89_copies937 a h).1
  have c29985 := (publicBatchWrapper89_copies937 a h).2.1
  have c29986 := (publicBatchWrapper89_copies937 a h).2.2.1
  have c29987 := (publicBatchWrapper89_copies937 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_10
  simp only [← c29982, ← c29983, ← c29984] at e_466_10
  have e_466_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_11
  simp only [← c29985, ← c29986, k2, ← c29987] at e_466_11
  have hr := e_466_11
  simp only [e_466_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1511)) := by
  have c29988 := (publicBatchWrapper89_copies937 a h).2.2.2.2.1
  have c29989 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.1
  have c29990 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.1
  have c29991 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.1
  have c29992 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.1
  have c29993 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_12
  simp only [← c29988, ← c29989, ← c29990] at e_466_12
  have e_466_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_13
  simp only [← c29991, ← c29992, k2, ← c29993] at e_466_13
  have hr := e_466_13
  simp only [e_466_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1512)) := by
  have c29994 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.1
  have c29995 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c29996 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29997 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29998 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c29999 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_14
  simp only [← c29994, ← c29995, ← c29996] at e_466_14
  have e_466_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_15
  simp only [← c29997, ← c29998, k2, ← c29999] at e_466_15
  have hr := e_466_15
  simp only [e_466_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1513)) := by
  have c30000 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30001 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30002 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30003 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30004 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30005 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_16
  simp only [← c30000, ← c30001, ← c30002] at e_466_16
  have e_466_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_17
  simp only [← c30003, ← c30004, k2, ← c30005] at e_466_17
  have hr := e_466_17
  simp only [e_466_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 466 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1514)) := by
  have c30006 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30007 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30008 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30009 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30010 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30011 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_466_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_18
  simp only [← c30006, ← c30007, ← c30008] at e_466_18
  have e_466_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_466_19
  simp only [← c30009, ← c30010, k2, ← c30011] at e_466_19
  have hr := e_466_19
  simp only [e_466_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1515)) := by
  have c30012 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30013 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30014 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30015 := (publicBatchWrapper89_copies937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30016 := (publicBatchWrapper89_copies938 a h).1
  have c30017 := (publicBatchWrapper89_copies938 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_0
  simp only [← c30012, ← c30013, ← c30014] at e_467_0
  have e_467_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_1
  simp only [← c30015, ← c30016, k2, ← c30017] at e_467_1
  have hr := e_467_1
  simp only [e_467_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1516)) := by
  have c30018 := (publicBatchWrapper89_copies938 a h).2.2.1
  have c30019 := (publicBatchWrapper89_copies938 a h).2.2.2.1
  have c30020 := (publicBatchWrapper89_copies938 a h).2.2.2.2.1
  have c30021 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.1
  have c30022 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.1
  have c30023 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_2
  simp only [← c30018, ← c30019, ← c30020] at e_467_2
  have e_467_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_3
  simp only [← c30021, ← c30022, k2, ← c30023] at e_467_3
  have hr := e_467_3
  simp only [e_467_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1517)) := by
  have c30024 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.1
  have c30025 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.1
  have c30026 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30027 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30028 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30029 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_4
  simp only [← c30024, ← c30025, ← c30026] at e_467_4
  have e_467_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_5
  simp only [← c30027, ← c30028, k2, ← c30029] at e_467_5
  have hr := e_467_5
  simp only [e_467_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1518)) := by
  have c30030 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30031 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30032 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30033 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30034 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30035 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_6
  simp only [← c30030, ← c30031, ← c30032] at e_467_6
  have e_467_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_7
  simp only [← c30033, ← c30034, k2, ← c30035] at e_467_7
  have hr := e_467_7
  simp only [e_467_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1519)) := by
  have c30036 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30037 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30038 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30039 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30040 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30041 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_8
  simp only [← c30036, ← c30037, ← c30038] at e_467_8
  have e_467_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_9
  simp only [← c30039, ← c30040, k2, ← c30041] at e_467_9
  have hr := e_467_9
  simp only [e_467_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1520)) := by
  have c30042 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30043 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30044 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30045 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30046 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30047 := (publicBatchWrapper89_copies938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_10
  simp only [← c30042, ← c30043, ← c30044] at e_467_10
  have e_467_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_11
  simp only [← c30045, ← c30046, k2, ← c30047] at e_467_11
  have hr := e_467_11
  simp only [e_467_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1521)) := by
  have c30048 := (publicBatchWrapper89_copies939 a h).1
  have c30049 := (publicBatchWrapper89_copies939 a h).2.1
  have c30050 := (publicBatchWrapper89_copies939 a h).2.2.1
  have c30051 := (publicBatchWrapper89_copies939 a h).2.2.2.1
  have c30052 := (publicBatchWrapper89_copies939 a h).2.2.2.2.1
  have c30053 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_12
  simp only [← c30048, ← c30049, ← c30050] at e_467_12
  have e_467_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_13
  simp only [← c30051, ← c30052, k2, ← c30053] at e_467_13
  have hr := e_467_13
  simp only [e_467_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1522)) := by
  have c30054 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.1
  have c30055 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.1
  have c30056 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.1
  have c30057 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.1
  have c30058 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30059 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_14
  simp only [← c30054, ← c30055, ← c30056] at e_467_14
  have e_467_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_15
  simp only [← c30057, ← c30058, k2, ← c30059] at e_467_15
  have hr := e_467_15
  simp only [e_467_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1523)) := by
  have c30060 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30061 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30062 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30063 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30064 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30065 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_16
  simp only [← c30060, ← c30061, ← c30062] at e_467_16
  have e_467_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_17
  simp only [← c30063, ← c30064, k2, ← c30065] at e_467_17
  have hr := e_467_17
  simp only [e_467_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 467 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1524)) := by
  have c30066 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30067 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30068 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30069 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30070 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30071 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_467_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_18
  simp only [← c30066, ← c30067, ← c30068] at e_467_18
  have e_467_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_467_19
  simp only [← c30069, ← c30070, k2, ← c30071] at e_467_19
  have hr := e_467_19
  simp only [e_467_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1525)) := by
  have c30072 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30073 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30074 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30075 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30076 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30077 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_0
  simp only [← c30072, ← c30073, ← c30074] at e_468_0
  have e_468_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_1
  simp only [← c30075, ← c30076, k2, ← c30077] at e_468_1
  have hr := e_468_1
  simp only [e_468_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1526)) := by
  have c30078 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30079 := (publicBatchWrapper89_copies939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30080 := (publicBatchWrapper89_copies940 a h).1
  have c30081 := (publicBatchWrapper89_copies940 a h).2.1
  have c30082 := (publicBatchWrapper89_copies940 a h).2.2.1
  have c30083 := (publicBatchWrapper89_copies940 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_2
  simp only [← c30078, ← c30079, ← c30080] at e_468_2
  have e_468_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_3
  simp only [← c30081, ← c30082, k2, ← c30083] at e_468_3
  have hr := e_468_3
  simp only [e_468_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1527)) := by
  have c30084 := (publicBatchWrapper89_copies940 a h).2.2.2.2.1
  have c30085 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.1
  have c30086 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.1
  have c30087 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.1
  have c30088 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.1
  have c30089 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_4
  simp only [← c30084, ← c30085, ← c30086] at e_468_4
  have e_468_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_5
  simp only [← c30087, ← c30088, k2, ← c30089] at e_468_5
  have hr := e_468_5
  simp only [e_468_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1528)) := by
  have c30090 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30091 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30092 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30093 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30094 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30095 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_6
  simp only [← c30090, ← c30091, ← c30092] at e_468_6
  have e_468_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_7
  simp only [← c30093, ← c30094, k2, ← c30095] at e_468_7
  have hr := e_468_7
  simp only [e_468_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1529)) := by
  have c30096 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30097 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30098 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30099 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30100 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30101 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_8
  simp only [← c30096, ← c30097, ← c30098] at e_468_8
  have e_468_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_9
  simp only [← c30099, ← c30100, k2, ← c30101] at e_468_9
  have hr := e_468_9
  simp only [e_468_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1530)) := by
  have c30102 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30103 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30104 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30105 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30106 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30107 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_10
  simp only [← c30102, ← c30103, ← c30104] at e_468_10
  have e_468_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_11
  simp only [← c30105, ← c30106, k2, ← c30107] at e_468_11
  have hr := e_468_11
  simp only [e_468_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1531)) := by
  have c30108 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30109 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30110 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30111 := (publicBatchWrapper89_copies940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30112 := (publicBatchWrapper89_copies941 a h).1
  have c30113 := (publicBatchWrapper89_copies941 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_12
  simp only [← c30108, ← c30109, ← c30110] at e_468_12
  have e_468_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_13
  simp only [← c30111, ← c30112, k2, ← c30113] at e_468_13
  have hr := e_468_13
  simp only [e_468_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1532)) := by
  have c30114 := (publicBatchWrapper89_copies941 a h).2.2.1
  have c30115 := (publicBatchWrapper89_copies941 a h).2.2.2.1
  have c30116 := (publicBatchWrapper89_copies941 a h).2.2.2.2.1
  have c30117 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.1
  have c30118 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.1
  have c30119 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_14
  simp only [← c30114, ← c30115, ← c30116] at e_468_14
  have e_468_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_15
  simp only [← c30117, ← c30118, k2, ← c30119] at e_468_15
  have hr := e_468_15
  simp only [e_468_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1533)) := by
  have c30120 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.1
  have c30121 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.1
  have c30122 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30123 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30124 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30125 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_16
  simp only [← c30120, ← c30121, ← c30122] at e_468_16
  have e_468_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_17
  simp only [← c30123, ← c30124, k2, ← c30125] at e_468_17
  have hr := e_468_17
  simp only [e_468_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 468 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1534)) := by
  have c30126 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30127 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30128 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30129 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30130 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30131 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_468_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_18
  simp only [← c30126, ← c30127, ← c30128] at e_468_18
  have e_468_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_468_19
  simp only [← c30129, ← c30130, k2, ← c30131] at e_468_19
  have hr := e_468_19
  simp only [e_468_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1535)) := by
  have c30132 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30133 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30134 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30135 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30136 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30137 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_0
  simp only [← c30132, ← c30133, ← c30134] at e_469_0
  have e_469_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_1
  simp only [← c30135, ← c30136, k2, ← c30137] at e_469_1
  have hr := e_469_1
  simp only [e_469_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1536)) := by
  have c30138 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30139 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30140 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30141 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30142 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30143 := (publicBatchWrapper89_copies941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_2
  simp only [← c30138, ← c30139, ← c30140] at e_469_2
  have e_469_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_3
  simp only [← c30141, ← c30142, k2, ← c30143] at e_469_3
  have hr := e_469_3
  simp only [e_469_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1537)) := by
  have c30144 := (publicBatchWrapper89_copies942 a h).1
  have c30145 := (publicBatchWrapper89_copies942 a h).2.1
  have c30146 := (publicBatchWrapper89_copies942 a h).2.2.1
  have c30147 := (publicBatchWrapper89_copies942 a h).2.2.2.1
  have c30148 := (publicBatchWrapper89_copies942 a h).2.2.2.2.1
  have c30149 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_4
  simp only [← c30144, ← c30145, ← c30146] at e_469_4
  have e_469_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_5
  simp only [← c30147, ← c30148, k2, ← c30149] at e_469_5
  have hr := e_469_5
  simp only [e_469_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1538)) := by
  have c30150 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.1
  have c30151 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.1
  have c30152 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.1
  have c30153 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.1
  have c30154 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30155 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_6
  simp only [← c30150, ← c30151, ← c30152] at e_469_6
  have e_469_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_7
  simp only [← c30153, ← c30154, k2, ← c30155] at e_469_7
  have hr := e_469_7
  simp only [e_469_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1539)) := by
  have c30156 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30157 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30158 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30159 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30160 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30161 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_8
  simp only [← c30156, ← c30157, ← c30158] at e_469_8
  have e_469_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_9
  simp only [← c30159, ← c30160, k2, ← c30161] at e_469_9
  have hr := e_469_9
  simp only [e_469_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1540)) := by
  have c30162 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30163 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30164 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30165 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30166 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30167 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_10
  simp only [← c30162, ← c30163, ← c30164] at e_469_10
  have e_469_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_11
  simp only [← c30165, ← c30166, k2, ← c30167] at e_469_11
  have hr := e_469_11
  simp only [e_469_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1541)) := by
  have c30168 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30169 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30170 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30171 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30172 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30173 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_12
  simp only [← c30168, ← c30169, ← c30170] at e_469_12
  have e_469_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_13
  simp only [← c30171, ← c30172, k2, ← c30173] at e_469_13
  have hr := e_469_13
  simp only [e_469_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1542)) := by
  have c30174 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30175 := (publicBatchWrapper89_copies942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30176 := (publicBatchWrapper89_copies943 a h).1
  have c30177 := (publicBatchWrapper89_copies943 a h).2.1
  have c30178 := (publicBatchWrapper89_copies943 a h).2.2.1
  have c30179 := (publicBatchWrapper89_copies943 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_14
  simp only [← c30174, ← c30175, ← c30176] at e_469_14
  have e_469_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_15
  simp only [← c30177, ← c30178, k2, ← c30179] at e_469_15
  have hr := e_469_15
  simp only [e_469_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1543)) := by
  have c30180 := (publicBatchWrapper89_copies943 a h).2.2.2.2.1
  have c30181 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.1
  have c30182 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.1
  have c30183 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.1
  have c30184 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.1
  have c30185 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_16
  simp only [← c30180, ← c30181, ← c30182] at e_469_16
  have e_469_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_17
  simp only [← c30183, ← c30184, k2, ← c30185] at e_469_17
  have hr := e_469_17
  simp only [e_469_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 469 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1544)) := by
  have c30186 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30187 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30188 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30189 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30190 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30191 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_469_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_18
  simp only [← c30186, ← c30187, ← c30188] at e_469_18
  have e_469_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_469_19
  simp only [← c30189, ← c30190, k2, ← c30191] at e_469_19
  have hr := e_469_19
  simp only [e_469_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1545)) := by
  have c30192 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30193 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30194 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30195 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30196 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30197 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_0
  simp only [← c30192, ← c30193, ← c30194] at e_470_0
  have e_470_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_1
  simp only [← c30195, ← c30196, k2, ← c30197] at e_470_1
  have hr := e_470_1
  simp only [e_470_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1546)) := by
  have c30198 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30199 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30200 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30201 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30202 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30203 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_2
  simp only [← c30198, ← c30199, ← c30200] at e_470_2
  have e_470_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_3
  simp only [← c30201, ← c30202, k2, ← c30203] at e_470_3
  have hr := e_470_3
  simp only [e_470_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1547)) := by
  have c30204 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30205 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30206 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30207 := (publicBatchWrapper89_copies943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30208 := (publicBatchWrapper89_copies944 a h).1
  have c30209 := (publicBatchWrapper89_copies944 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_4
  simp only [← c30204, ← c30205, ← c30206] at e_470_4
  have e_470_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_5
  simp only [← c30207, ← c30208, k2, ← c30209] at e_470_5
  have hr := e_470_5
  simp only [e_470_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1548)) := by
  have c30210 := (publicBatchWrapper89_copies944 a h).2.2.1
  have c30211 := (publicBatchWrapper89_copies944 a h).2.2.2.1
  have c30212 := (publicBatchWrapper89_copies944 a h).2.2.2.2.1
  have c30213 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.1
  have c30214 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.1
  have c30215 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_6
  simp only [← c30210, ← c30211, ← c30212] at e_470_6
  have e_470_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_7
  simp only [← c30213, ← c30214, k2, ← c30215] at e_470_7
  have hr := e_470_7
  simp only [e_470_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1549)) := by
  have c30216 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.1
  have c30217 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.1
  have c30218 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30219 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30220 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30221 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_8
  simp only [← c30216, ← c30217, ← c30218] at e_470_8
  have e_470_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_9
  simp only [← c30219, ← c30220, k2, ← c30221] at e_470_9
  have hr := e_470_9
  simp only [e_470_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1550)) := by
  have c30222 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30223 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30224 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30225 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30226 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30227 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_10
  simp only [← c30222, ← c30223, ← c30224] at e_470_10
  have e_470_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_11
  simp only [← c30225, ← c30226, k2, ← c30227] at e_470_11
  have hr := e_470_11
  simp only [e_470_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1551)) := by
  have c30228 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30229 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30230 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30231 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30232 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30233 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_12
  simp only [← c30228, ← c30229, ← c30230] at e_470_12
  have e_470_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_13
  simp only [← c30231, ← c30232, k2, ← c30233] at e_470_13
  have hr := e_470_13
  simp only [e_470_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1552)) := by
  have c30234 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30235 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30236 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30237 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30238 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30239 := (publicBatchWrapper89_copies944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_14
  simp only [← c30234, ← c30235, ← c30236] at e_470_14
  have e_470_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_15
  simp only [← c30237, ← c30238, k2, ← c30239] at e_470_15
  have hr := e_470_15
  simp only [e_470_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1553)) := by
  have c30240 := (publicBatchWrapper89_copies945 a h).1
  have c30241 := (publicBatchWrapper89_copies945 a h).2.1
  have c30242 := (publicBatchWrapper89_copies945 a h).2.2.1
  have c30243 := (publicBatchWrapper89_copies945 a h).2.2.2.1
  have c30244 := (publicBatchWrapper89_copies945 a h).2.2.2.2.1
  have c30245 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_16
  simp only [← c30240, ← c30241, ← c30242] at e_470_16
  have e_470_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_17
  simp only [← c30243, ← c30244, k2, ← c30245] at e_470_17
  have hr := e_470_17
  simp only [e_470_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 470 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1554)) := by
  have c30246 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.1
  have c30247 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.1
  have c30248 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.1
  have c30249 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.1
  have c30250 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30251 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_470_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_18
  simp only [← c30246, ← c30247, ← c30248] at e_470_18
  have e_470_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_470_19
  simp only [← c30249, ← c30250, k2, ← c30251] at e_470_19
  have hr := e_470_19
  simp only [e_470_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1555)) := by
  have c30252 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30253 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30254 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30255 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30256 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30257 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_0
  simp only [← c30252, ← c30253, ← c30254] at e_471_0
  have e_471_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_1
  simp only [← c30255, ← c30256, k2, ← c30257] at e_471_1
  have hr := e_471_1
  simp only [e_471_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1556)) := by
  have c30258 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30259 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30260 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30261 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30262 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30263 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_2
  simp only [← c30258, ← c30259, ← c30260] at e_471_2
  have e_471_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_3
  simp only [← c30261, ← c30262, k2, ← c30263] at e_471_3
  have hr := e_471_3
  simp only [e_471_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1557)) := by
  have c30264 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30265 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30266 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30267 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30268 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30269 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_4
  simp only [← c30264, ← c30265, ← c30266] at e_471_4
  have e_471_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_5
  simp only [← c30267, ← c30268, k2, ← c30269] at e_471_5
  have hr := e_471_5
  simp only [e_471_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1558)) := by
  have c30270 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30271 := (publicBatchWrapper89_copies945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30272 := (publicBatchWrapper89_copies946 a h).1
  have c30273 := (publicBatchWrapper89_copies946 a h).2.1
  have c30274 := (publicBatchWrapper89_copies946 a h).2.2.1
  have c30275 := (publicBatchWrapper89_copies946 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_6
  simp only [← c30270, ← c30271, ← c30272] at e_471_6
  have e_471_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_7
  simp only [← c30273, ← c30274, k2, ← c30275] at e_471_7
  have hr := e_471_7
  simp only [e_471_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1559)) := by
  have c30276 := (publicBatchWrapper89_copies946 a h).2.2.2.2.1
  have c30277 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.1
  have c30278 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.1
  have c30279 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.1
  have c30280 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.1
  have c30281 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_8
  simp only [← c30276, ← c30277, ← c30278] at e_471_8
  have e_471_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_9
  simp only [← c30279, ← c30280, k2, ← c30281] at e_471_9
  have hr := e_471_9
  simp only [e_471_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1560)) := by
  have c30282 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30283 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30284 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30285 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30286 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30287 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_10
  simp only [← c30282, ← c30283, ← c30284] at e_471_10
  have e_471_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_11
  simp only [← c30285, ← c30286, k2, ← c30287] at e_471_11
  have hr := e_471_11
  simp only [e_471_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1561)) := by
  have c30288 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30289 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30290 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30291 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30292 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30293 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_12
  simp only [← c30288, ← c30289, ← c30290] at e_471_12
  have e_471_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_13
  simp only [← c30291, ← c30292, k2, ← c30293] at e_471_13
  have hr := e_471_13
  simp only [e_471_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1562)) := by
  have c30294 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30295 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30296 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30297 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30298 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30299 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_14
  simp only [← c30294, ← c30295, ← c30296] at e_471_14
  have e_471_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_15
  simp only [← c30297, ← c30298, k2, ← c30299] at e_471_15
  have hr := e_471_15
  simp only [e_471_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1563)) := by
  have c30300 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30301 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30302 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30303 := (publicBatchWrapper89_copies946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30304 := (publicBatchWrapper89_copies947 a h).1
  have c30305 := (publicBatchWrapper89_copies947 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_16
  simp only [← c30300, ← c30301, ← c30302] at e_471_16
  have e_471_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_17
  simp only [← c30303, ← c30304, k2, ← c30305] at e_471_17
  have hr := e_471_17
  simp only [e_471_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 471 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1564)) := by
  have c30306 := (publicBatchWrapper89_copies947 a h).2.2.1
  have c30307 := (publicBatchWrapper89_copies947 a h).2.2.2.1
  have c30308 := (publicBatchWrapper89_copies947 a h).2.2.2.2.1
  have c30309 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.1
  have c30310 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.1
  have c30311 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_471_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_18
  simp only [← c30306, ← c30307, ← c30308] at e_471_18
  have e_471_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_471_19
  simp only [← c30309, ← c30310, k2, ← c30311] at e_471_19
  have hr := e_471_19
  simp only [e_471_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1565)) := by
  have c30312 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.1
  have c30313 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.1
  have c30314 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30315 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30316 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30317 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_0
  simp only [← c30312, ← c30313, ← c30314] at e_472_0
  have e_472_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_1
  simp only [← c30315, ← c30316, k2, ← c30317] at e_472_1
  have hr := e_472_1
  simp only [e_472_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1566)) := by
  have c30318 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30319 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30320 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30321 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30322 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30323 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_2
  simp only [← c30318, ← c30319, ← c30320] at e_472_2
  have e_472_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_3
  simp only [← c30321, ← c30322, k2, ← c30323] at e_472_3
  have hr := e_472_3
  simp only [e_472_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1567)) := by
  have c30324 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30325 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30326 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30327 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30328 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30329 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_4
  simp only [← c30324, ← c30325, ← c30326] at e_472_4
  have e_472_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_5
  simp only [← c30327, ← c30328, k2, ← c30329] at e_472_5
  have hr := e_472_5
  simp only [e_472_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1568)) := by
  have c30330 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30331 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30332 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30333 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30334 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30335 := (publicBatchWrapper89_copies947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_6
  simp only [← c30330, ← c30331, ← c30332] at e_472_6
  have e_472_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_7
  simp only [← c30333, ← c30334, k2, ← c30335] at e_472_7
  have hr := e_472_7
  simp only [e_472_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1569)) := by
  have c30336 := (publicBatchWrapper89_copies948 a h).1
  have c30337 := (publicBatchWrapper89_copies948 a h).2.1
  have c30338 := (publicBatchWrapper89_copies948 a h).2.2.1
  have c30339 := (publicBatchWrapper89_copies948 a h).2.2.2.1
  have c30340 := (publicBatchWrapper89_copies948 a h).2.2.2.2.1
  have c30341 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_8
  simp only [← c30336, ← c30337, ← c30338] at e_472_8
  have e_472_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_9
  simp only [← c30339, ← c30340, k2, ← c30341] at e_472_9
  have hr := e_472_9
  simp only [e_472_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1610)) := by
  have c30342 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.1
  have c30343 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.1
  have c30344 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.1
  have c30345 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.1
  have c30346 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30347 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_10
  simp only [← c30342, ← c30343, ← c30344] at e_472_10
  have e_472_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_11
  simp only [← c30345, ← c30346, k2, ← c30347] at e_472_11
  have hr := e_472_11
  simp only [e_472_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1611)) := by
  have c30348 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30349 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30350 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30351 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30352 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30353 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_12
  simp only [← c30348, ← c30349, ← c30350] at e_472_12
  have e_472_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_13
  simp only [← c30351, ← c30352, k2, ← c30353] at e_472_13
  have hr := e_472_13
  simp only [e_472_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1612)) := by
  have c30354 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30355 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30356 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30357 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30358 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30359 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_14
  simp only [← c30354, ← c30355, ← c30356] at e_472_14
  have e_472_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_15
  simp only [← c30357, ← c30358, k2, ← c30359] at e_472_15
  have hr := e_472_15
  simp only [e_472_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1613)) := by
  have c30360 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30361 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30362 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30363 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30364 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30365 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_16
  simp only [← c30360, ← c30361, ← c30362] at e_472_16
  have e_472_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_17
  simp only [← c30363, ← c30364, k2, ← c30365] at e_472_17
  have hr := e_472_17
  simp only [e_472_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 472 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1614)) := by
  have c30366 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30367 := (publicBatchWrapper89_copies948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30368 := (publicBatchWrapper89_copies949 a h).1
  have c30369 := (publicBatchWrapper89_copies949 a h).2.1
  have c30370 := (publicBatchWrapper89_copies949 a h).2.2.1
  have c30371 := (publicBatchWrapper89_copies949 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_472_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_18
  simp only [← c30366, ← c30367, ← c30368] at e_472_18
  have e_472_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_472_19
  simp only [← c30369, ← c30370, k2, ← c30371] at e_472_19
  have hr := e_472_19
  simp only [e_472_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1615)) := by
  have c30372 := (publicBatchWrapper89_copies949 a h).2.2.2.2.1
  have c30373 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.1
  have c30374 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.1
  have c30375 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.1
  have c30376 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.1
  have c30377 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_0
  simp only [← c30372, ← c30373, ← c30374] at e_473_0
  have e_473_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_1
  simp only [← c30375, ← c30376, k2, ← c30377] at e_473_1
  have hr := e_473_1
  simp only [e_473_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1616)) := by
  have c30378 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30379 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30380 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30381 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30382 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30383 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_2
  simp only [← c30378, ← c30379, ← c30380] at e_473_2
  have e_473_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_3
  simp only [← c30381, ← c30382, k2, ← c30383] at e_473_3
  have hr := e_473_3
  simp only [e_473_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1617)) := by
  have c30384 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30385 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30386 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30387 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30388 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30389 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_4
  simp only [← c30384, ← c30385, ← c30386] at e_473_4
  have e_473_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_5
  simp only [← c30387, ← c30388, k2, ← c30389] at e_473_5
  have hr := e_473_5
  simp only [e_473_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1618)) := by
  have c30390 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30391 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30392 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30393 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30394 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30395 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_6
  simp only [← c30390, ← c30391, ← c30392] at e_473_6
  have e_473_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_7
  simp only [← c30393, ← c30394, k2, ← c30395] at e_473_7
  have hr := e_473_7
  simp only [e_473_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1619)) := by
  have c30396 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30397 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30398 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30399 := (publicBatchWrapper89_copies949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30400 := (publicBatchWrapper89_copies950 a h).1
  have c30401 := (publicBatchWrapper89_copies950 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_8
  simp only [← c30396, ← c30397, ← c30398] at e_473_8
  have e_473_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_9
  simp only [← c30399, ← c30400, k2, ← c30401] at e_473_9
  have hr := e_473_9
  simp only [e_473_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1620)) := by
  have c30402 := (publicBatchWrapper89_copies950 a h).2.2.1
  have c30403 := (publicBatchWrapper89_copies950 a h).2.2.2.1
  have c30404 := (publicBatchWrapper89_copies950 a h).2.2.2.2.1
  have c30405 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.1
  have c30406 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.1
  have c30407 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_10
  simp only [← c30402, ← c30403, ← c30404] at e_473_10
  have e_473_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_11
  simp only [← c30405, ← c30406, k2, ← c30407] at e_473_11
  have hr := e_473_11
  simp only [e_473_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1621)) := by
  have c30408 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.1
  have c30409 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.1
  have c30410 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30411 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30412 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30413 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_12
  simp only [← c30408, ← c30409, ← c30410] at e_473_12
  have e_473_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_13
  simp only [← c30411, ← c30412, k2, ← c30413] at e_473_13
  have hr := e_473_13
  simp only [e_473_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1622)) := by
  have c30414 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30415 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30416 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30417 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30418 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30419 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_14
  simp only [← c30414, ← c30415, ← c30416] at e_473_14
  have e_473_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_15
  simp only [← c30417, ← c30418, k2, ← c30419] at e_473_15
  have hr := e_473_15
  simp only [e_473_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1623)) := by
  have c30420 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30421 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30422 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30423 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30424 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30425 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_16
  simp only [← c30420, ← c30421, ← c30422] at e_473_16
  have e_473_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_17
  simp only [← c30423, ← c30424, k2, ← c30425] at e_473_17
  have hr := e_473_17
  simp only [e_473_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 473 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1624)) := by
  have c30426 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30427 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30428 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30429 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30430 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30431 := (publicBatchWrapper89_copies950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_473_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_18
  simp only [← c30426, ← c30427, ← c30428] at e_473_18
  have e_473_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_473_19
  simp only [← c30429, ← c30430, k2, ← c30431] at e_473_19
  have hr := e_473_19
  simp only [e_473_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1625)) := by
  have c30432 := (publicBatchWrapper89_copies951 a h).1
  have c30433 := (publicBatchWrapper89_copies951 a h).2.1
  have c30434 := (publicBatchWrapper89_copies951 a h).2.2.1
  have c30435 := (publicBatchWrapper89_copies951 a h).2.2.2.1
  have c30436 := (publicBatchWrapper89_copies951 a h).2.2.2.2.1
  have c30437 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_0
  simp only [← c30432, ← c30433, ← c30434] at e_474_0
  have e_474_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_1
  simp only [← c30435, ← c30436, k2, ← c30437] at e_474_1
  have hr := e_474_1
  simp only [e_474_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1626)) := by
  have c30438 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.1
  have c30439 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.1
  have c30440 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.1
  have c30441 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.1
  have c30442 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30443 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_2
  simp only [← c30438, ← c30439, ← c30440] at e_474_2
  have e_474_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_3
  simp only [← c30441, ← c30442, k2, ← c30443] at e_474_3
  have hr := e_474_3
  simp only [e_474_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1627)) := by
  have c30444 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30445 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30446 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30447 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30448 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30449 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_4
  simp only [← c30444, ← c30445, ← c30446] at e_474_4
  have e_474_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_5
  simp only [← c30447, ← c30448, k2, ← c30449] at e_474_5
  have hr := e_474_5
  simp only [e_474_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1628)) := by
  have c30450 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30451 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30452 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30453 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30454 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30455 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_6
  simp only [← c30450, ← c30451, ← c30452] at e_474_6
  have e_474_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_7
  simp only [← c30453, ← c30454, k2, ← c30455] at e_474_7
  have hr := e_474_7
  simp only [e_474_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1629)) := by
  have c30456 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30457 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30458 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30459 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30460 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30461 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_8
  simp only [← c30456, ← c30457, ← c30458] at e_474_8
  have e_474_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_9
  simp only [← c30459, ← c30460, k2, ← c30461] at e_474_9
  have hr := e_474_9
  simp only [e_474_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1630)) := by
  have c30462 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30463 := (publicBatchWrapper89_copies951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30464 := (publicBatchWrapper89_copies952 a h).1
  have c30465 := (publicBatchWrapper89_copies952 a h).2.1
  have c30466 := (publicBatchWrapper89_copies952 a h).2.2.1
  have c30467 := (publicBatchWrapper89_copies952 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_10
  simp only [← c30462, ← c30463, ← c30464] at e_474_10
  have e_474_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_11
  simp only [← c30465, ← c30466, k2, ← c30467] at e_474_11
  have hr := e_474_11
  simp only [e_474_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1631)) := by
  have c30468 := (publicBatchWrapper89_copies952 a h).2.2.2.2.1
  have c30469 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.1
  have c30470 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.1
  have c30471 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.1
  have c30472 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.1
  have c30473 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_12
  simp only [← c30468, ← c30469, ← c30470] at e_474_12
  have e_474_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_13
  simp only [← c30471, ← c30472, k2, ← c30473] at e_474_13
  have hr := e_474_13
  simp only [e_474_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1632)) := by
  have c30474 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30475 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30476 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30477 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30478 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30479 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_14
  simp only [← c30474, ← c30475, ← c30476] at e_474_14
  have e_474_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_15
  simp only [← c30477, ← c30478, k2, ← c30479] at e_474_15
  have hr := e_474_15
  simp only [e_474_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1633)) := by
  have c30480 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30481 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30482 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30483 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30484 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30485 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_16
  simp only [← c30480, ← c30481, ← c30482] at e_474_16
  have e_474_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_17
  simp only [← c30483, ← c30484, k2, ← c30485] at e_474_17
  have hr := e_474_17
  simp only [e_474_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 474 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1634)) := by
  have c30486 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30487 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30488 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30489 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30490 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30491 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_474_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_18
  simp only [← c30486, ← c30487, ← c30488] at e_474_18
  have e_474_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_474_19
  simp only [← c30489, ← c30490, k2, ← c30491] at e_474_19
  have hr := e_474_19
  simp only [e_474_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1635)) := by
  have c30492 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30493 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30494 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30495 := (publicBatchWrapper89_copies952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30496 := (publicBatchWrapper89_copies953 a h).1
  have c30497 := (publicBatchWrapper89_copies953 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_0
  simp only [← c30492, ← c30493, ← c30494] at e_475_0
  have e_475_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_1
  simp only [← c30495, ← c30496, k2, ← c30497] at e_475_1
  have hr := e_475_1
  simp only [e_475_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1636)) := by
  have c30498 := (publicBatchWrapper89_copies953 a h).2.2.1
  have c30499 := (publicBatchWrapper89_copies953 a h).2.2.2.1
  have c30500 := (publicBatchWrapper89_copies953 a h).2.2.2.2.1
  have c30501 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.1
  have c30502 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.1
  have c30503 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_2
  simp only [← c30498, ← c30499, ← c30500] at e_475_2
  have e_475_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_3
  simp only [← c30501, ← c30502, k2, ← c30503] at e_475_3
  have hr := e_475_3
  simp only [e_475_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1637)) := by
  have c30504 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.1
  have c30505 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.1
  have c30506 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30507 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30508 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30509 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_4
  simp only [← c30504, ← c30505, ← c30506] at e_475_4
  have e_475_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_5
  simp only [← c30507, ← c30508, k2, ← c30509] at e_475_5
  have hr := e_475_5
  simp only [e_475_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1638)) := by
  have c30510 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30511 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30512 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30513 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30514 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30515 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_6
  simp only [← c30510, ← c30511, ← c30512] at e_475_6
  have e_475_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_7
  simp only [← c30513, ← c30514, k2, ← c30515] at e_475_7
  have hr := e_475_7
  simp only [e_475_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1639)) := by
  have c30516 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30517 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30518 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30519 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30520 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30521 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_8
  simp only [← c30516, ← c30517, ← c30518] at e_475_8
  have e_475_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_9
  simp only [← c30519, ← c30520, k2, ← c30521] at e_475_9
  have hr := e_475_9
  simp only [e_475_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1640)) := by
  have c30522 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30523 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30524 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30525 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30526 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30527 := (publicBatchWrapper89_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_10
  simp only [← c30522, ← c30523, ← c30524] at e_475_10
  have e_475_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_11
  simp only [← c30525, ← c30526, k2, ← c30527] at e_475_11
  have hr := e_475_11
  simp only [e_475_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1641)) := by
  have c30528 := (publicBatchWrapper89_copies954 a h).1
  have c30529 := (publicBatchWrapper89_copies954 a h).2.1
  have c30530 := (publicBatchWrapper89_copies954 a h).2.2.1
  have c30531 := (publicBatchWrapper89_copies954 a h).2.2.2.1
  have c30532 := (publicBatchWrapper89_copies954 a h).2.2.2.2.1
  have c30533 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_12
  simp only [← c30528, ← c30529, ← c30530] at e_475_12
  have e_475_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_13
  simp only [← c30531, ← c30532, k2, ← c30533] at e_475_13
  have hr := e_475_13
  simp only [e_475_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1642)) := by
  have c30534 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.1
  have c30535 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.1
  have c30536 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.1
  have c30537 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.1
  have c30538 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30539 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_14
  simp only [← c30534, ← c30535, ← c30536] at e_475_14
  have e_475_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_15
  simp only [← c30537, ← c30538, k2, ← c30539] at e_475_15
  have hr := e_475_15
  simp only [e_475_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1643)) := by
  have c30540 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30541 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30542 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30543 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30544 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30545 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_16
  simp only [← c30540, ← c30541, ← c30542] at e_475_16
  have e_475_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_17
  simp only [← c30543, ← c30544, k2, ← c30545] at e_475_17
  have hr := e_475_17
  simp only [e_475_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 475 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1644)) := by
  have c30546 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30547 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30548 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30549 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30550 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30551 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_475_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_18
  simp only [← c30546, ← c30547, ← c30548] at e_475_18
  have e_475_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_475_19
  simp only [← c30549, ← c30550, k2, ← c30551] at e_475_19
  have hr := e_475_19
  simp only [e_475_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1645)) := by
  have c30552 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30553 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30554 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30555 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30556 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30557 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_0
  simp only [← c30552, ← c30553, ← c30554] at e_476_0
  have e_476_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_1
  simp only [← c30555, ← c30556, k2, ← c30557] at e_476_1
  have hr := e_476_1
  simp only [e_476_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1646)) := by
  have c30558 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30559 := (publicBatchWrapper89_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30560 := (publicBatchWrapper89_copies955 a h).1
  have c30561 := (publicBatchWrapper89_copies955 a h).2.1
  have c30562 := (publicBatchWrapper89_copies955 a h).2.2.1
  have c30563 := (publicBatchWrapper89_copies955 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_2
  simp only [← c30558, ← c30559, ← c30560] at e_476_2
  have e_476_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_3
  simp only [← c30561, ← c30562, k2, ← c30563] at e_476_3
  have hr := e_476_3
  simp only [e_476_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1647)) := by
  have c30564 := (publicBatchWrapper89_copies955 a h).2.2.2.2.1
  have c30565 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.1
  have c30566 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.1
  have c30567 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.1
  have c30568 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.1
  have c30569 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_4
  simp only [← c30564, ← c30565, ← c30566] at e_476_4
  have e_476_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_5
  simp only [← c30567, ← c30568, k2, ← c30569] at e_476_5
  have hr := e_476_5
  simp only [e_476_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1648)) := by
  have c30570 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30571 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30572 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30573 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30574 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30575 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_6
  simp only [← c30570, ← c30571, ← c30572] at e_476_6
  have e_476_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_7
  simp only [← c30573, ← c30574, k2, ← c30575] at e_476_7
  have hr := e_476_7
  simp only [e_476_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1649)) := by
  have c30576 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30577 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30578 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30579 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30580 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30581 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_8
  simp only [← c30576, ← c30577, ← c30578] at e_476_8
  have e_476_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_9
  simp only [← c30579, ← c30580, k2, ← c30581] at e_476_9
  have hr := e_476_9
  simp only [e_476_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1650)) := by
  have c30582 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30583 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30584 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30585 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30586 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30587 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_10
  simp only [← c30582, ← c30583, ← c30584] at e_476_10
  have e_476_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_11
  simp only [← c30585, ← c30586, k2, ← c30587] at e_476_11
  have hr := e_476_11
  simp only [e_476_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1651)) := by
  have c30588 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30589 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30590 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30591 := (publicBatchWrapper89_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30592 := (publicBatchWrapper89_copies956 a h).1
  have c30593 := (publicBatchWrapper89_copies956 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_12
  simp only [← c30588, ← c30589, ← c30590] at e_476_12
  have e_476_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_13
  simp only [← c30591, ← c30592, k2, ← c30593] at e_476_13
  have hr := e_476_13
  simp only [e_476_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1652)) := by
  have c30594 := (publicBatchWrapper89_copies956 a h).2.2.1
  have c30595 := (publicBatchWrapper89_copies956 a h).2.2.2.1
  have c30596 := (publicBatchWrapper89_copies956 a h).2.2.2.2.1
  have c30597 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.1
  have c30598 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.1
  have c30599 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_14
  simp only [← c30594, ← c30595, ← c30596] at e_476_14
  have e_476_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_15
  simp only [← c30597, ← c30598, k2, ← c30599] at e_476_15
  have hr := e_476_15
  simp only [e_476_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1653)) := by
  have c30600 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.1
  have c30601 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.1
  have c30602 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30603 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30604 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30605 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_16
  simp only [← c30600, ← c30601, ← c30602] at e_476_16
  have e_476_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_17
  simp only [← c30603, ← c30604, k2, ← c30605] at e_476_17
  have hr := e_476_17
  simp only [e_476_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f4350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 476 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1654)) := by
  have c30606 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30607 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30608 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30609 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30610 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30611 := (publicBatchWrapper89_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_476_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_18
  simp only [← c30606, ← c30607, ← c30608] at e_476_18
  have e_476_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_476_19
  simp only [← c30609, ← c30610, k2, ← c30611] at e_476_19
  have hr := e_476_19
  simp only [e_476_18] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
