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

theorem publicBatchWrapper89_f14404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14228)) := by
  have c90930 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90931 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90932 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90933 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90934 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90935 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_6
  simp only [← c90930, ← c90931, ← c90932] at e_1482_6
  have e_1482_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_7
  simp only [← c90933, ← c90934, k2, ← c90935] at e_1482_7
  have hr := e_1482_7
  simp only [e_1482_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14229)) := by
  have c90936 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90937 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90938 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90939 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90940 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90941 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_8
  simp only [← c90936, ← c90937, ← c90938] at e_1482_8
  have e_1482_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_9
  simp only [← c90939, ← c90940, k2, ← c90941] at e_1482_9
  have hr := e_1482_9
  simp only [e_1482_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14230)) := by
  have c90942 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90943 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90944 := (publicBatchWrapper89_copies2842 a h).1
  have c90945 := (publicBatchWrapper89_copies2842 a h).2.1
  have c90946 := (publicBatchWrapper89_copies2842 a h).2.2.1
  have c90947 := (publicBatchWrapper89_copies2842 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_10
  simp only [← c90942, ← c90943, ← c90944] at e_1482_10
  have e_1482_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_11
  simp only [← c90945, ← c90946, k2, ← c90947] at e_1482_11
  have hr := e_1482_11
  simp only [e_1482_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14231)) := by
  have c90948 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.1
  have c90949 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.1
  have c90950 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.1
  have c90951 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.1
  have c90952 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.1
  have c90953 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_12
  simp only [← c90948, ← c90949, ← c90950] at e_1482_12
  have e_1482_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_13
  simp only [← c90951, ← c90952, k2, ← c90953] at e_1482_13
  have hr := e_1482_13
  simp only [e_1482_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14232)) := by
  have c90954 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90955 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90956 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90957 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90958 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90959 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_14
  simp only [← c90954, ← c90955, ← c90956] at e_1482_14
  have e_1482_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_15
  simp only [← c90957, ← c90958, k2, ← c90959] at e_1482_15
  have hr := e_1482_15
  simp only [e_1482_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14233)) := by
  have c90960 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90961 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90962 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90963 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90964 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90965 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_16
  simp only [← c90960, ← c90961, ← c90962] at e_1482_16
  have e_1482_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_17
  simp only [← c90963, ← c90964, k2, ← c90965] at e_1482_17
  have hr := e_1482_17
  simp only [e_1482_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14234)) := by
  have c90966 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90967 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90968 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90969 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90970 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90971 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_18
  simp only [← c90966, ← c90967, ← c90968] at e_1482_18
  have e_1482_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_19
  simp only [← c90969, ← c90970, k2, ← c90971] at e_1482_19
  have hr := e_1482_19
  simp only [e_1482_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14235)) := by
  have c90972 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90973 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90974 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90975 := (publicBatchWrapper89_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90976 := (publicBatchWrapper89_copies2843 a h).1
  have c90977 := (publicBatchWrapper89_copies2843 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_0
  simp only [← c90972, ← c90973, ← c90974] at e_1483_0
  have e_1483_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_1
  simp only [← c90975, ← c90976, k2, ← c90977] at e_1483_1
  have hr := e_1483_1
  simp only [e_1483_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14236)) := by
  have c90978 := (publicBatchWrapper89_copies2843 a h).2.2.1
  have c90979 := (publicBatchWrapper89_copies2843 a h).2.2.2.1
  have c90980 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.1
  have c90981 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.1
  have c90982 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.1
  have c90983 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_2
  simp only [← c90978, ← c90979, ← c90980] at e_1483_2
  have e_1483_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_3
  simp only [← c90981, ← c90982, k2, ← c90983] at e_1483_3
  have hr := e_1483_3
  simp only [e_1483_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14237)) := by
  have c90984 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.1
  have c90985 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.1
  have c90986 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90987 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90988 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90989 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_4
  simp only [← c90984, ← c90985, ← c90986] at e_1483_4
  have e_1483_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_5
  simp only [← c90987, ← c90988, k2, ← c90989] at e_1483_5
  have hr := e_1483_5
  simp only [e_1483_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14238)) := by
  have c90990 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90991 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90992 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90993 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90994 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90995 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_6
  simp only [← c90990, ← c90991, ← c90992] at e_1483_6
  have e_1483_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_7
  simp only [← c90993, ← c90994, k2, ← c90995] at e_1483_7
  have hr := e_1483_7
  simp only [e_1483_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14239)) := by
  have c90996 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90997 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90998 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90999 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91000 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91001 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_8
  simp only [← c90996, ← c90997, ← c90998] at e_1483_8
  have e_1483_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_9
  simp only [← c90999, ← c91000, k2, ← c91001] at e_1483_9
  have hr := e_1483_9
  simp only [e_1483_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14240)) := by
  have c91002 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91003 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91004 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91005 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91006 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91007 := (publicBatchWrapper89_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_10
  simp only [← c91002, ← c91003, ← c91004] at e_1483_10
  have e_1483_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_11
  simp only [← c91005, ← c91006, k2, ← c91007] at e_1483_11
  have hr := e_1483_11
  simp only [e_1483_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14241)) := by
  have c91008 := (publicBatchWrapper89_copies2844 a h).1
  have c91009 := (publicBatchWrapper89_copies2844 a h).2.1
  have c91010 := (publicBatchWrapper89_copies2844 a h).2.2.1
  have c91011 := (publicBatchWrapper89_copies2844 a h).2.2.2.1
  have c91012 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.1
  have c91013 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_12
  simp only [← c91008, ← c91009, ← c91010] at e_1483_12
  have e_1483_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_13
  simp only [← c91011, ← c91012, k2, ← c91013] at e_1483_13
  have hr := e_1483_13
  simp only [e_1483_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14242)) := by
  have c91014 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.1
  have c91015 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.1
  have c91016 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.1
  have c91017 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.1
  have c91018 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91019 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_14
  simp only [← c91014, ← c91015, ← c91016] at e_1483_14
  have e_1483_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_15
  simp only [← c91017, ← c91018, k2, ← c91019] at e_1483_15
  have hr := e_1483_15
  simp only [e_1483_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14243)) := by
  have c91020 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91021 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91022 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91023 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91024 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91025 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_16
  simp only [← c91020, ← c91021, ← c91022] at e_1483_16
  have e_1483_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_17
  simp only [← c91023, ← c91024, k2, ← c91025] at e_1483_17
  have hr := e_1483_17
  simp only [e_1483_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1483 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14244)) := by
  have c91026 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91027 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91028 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91029 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91030 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91031 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1483_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_18
  simp only [← c91026, ← c91027, ← c91028] at e_1483_18
  have e_1483_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_19
  simp only [← c91029, ← c91030, k2, ← c91031] at e_1483_19
  have hr := e_1483_19
  simp only [e_1483_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14245)) := by
  have c91032 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91033 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91034 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91035 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91036 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91037 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_0
  simp only [← c91032, ← c91033, ← c91034] at e_1484_0
  have e_1484_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_1
  simp only [← c91035, ← c91036, k2, ← c91037] at e_1484_1
  have hr := e_1484_1
  simp only [e_1484_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14246)) := by
  have c91038 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91039 := (publicBatchWrapper89_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91040 := (publicBatchWrapper89_copies2845 a h).1
  have c91041 := (publicBatchWrapper89_copies2845 a h).2.1
  have c91042 := (publicBatchWrapper89_copies2845 a h).2.2.1
  have c91043 := (publicBatchWrapper89_copies2845 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_2
  simp only [← c91038, ← c91039, ← c91040] at e_1484_2
  have e_1484_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_3
  simp only [← c91041, ← c91042, k2, ← c91043] at e_1484_3
  have hr := e_1484_3
  simp only [e_1484_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14247)) := by
  have c91044 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.1
  have c91045 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.1
  have c91046 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.1
  have c91047 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.1
  have c91048 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.1
  have c91049 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_4
  simp only [← c91044, ← c91045, ← c91046] at e_1484_4
  have e_1484_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_5
  simp only [← c91047, ← c91048, k2, ← c91049] at e_1484_5
  have hr := e_1484_5
  simp only [e_1484_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14248)) := by
  have c91050 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91051 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91052 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91053 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91054 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91055 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_6
  simp only [← c91050, ← c91051, ← c91052] at e_1484_6
  have e_1484_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_7
  simp only [← c91053, ← c91054, k2, ← c91055] at e_1484_7
  have hr := e_1484_7
  simp only [e_1484_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14249)) := by
  have c91056 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91057 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91058 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91059 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91060 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91061 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_8
  simp only [← c91056, ← c91057, ← c91058] at e_1484_8
  have e_1484_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_9
  simp only [← c91059, ← c91060, k2, ← c91061] at e_1484_9
  have hr := e_1484_9
  simp only [e_1484_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14250)) := by
  have c91062 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91063 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91064 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91065 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91066 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91067 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_10
  simp only [← c91062, ← c91063, ← c91064] at e_1484_10
  have e_1484_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_11
  simp only [← c91065, ← c91066, k2, ← c91067] at e_1484_11
  have hr := e_1484_11
  simp only [e_1484_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14251)) := by
  have c91068 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91069 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91070 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91071 := (publicBatchWrapper89_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91072 := (publicBatchWrapper89_copies2846 a h).1
  have c91073 := (publicBatchWrapper89_copies2846 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_12
  simp only [← c91068, ← c91069, ← c91070] at e_1484_12
  have e_1484_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_13
  simp only [← c91071, ← c91072, k2, ← c91073] at e_1484_13
  have hr := e_1484_13
  simp only [e_1484_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14252)) := by
  have c91074 := (publicBatchWrapper89_copies2846 a h).2.2.1
  have c91075 := (publicBatchWrapper89_copies2846 a h).2.2.2.1
  have c91076 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.1
  have c91077 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.1
  have c91078 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.1
  have c91079 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_14
  simp only [← c91074, ← c91075, ← c91076] at e_1484_14
  have e_1484_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_15
  simp only [← c91077, ← c91078, k2, ← c91079] at e_1484_15
  have hr := e_1484_15
  simp only [e_1484_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14253)) := by
  have c91080 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.1
  have c91081 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.1
  have c91082 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91083 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91084 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91085 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_16
  simp only [← c91080, ← c91081, ← c91082] at e_1484_16
  have e_1484_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_17
  simp only [← c91083, ← c91084, k2, ← c91085] at e_1484_17
  have hr := e_1484_17
  simp only [e_1484_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1484 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14254)) := by
  have c91086 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91087 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91088 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91089 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91090 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91091 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1484_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_18
  simp only [← c91086, ← c91087, ← c91088] at e_1484_18
  have e_1484_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_19
  simp only [← c91089, ← c91090, k2, ← c91091] at e_1484_19
  have hr := e_1484_19
  simp only [e_1484_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14255)) := by
  have c91092 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91093 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91094 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91095 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91096 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91097 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_0
  simp only [← c91092, ← c91093, ← c91094] at e_1485_0
  have e_1485_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_1
  simp only [← c91095, ← c91096, k2, ← c91097] at e_1485_1
  have hr := e_1485_1
  simp only [e_1485_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14256)) := by
  have c91098 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91099 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91100 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91101 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91102 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91103 := (publicBatchWrapper89_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_2
  simp only [← c91098, ← c91099, ← c91100] at e_1485_2
  have e_1485_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_3
  simp only [← c91101, ← c91102, k2, ← c91103] at e_1485_3
  have hr := e_1485_3
  simp only [e_1485_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14257)) := by
  have c91104 := (publicBatchWrapper89_copies2847 a h).1
  have c91105 := (publicBatchWrapper89_copies2847 a h).2.1
  have c91106 := (publicBatchWrapper89_copies2847 a h).2.2.1
  have c91107 := (publicBatchWrapper89_copies2847 a h).2.2.2.1
  have c91108 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.1
  have c91109 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_4
  simp only [← c91104, ← c91105, ← c91106] at e_1485_4
  have e_1485_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_5
  simp only [← c91107, ← c91108, k2, ← c91109] at e_1485_5
  have hr := e_1485_5
  simp only [e_1485_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14258)) := by
  have c91110 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.1
  have c91111 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.1
  have c91112 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.1
  have c91113 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.1
  have c91114 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91115 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_6
  simp only [← c91110, ← c91111, ← c91112] at e_1485_6
  have e_1485_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_7
  simp only [← c91113, ← c91114, k2, ← c91115] at e_1485_7
  have hr := e_1485_7
  simp only [e_1485_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14259)) := by
  have c91116 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91117 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91118 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91119 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91120 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91121 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_8
  simp only [← c91116, ← c91117, ← c91118] at e_1485_8
  have e_1485_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_9
  simp only [← c91119, ← c91120, k2, ← c91121] at e_1485_9
  have hr := e_1485_9
  simp only [e_1485_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14260)) := by
  have c91122 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91123 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91124 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91125 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91126 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91127 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_10
  simp only [← c91122, ← c91123, ← c91124] at e_1485_10
  have e_1485_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_11
  simp only [← c91125, ← c91126, k2, ← c91127] at e_1485_11
  have hr := e_1485_11
  simp only [e_1485_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14261)) := by
  have c91128 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91129 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91130 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91131 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91132 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91133 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_12
  simp only [← c91128, ← c91129, ← c91130] at e_1485_12
  have e_1485_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_13
  simp only [← c91131, ← c91132, k2, ← c91133] at e_1485_13
  have hr := e_1485_13
  simp only [e_1485_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14262)) := by
  have c91134 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91135 := (publicBatchWrapper89_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91136 := (publicBatchWrapper89_copies2848 a h).1
  have c91137 := (publicBatchWrapper89_copies2848 a h).2.1
  have c91138 := (publicBatchWrapper89_copies2848 a h).2.2.1
  have c91139 := (publicBatchWrapper89_copies2848 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_14
  simp only [← c91134, ← c91135, ← c91136] at e_1485_14
  have e_1485_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_15
  simp only [← c91137, ← c91138, k2, ← c91139] at e_1485_15
  have hr := e_1485_15
  simp only [e_1485_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14263)) := by
  have c91140 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.1
  have c91141 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.1
  have c91142 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.1
  have c91143 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.1
  have c91144 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.1
  have c91145 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_16
  simp only [← c91140, ← c91141, ← c91142] at e_1485_16
  have e_1485_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_17
  simp only [← c91143, ← c91144, k2, ← c91145] at e_1485_17
  have hr := e_1485_17
  simp only [e_1485_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1485 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14264)) := by
  have c91146 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91147 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91148 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91149 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91150 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91151 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1485_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_18
  simp only [← c91146, ← c91147, ← c91148] at e_1485_18
  have e_1485_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_19
  simp only [← c91149, ← c91150, k2, ← c91151] at e_1485_19
  have hr := e_1485_19
  simp only [e_1485_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14265)) := by
  have c91152 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91153 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91154 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91155 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91156 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91157 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_0
  simp only [← c91152, ← c91153, ← c91154] at e_1486_0
  have e_1486_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_1
  simp only [← c91155, ← c91156, k2, ← c91157] at e_1486_1
  have hr := e_1486_1
  simp only [e_1486_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14266)) := by
  have c91158 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91159 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91160 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91161 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91162 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91163 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_2
  simp only [← c91158, ← c91159, ← c91160] at e_1486_2
  have e_1486_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_3
  simp only [← c91161, ← c91162, k2, ← c91163] at e_1486_3
  have hr := e_1486_3
  simp only [e_1486_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14267)) := by
  have c91164 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91165 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91166 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91167 := (publicBatchWrapper89_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91168 := (publicBatchWrapper89_copies2849 a h).1
  have c91169 := (publicBatchWrapper89_copies2849 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_4
  simp only [← c91164, ← c91165, ← c91166] at e_1486_4
  have e_1486_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_5
  simp only [← c91167, ← c91168, k2, ← c91169] at e_1486_5
  have hr := e_1486_5
  simp only [e_1486_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14268)) := by
  have c91170 := (publicBatchWrapper89_copies2849 a h).2.2.1
  have c91171 := (publicBatchWrapper89_copies2849 a h).2.2.2.1
  have c91172 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.1
  have c91173 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.1
  have c91174 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.1
  have c91175 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_6
  simp only [← c91170, ← c91171, ← c91172] at e_1486_6
  have e_1486_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_7
  simp only [← c91173, ← c91174, k2, ← c91175] at e_1486_7
  have hr := e_1486_7
  simp only [e_1486_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14269)) := by
  have c91176 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.1
  have c91177 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.1
  have c91178 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91179 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91180 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91181 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_8
  simp only [← c91176, ← c91177, ← c91178] at e_1486_8
  have e_1486_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_9
  simp only [← c91179, ← c91180, k2, ← c91181] at e_1486_9
  have hr := e_1486_9
  simp only [e_1486_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14270)) := by
  have c91182 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91183 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91184 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91185 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91186 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91187 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_10
  simp only [← c91182, ← c91183, ← c91184] at e_1486_10
  have e_1486_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_11
  simp only [← c91185, ← c91186, k2, ← c91187] at e_1486_11
  have hr := e_1486_11
  simp only [e_1486_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14271)) := by
  have c91188 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91189 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91190 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91191 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91192 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91193 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_12
  simp only [← c91188, ← c91189, ← c91190] at e_1486_12
  have e_1486_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_13
  simp only [← c91191, ← c91192, k2, ← c91193] at e_1486_13
  have hr := e_1486_13
  simp only [e_1486_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14272)) := by
  have c91194 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91195 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91196 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91197 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91198 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91199 := (publicBatchWrapper89_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_14
  simp only [← c91194, ← c91195, ← c91196] at e_1486_14
  have e_1486_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_15
  simp only [← c91197, ← c91198, k2, ← c91199] at e_1486_15
  have hr := e_1486_15
  simp only [e_1486_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14273)) := by
  have c91200 := (publicBatchWrapper89_copies2850 a h).1
  have c91201 := (publicBatchWrapper89_copies2850 a h).2.1
  have c91202 := (publicBatchWrapper89_copies2850 a h).2.2.1
  have c91203 := (publicBatchWrapper89_copies2850 a h).2.2.2.1
  have c91204 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.1
  have c91205 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_16
  simp only [← c91200, ← c91201, ← c91202] at e_1486_16
  have e_1486_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_17
  simp only [← c91203, ← c91204, k2, ← c91205] at e_1486_17
  have hr := e_1486_17
  simp only [e_1486_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1486 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14274)) := by
  have c91206 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.1
  have c91207 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.1
  have c91208 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.1
  have c91209 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.1
  have c91210 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91211 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1486_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_18
  simp only [← c91206, ← c91207, ← c91208] at e_1486_18
  have e_1486_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_19
  simp only [← c91209, ← c91210, k2, ← c91211] at e_1486_19
  have hr := e_1486_19
  simp only [e_1486_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14275)) := by
  have c91212 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91213 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91214 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91215 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91216 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91217 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_0
  simp only [← c91212, ← c91213, ← c91214] at e_1487_0
  have e_1487_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_1
  simp only [← c91215, ← c91216, k2, ← c91217] at e_1487_1
  have hr := e_1487_1
  simp only [e_1487_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14276)) := by
  have c91218 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91219 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91220 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91221 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91222 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91223 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_2
  simp only [← c91218, ← c91219, ← c91220] at e_1487_2
  have e_1487_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_3
  simp only [← c91221, ← c91222, k2, ← c91223] at e_1487_3
  have hr := e_1487_3
  simp only [e_1487_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14277)) := by
  have c91224 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91225 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91226 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91227 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91228 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91229 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_4
  simp only [← c91224, ← c91225, ← c91226] at e_1487_4
  have e_1487_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_5
  simp only [← c91227, ← c91228, k2, ← c91229] at e_1487_5
  have hr := e_1487_5
  simp only [e_1487_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14278)) := by
  have c91230 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91231 := (publicBatchWrapper89_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91232 := (publicBatchWrapper89_copies2851 a h).1
  have c91233 := (publicBatchWrapper89_copies2851 a h).2.1
  have c91234 := (publicBatchWrapper89_copies2851 a h).2.2.1
  have c91235 := (publicBatchWrapper89_copies2851 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_6
  simp only [← c91230, ← c91231, ← c91232] at e_1487_6
  have e_1487_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_7
  simp only [← c91233, ← c91234, k2, ← c91235] at e_1487_7
  have hr := e_1487_7
  simp only [e_1487_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14279)) := by
  have c91236 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.1
  have c91237 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.1
  have c91238 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.1
  have c91239 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.1
  have c91240 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.1
  have c91241 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_8
  simp only [← c91236, ← c91237, ← c91238] at e_1487_8
  have e_1487_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_9
  simp only [← c91239, ← c91240, k2, ← c91241] at e_1487_9
  have hr := e_1487_9
  simp only [e_1487_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14280)) := by
  have c91242 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91243 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91244 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91245 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91246 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91247 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_10
  simp only [← c91242, ← c91243, ← c91244] at e_1487_10
  have e_1487_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_11
  simp only [← c91245, ← c91246, k2, ← c91247] at e_1487_11
  have hr := e_1487_11
  simp only [e_1487_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14281)) := by
  have c91248 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91249 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91250 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91251 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91252 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91253 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_12
  simp only [← c91248, ← c91249, ← c91250] at e_1487_12
  have e_1487_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_13
  simp only [← c91251, ← c91252, k2, ← c91253] at e_1487_13
  have hr := e_1487_13
  simp only [e_1487_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14282)) := by
  have c91254 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91255 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91256 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91257 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91258 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91259 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_14
  simp only [← c91254, ← c91255, ← c91256] at e_1487_14
  have e_1487_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_15
  simp only [← c91257, ← c91258, k2, ← c91259] at e_1487_15
  have hr := e_1487_15
  simp only [e_1487_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14283)) := by
  have c91260 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91261 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91262 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91263 := (publicBatchWrapper89_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91264 := (publicBatchWrapper89_copies2852 a h).1
  have c91265 := (publicBatchWrapper89_copies2852 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_16
  simp only [← c91260, ← c91261, ← c91262] at e_1487_16
  have e_1487_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_17
  simp only [← c91263, ← c91264, k2, ← c91265] at e_1487_17
  have hr := e_1487_17
  simp only [e_1487_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1487 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14284)) := by
  have c91266 := (publicBatchWrapper89_copies2852 a h).2.2.1
  have c91267 := (publicBatchWrapper89_copies2852 a h).2.2.2.1
  have c91268 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.1
  have c91269 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.1
  have c91270 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.1
  have c91271 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1487_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_18
  simp only [← c91266, ← c91267, ← c91268] at e_1487_18
  have e_1487_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_19
  simp only [← c91269, ← c91270, k2, ← c91271] at e_1487_19
  have hr := e_1487_19
  simp only [e_1487_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14285)) := by
  have c91272 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.1
  have c91273 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.1
  have c91274 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91275 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91276 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91277 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_0
  simp only [← c91272, ← c91273, ← c91274] at e_1488_0
  have e_1488_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_1
  simp only [← c91275, ← c91276, k2, ← c91277] at e_1488_1
  have hr := e_1488_1
  simp only [e_1488_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14286)) := by
  have c91278 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91279 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91280 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91281 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91282 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91283 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_2
  simp only [← c91278, ← c91279, ← c91280] at e_1488_2
  have e_1488_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_3
  simp only [← c91281, ← c91282, k2, ← c91283] at e_1488_3
  have hr := e_1488_3
  simp only [e_1488_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14287)) := by
  have c91284 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91285 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91286 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91287 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91288 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91289 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_4
  simp only [← c91284, ← c91285, ← c91286] at e_1488_4
  have e_1488_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_5
  simp only [← c91287, ← c91288, k2, ← c91289] at e_1488_5
  have hr := e_1488_5
  simp only [e_1488_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14288)) := by
  have c91290 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91291 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91292 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91293 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91294 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91295 := (publicBatchWrapper89_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_6
  simp only [← c91290, ← c91291, ← c91292] at e_1488_6
  have e_1488_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_7
  simp only [← c91293, ← c91294, k2, ← c91295] at e_1488_7
  have hr := e_1488_7
  simp only [e_1488_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14289)) := by
  have c91296 := (publicBatchWrapper89_copies2853 a h).1
  have c91297 := (publicBatchWrapper89_copies2853 a h).2.1
  have c91298 := (publicBatchWrapper89_copies2853 a h).2.2.1
  have c91299 := (publicBatchWrapper89_copies2853 a h).2.2.2.1
  have c91300 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.1
  have c91301 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_8
  simp only [← c91296, ← c91297, ← c91298] at e_1488_8
  have e_1488_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_9
  simp only [← c91299, ← c91300, k2, ← c91301] at e_1488_9
  have hr := e_1488_9
  simp only [e_1488_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14290)) := by
  have c91302 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.1
  have c91303 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.1
  have c91304 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.1
  have c91305 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.1
  have c91306 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91307 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_10
  simp only [← c91302, ← c91303, ← c91304] at e_1488_10
  have e_1488_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_11
  simp only [← c91305, ← c91306, k2, ← c91307] at e_1488_11
  have hr := e_1488_11
  simp only [e_1488_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14291)) := by
  have c91308 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91309 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91310 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91311 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91312 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91313 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_12
  simp only [← c91308, ← c91309, ← c91310] at e_1488_12
  have e_1488_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_13
  simp only [← c91311, ← c91312, k2, ← c91313] at e_1488_13
  have hr := e_1488_13
  simp only [e_1488_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14292)) := by
  have c91314 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91315 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91316 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91317 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91318 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91319 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_14
  simp only [← c91314, ← c91315, ← c91316] at e_1488_14
  have e_1488_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_15
  simp only [← c91317, ← c91318, k2, ← c91319] at e_1488_15
  have hr := e_1488_15
  simp only [e_1488_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14293)) := by
  have c91320 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91321 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91322 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91323 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91324 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91325 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_16
  simp only [← c91320, ← c91321, ← c91322] at e_1488_16
  have e_1488_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_17
  simp only [← c91323, ← c91324, k2, ← c91325] at e_1488_17
  have hr := e_1488_17
  simp only [e_1488_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1488 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14294)) := by
  have c91326 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91327 := (publicBatchWrapper89_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91328 := (publicBatchWrapper89_copies2854 a h).1
  have c91329 := (publicBatchWrapper89_copies2854 a h).2.1
  have c91330 := (publicBatchWrapper89_copies2854 a h).2.2.1
  have c91331 := (publicBatchWrapper89_copies2854 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1488_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_18
  simp only [← c91326, ← c91327, ← c91328] at e_1488_18
  have e_1488_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_19
  simp only [← c91329, ← c91330, k2, ← c91331] at e_1488_19
  have hr := e_1488_19
  simp only [e_1488_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14295)) := by
  have c91332 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.1
  have c91333 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.1
  have c91334 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.1
  have c91335 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.1
  have c91336 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.1
  have c91337 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_0
  simp only [← c91332, ← c91333, ← c91334] at e_1489_0
  have e_1489_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_1
  simp only [← c91335, ← c91336, k2, ← c91337] at e_1489_1
  have hr := e_1489_1
  simp only [e_1489_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14296)) := by
  have c91338 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91339 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91340 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91341 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91342 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91343 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_2
  simp only [← c91338, ← c91339, ← c91340] at e_1489_2
  have e_1489_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_3
  simp only [← c91341, ← c91342, k2, ← c91343] at e_1489_3
  have hr := e_1489_3
  simp only [e_1489_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14297)) := by
  have c91344 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91345 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91346 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91347 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91348 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91349 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_4
  simp only [← c91344, ← c91345, ← c91346] at e_1489_4
  have e_1489_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_5
  simp only [← c91347, ← c91348, k2, ← c91349] at e_1489_5
  have hr := e_1489_5
  simp only [e_1489_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14298)) := by
  have c91350 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91351 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91352 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91353 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91354 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91355 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_6
  simp only [← c91350, ← c91351, ← c91352] at e_1489_6
  have e_1489_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_7
  simp only [← c91353, ← c91354, k2, ← c91355] at e_1489_7
  have hr := e_1489_7
  simp only [e_1489_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14299)) := by
  have c91356 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91357 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91358 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91359 := (publicBatchWrapper89_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91360 := (publicBatchWrapper89_copies2855 a h).1
  have c91361 := (publicBatchWrapper89_copies2855 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_8
  simp only [← c91356, ← c91357, ← c91358] at e_1489_8
  have e_1489_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_9
  simp only [← c91359, ← c91360, k2, ← c91361] at e_1489_9
  have hr := e_1489_9
  simp only [e_1489_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14300)) := by
  have c91362 := (publicBatchWrapper89_copies2855 a h).2.2.1
  have c91363 := (publicBatchWrapper89_copies2855 a h).2.2.2.1
  have c91364 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.1
  have c91365 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.1
  have c91366 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.1
  have c91367 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_10
  simp only [← c91362, ← c91363, ← c91364] at e_1489_10
  have e_1489_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_11
  simp only [← c91365, ← c91366, k2, ← c91367] at e_1489_11
  have hr := e_1489_11
  simp only [e_1489_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14301)) := by
  have c91368 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.1
  have c91369 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.1
  have c91370 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91371 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91372 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91373 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_12
  simp only [← c91368, ← c91369, ← c91370] at e_1489_12
  have e_1489_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_13
  simp only [← c91371, ← c91372, k2, ← c91373] at e_1489_13
  have hr := e_1489_13
  simp only [e_1489_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14302)) := by
  have c91374 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91375 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91376 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91377 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91378 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91379 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_14
  simp only [← c91374, ← c91375, ← c91376] at e_1489_14
  have e_1489_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_15
  simp only [← c91377, ← c91378, k2, ← c91379] at e_1489_15
  have hr := e_1489_15
  simp only [e_1489_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14303)) := by
  have c91380 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91381 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91382 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91383 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91384 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91385 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_16
  simp only [← c91380, ← c91381, ← c91382] at e_1489_16
  have e_1489_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_17
  simp only [← c91383, ← c91384, k2, ← c91385] at e_1489_17
  have hr := e_1489_17
  simp only [e_1489_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1489 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14304)) := by
  have c91386 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91387 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91388 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91389 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91390 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91391 := (publicBatchWrapper89_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1489_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_18
  simp only [← c91386, ← c91387, ← c91388] at e_1489_18
  have e_1489_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_19
  simp only [← c91389, ← c91390, k2, ← c91391] at e_1489_19
  have hr := e_1489_19
  simp only [e_1489_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14305)) := by
  have c91392 := (publicBatchWrapper89_copies2856 a h).1
  have c91393 := (publicBatchWrapper89_copies2856 a h).2.1
  have c91394 := (publicBatchWrapper89_copies2856 a h).2.2.1
  have c91395 := (publicBatchWrapper89_copies2856 a h).2.2.2.1
  have c91396 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.1
  have c91397 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_0
  simp only [← c91392, ← c91393, ← c91394] at e_1490_0
  have e_1490_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_1
  simp only [← c91395, ← c91396, k2, ← c91397] at e_1490_1
  have hr := e_1490_1
  simp only [e_1490_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14306)) := by
  have c91398 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.1
  have c91399 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.1
  have c91400 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.1
  have c91401 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.1
  have c91402 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91403 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_2
  simp only [← c91398, ← c91399, ← c91400] at e_1490_2
  have e_1490_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_3
  simp only [← c91401, ← c91402, k2, ← c91403] at e_1490_3
  have hr := e_1490_3
  simp only [e_1490_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14307)) := by
  have c91404 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91405 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91406 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91407 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91408 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91409 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_4
  simp only [← c91404, ← c91405, ← c91406] at e_1490_4
  have e_1490_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_5
  simp only [← c91407, ← c91408, k2, ← c91409] at e_1490_5
  have hr := e_1490_5
  simp only [e_1490_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14308)) := by
  have c91410 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91411 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91412 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91413 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91414 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91415 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_6
  simp only [← c91410, ← c91411, ← c91412] at e_1490_6
  have e_1490_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_7
  simp only [← c91413, ← c91414, k2, ← c91415] at e_1490_7
  have hr := e_1490_7
  simp only [e_1490_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14309)) := by
  have c91416 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91417 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91418 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91419 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91420 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91421 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_8
  simp only [← c91416, ← c91417, ← c91418] at e_1490_8
  have e_1490_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_9
  simp only [← c91419, ← c91420, k2, ← c91421] at e_1490_9
  have hr := e_1490_9
  simp only [e_1490_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14310)) := by
  have c91422 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91423 := (publicBatchWrapper89_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91424 := (publicBatchWrapper89_copies2857 a h).1
  have c91425 := (publicBatchWrapper89_copies2857 a h).2.1
  have c91426 := (publicBatchWrapper89_copies2857 a h).2.2.1
  have c91427 := (publicBatchWrapper89_copies2857 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_10
  simp only [← c91422, ← c91423, ← c91424] at e_1490_10
  have e_1490_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_11
  simp only [← c91425, ← c91426, k2, ← c91427] at e_1490_11
  have hr := e_1490_11
  simp only [e_1490_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14311)) := by
  have c91428 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.1
  have c91429 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.1
  have c91430 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.1
  have c91431 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.1
  have c91432 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.1
  have c91433 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_12
  simp only [← c91428, ← c91429, ← c91430] at e_1490_12
  have e_1490_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_13
  simp only [← c91431, ← c91432, k2, ← c91433] at e_1490_13
  have hr := e_1490_13
  simp only [e_1490_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14312)) := by
  have c91434 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91435 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91436 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91437 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91438 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91439 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_14
  simp only [← c91434, ← c91435, ← c91436] at e_1490_14
  have e_1490_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_15
  simp only [← c91437, ← c91438, k2, ← c91439] at e_1490_15
  have hr := e_1490_15
  simp only [e_1490_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14313)) := by
  have c91440 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91441 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91442 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91443 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91444 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91445 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_16
  simp only [← c91440, ← c91441, ← c91442] at e_1490_16
  have e_1490_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_17
  simp only [← c91443, ← c91444, k2, ← c91445] at e_1490_17
  have hr := e_1490_17
  simp only [e_1490_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1490 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14314)) := by
  have c91446 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91447 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91448 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91449 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91450 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91451 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1490_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_18
  simp only [← c91446, ← c91447, ← c91448] at e_1490_18
  have e_1490_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_19
  simp only [← c91449, ← c91450, k2, ← c91451] at e_1490_19
  have hr := e_1490_19
  simp only [e_1490_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14315)) := by
  have c91452 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91453 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91454 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91455 := (publicBatchWrapper89_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91456 := (publicBatchWrapper89_copies2858 a h).1
  have c91457 := (publicBatchWrapper89_copies2858 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_0
  simp only [← c91452, ← c91453, ← c91454] at e_1491_0
  have e_1491_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_1
  simp only [← c91455, ← c91456, k2, ← c91457] at e_1491_1
  have hr := e_1491_1
  simp only [e_1491_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14316)) := by
  have c91458 := (publicBatchWrapper89_copies2858 a h).2.2.1
  have c91459 := (publicBatchWrapper89_copies2858 a h).2.2.2.1
  have c91460 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.1
  have c91461 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.1
  have c91462 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.1
  have c91463 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_2
  simp only [← c91458, ← c91459, ← c91460] at e_1491_2
  have e_1491_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_3
  simp only [← c91461, ← c91462, k2, ← c91463] at e_1491_3
  have hr := e_1491_3
  simp only [e_1491_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14317)) := by
  have c91464 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.1
  have c91465 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.1
  have c91466 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91467 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91468 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91469 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_4
  simp only [← c91464, ← c91465, ← c91466] at e_1491_4
  have e_1491_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_5
  simp only [← c91467, ← c91468, k2, ← c91469] at e_1491_5
  have hr := e_1491_5
  simp only [e_1491_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14318)) := by
  have c91470 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91471 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91472 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91473 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91474 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91475 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_6
  simp only [← c91470, ← c91471, ← c91472] at e_1491_6
  have e_1491_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_7
  simp only [← c91473, ← c91474, k2, ← c91475] at e_1491_7
  have hr := e_1491_7
  simp only [e_1491_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14319)) := by
  have c91476 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91477 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91478 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91479 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91480 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91481 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_8
  simp only [← c91476, ← c91477, ← c91478] at e_1491_8
  have e_1491_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_9
  simp only [← c91479, ← c91480, k2, ← c91481] at e_1491_9
  have hr := e_1491_9
  simp only [e_1491_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14320)) := by
  have c91482 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91483 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91484 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91485 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91486 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91487 := (publicBatchWrapper89_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_10
  simp only [← c91482, ← c91483, ← c91484] at e_1491_10
  have e_1491_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_11
  simp only [← c91485, ← c91486, k2, ← c91487] at e_1491_11
  have hr := e_1491_11
  simp only [e_1491_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14321)) := by
  have c91488 := (publicBatchWrapper89_copies2859 a h).1
  have c91489 := (publicBatchWrapper89_copies2859 a h).2.1
  have c91490 := (publicBatchWrapper89_copies2859 a h).2.2.1
  have c91491 := (publicBatchWrapper89_copies2859 a h).2.2.2.1
  have c91492 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.1
  have c91493 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_12
  simp only [← c91488, ← c91489, ← c91490] at e_1491_12
  have e_1491_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_13
  simp only [← c91491, ← c91492, k2, ← c91493] at e_1491_13
  have hr := e_1491_13
  simp only [e_1491_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14322)) := by
  have c91494 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.1
  have c91495 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.1
  have c91496 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.1
  have c91497 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.1
  have c91498 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91499 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_14
  simp only [← c91494, ← c91495, ← c91496] at e_1491_14
  have e_1491_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_15
  simp only [← c91497, ← c91498, k2, ← c91499] at e_1491_15
  have hr := e_1491_15
  simp only [e_1491_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14323)) := by
  have c91500 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91501 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91502 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91503 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91504 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91505 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_16
  simp only [← c91500, ← c91501, ← c91502] at e_1491_16
  have e_1491_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_17
  simp only [← c91503, ← c91504, k2, ← c91505] at e_1491_17
  have hr := e_1491_17
  simp only [e_1491_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1491 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14324)) := by
  have c91506 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91507 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91508 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91509 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91510 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91511 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1491_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_18
  simp only [← c91506, ← c91507, ← c91508] at e_1491_18
  have e_1491_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_19
  simp only [← c91509, ← c91510, k2, ← c91511] at e_1491_19
  have hr := e_1491_19
  simp only [e_1491_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14325)) := by
  have c91512 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91513 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91514 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91515 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91516 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91517 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_0
  simp only [← c91512, ← c91513, ← c91514] at e_1492_0
  have e_1492_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_1
  simp only [← c91515, ← c91516, k2, ← c91517] at e_1492_1
  have hr := e_1492_1
  simp only [e_1492_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14326)) := by
  have c91518 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91519 := (publicBatchWrapper89_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91520 := (publicBatchWrapper89_copies2860 a h).1
  have c91521 := (publicBatchWrapper89_copies2860 a h).2.1
  have c91522 := (publicBatchWrapper89_copies2860 a h).2.2.1
  have c91523 := (publicBatchWrapper89_copies2860 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_2
  simp only [← c91518, ← c91519, ← c91520] at e_1492_2
  have e_1492_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_3
  simp only [← c91521, ← c91522, k2, ← c91523] at e_1492_3
  have hr := e_1492_3
  simp only [e_1492_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14327)) := by
  have c91524 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.1
  have c91525 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.1
  have c91526 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.1
  have c91527 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.1
  have c91528 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.1
  have c91529 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_4
  simp only [← c91524, ← c91525, ← c91526] at e_1492_4
  have e_1492_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_5
  simp only [← c91527, ← c91528, k2, ← c91529] at e_1492_5
  have hr := e_1492_5
  simp only [e_1492_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14328)) := by
  have c91530 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91531 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91532 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91533 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91534 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91535 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_6
  simp only [← c91530, ← c91531, ← c91532] at e_1492_6
  have e_1492_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_7
  simp only [← c91533, ← c91534, k2, ← c91535] at e_1492_7
  have hr := e_1492_7
  simp only [e_1492_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14329)) := by
  have c91536 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91537 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91538 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91539 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91540 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91541 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_8
  simp only [← c91536, ← c91537, ← c91538] at e_1492_8
  have e_1492_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_9
  simp only [← c91539, ← c91540, k2, ← c91541] at e_1492_9
  have hr := e_1492_9
  simp only [e_1492_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14330)) := by
  have c91542 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91543 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91544 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91545 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91546 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91547 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_10
  simp only [← c91542, ← c91543, ← c91544] at e_1492_10
  have e_1492_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_11
  simp only [← c91545, ← c91546, k2, ← c91547] at e_1492_11
  have hr := e_1492_11
  simp only [e_1492_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14331)) := by
  have c91548 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91549 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91550 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91551 := (publicBatchWrapper89_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91552 := (publicBatchWrapper89_copies2861 a h).1
  have c91553 := (publicBatchWrapper89_copies2861 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_12
  simp only [← c91548, ← c91549, ← c91550] at e_1492_12
  have e_1492_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_13
  simp only [← c91551, ← c91552, k2, ← c91553] at e_1492_13
  have hr := e_1492_13
  simp only [e_1492_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14332)) := by
  have c91554 := (publicBatchWrapper89_copies2861 a h).2.2.1
  have c91555 := (publicBatchWrapper89_copies2861 a h).2.2.2.1
  have c91556 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.1
  have c91557 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.1
  have c91558 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.1
  have c91559 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_14
  simp only [← c91554, ← c91555, ← c91556] at e_1492_14
  have e_1492_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_15
  simp only [← c91557, ← c91558, k2, ← c91559] at e_1492_15
  have hr := e_1492_15
  simp only [e_1492_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14333)) := by
  have c91560 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.1
  have c91561 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.1
  have c91562 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91563 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91564 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91565 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_16
  simp only [← c91560, ← c91561, ← c91562] at e_1492_16
  have e_1492_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_17
  simp only [← c91563, ← c91564, k2, ← c91565] at e_1492_17
  have hr := e_1492_17
  simp only [e_1492_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1492 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14334)) := by
  have c91566 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91567 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91568 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91569 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91570 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91571 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1492_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_18
  simp only [← c91566, ← c91567, ← c91568] at e_1492_18
  have e_1492_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_19
  simp only [← c91569, ← c91570, k2, ← c91571] at e_1492_19
  have hr := e_1492_19
  simp only [e_1492_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14335)) := by
  have c91572 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91573 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91574 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91575 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91576 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91577 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_0
  simp only [← c91572, ← c91573, ← c91574] at e_1493_0
  have e_1493_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_1
  simp only [← c91575, ← c91576, k2, ← c91577] at e_1493_1
  have hr := e_1493_1
  simp only [e_1493_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14336)) := by
  have c91578 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91579 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91580 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91581 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91582 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91583 := (publicBatchWrapper89_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_2
  simp only [← c91578, ← c91579, ← c91580] at e_1493_2
  have e_1493_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_3
  simp only [← c91581, ← c91582, k2, ← c91583] at e_1493_3
  have hr := e_1493_3
  simp only [e_1493_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14337)) := by
  have c91584 := (publicBatchWrapper89_copies2862 a h).1
  have c91585 := (publicBatchWrapper89_copies2862 a h).2.1
  have c91586 := (publicBatchWrapper89_copies2862 a h).2.2.1
  have c91587 := (publicBatchWrapper89_copies2862 a h).2.2.2.1
  have c91588 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.1
  have c91589 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_4
  simp only [← c91584, ← c91585, ← c91586] at e_1493_4
  have e_1493_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_5
  simp only [← c91587, ← c91588, k2, ← c91589] at e_1493_5
  have hr := e_1493_5
  simp only [e_1493_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14338)) := by
  have c91590 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.1
  have c91591 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.1
  have c91592 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.1
  have c91593 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.1
  have c91594 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91595 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_6
  simp only [← c91590, ← c91591, ← c91592] at e_1493_6
  have e_1493_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_7
  simp only [← c91593, ← c91594, k2, ← c91595] at e_1493_7
  have hr := e_1493_7
  simp only [e_1493_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14339)) := by
  have c91596 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91597 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91598 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91599 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91600 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91601 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_8
  simp only [← c91596, ← c91597, ← c91598] at e_1493_8
  have e_1493_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_9
  simp only [← c91599, ← c91600, k2, ← c91601] at e_1493_9
  have hr := e_1493_9
  simp only [e_1493_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14340)) := by
  have c91602 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91603 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91604 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91605 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91606 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91607 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_10
  simp only [← c91602, ← c91603, ← c91604] at e_1493_10
  have e_1493_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_11
  simp only [← c91605, ← c91606, k2, ← c91607] at e_1493_11
  have hr := e_1493_11
  simp only [e_1493_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14341)) := by
  have c91608 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91609 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91610 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91611 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91612 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91613 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_12
  simp only [← c91608, ← c91609, ← c91610] at e_1493_12
  have e_1493_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_13
  simp only [← c91611, ← c91612, k2, ← c91613] at e_1493_13
  have hr := e_1493_13
  simp only [e_1493_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14342)) := by
  have c91614 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91615 := (publicBatchWrapper89_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91616 := (publicBatchWrapper89_copies2863 a h).1
  have c91617 := (publicBatchWrapper89_copies2863 a h).2.1
  have c91618 := (publicBatchWrapper89_copies2863 a h).2.2.1
  have c91619 := (publicBatchWrapper89_copies2863 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_14
  simp only [← c91614, ← c91615, ← c91616] at e_1493_14
  have e_1493_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_15
  simp only [← c91617, ← c91618, k2, ← c91619] at e_1493_15
  have hr := e_1493_15
  simp only [e_1493_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14343)) := by
  have c91620 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.1
  have c91621 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.1
  have c91622 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.1
  have c91623 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.1
  have c91624 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.1
  have c91625 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_16
  simp only [← c91620, ← c91621, ← c91622] at e_1493_16
  have e_1493_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_17
  simp only [← c91623, ← c91624, k2, ← c91625] at e_1493_17
  have hr := e_1493_17
  simp only [e_1493_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1493 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14344)) := by
  have c91626 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91627 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91628 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91629 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91630 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91631 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1493_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_18
  simp only [← c91626, ← c91627, ← c91628] at e_1493_18
  have e_1493_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_19
  simp only [← c91629, ← c91630, k2, ← c91631] at e_1493_19
  have hr := e_1493_19
  simp only [e_1493_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14345)) := by
  have c91632 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91633 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91634 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91635 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91636 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91637 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_0
  simp only [← c91632, ← c91633, ← c91634] at e_1494_0
  have e_1494_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_1
  simp only [← c91635, ← c91636, k2, ← c91637] at e_1494_1
  have hr := e_1494_1
  simp only [e_1494_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14346)) := by
  have c91638 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91639 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91640 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91641 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91642 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91643 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_2
  simp only [← c91638, ← c91639, ← c91640] at e_1494_2
  have e_1494_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_3
  simp only [← c91641, ← c91642, k2, ← c91643] at e_1494_3
  have hr := e_1494_3
  simp only [e_1494_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14347)) := by
  have c91644 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91645 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91646 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91647 := (publicBatchWrapper89_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91648 := (publicBatchWrapper89_copies2864 a h).1
  have c91649 := (publicBatchWrapper89_copies2864 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_4
  simp only [← c91644, ← c91645, ← c91646] at e_1494_4
  have e_1494_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_5
  simp only [← c91647, ← c91648, k2, ← c91649] at e_1494_5
  have hr := e_1494_5
  simp only [e_1494_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14348)) := by
  have c91650 := (publicBatchWrapper89_copies2864 a h).2.2.1
  have c91651 := (publicBatchWrapper89_copies2864 a h).2.2.2.1
  have c91652 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.1
  have c91653 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.1
  have c91654 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.1
  have c91655 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_6
  simp only [← c91650, ← c91651, ← c91652] at e_1494_6
  have e_1494_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_7
  simp only [← c91653, ← c91654, k2, ← c91655] at e_1494_7
  have hr := e_1494_7
  simp only [e_1494_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14349)) := by
  have c91656 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.1
  have c91657 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.1
  have c91658 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91659 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91660 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91661 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_8
  simp only [← c91656, ← c91657, ← c91658] at e_1494_8
  have e_1494_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_9
  simp only [← c91659, ← c91660, k2, ← c91661] at e_1494_9
  have hr := e_1494_9
  simp only [e_1494_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14350)) := by
  have c91662 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91663 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91664 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91665 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91666 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91667 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_10
  simp only [← c91662, ← c91663, ← c91664] at e_1494_10
  have e_1494_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_11
  simp only [← c91665, ← c91666, k2, ← c91667] at e_1494_11
  have hr := e_1494_11
  simp only [e_1494_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14351)) := by
  have c91668 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91669 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91670 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91671 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91672 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91673 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_12
  simp only [← c91668, ← c91669, ← c91670] at e_1494_12
  have e_1494_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_13
  simp only [← c91671, ← c91672, k2, ← c91673] at e_1494_13
  have hr := e_1494_13
  simp only [e_1494_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14352)) := by
  have c91674 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91675 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91676 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91677 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91678 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91679 := (publicBatchWrapper89_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_14
  simp only [← c91674, ← c91675, ← c91676] at e_1494_14
  have e_1494_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_15
  simp only [← c91677, ← c91678, k2, ← c91679] at e_1494_15
  have hr := e_1494_15
  simp only [e_1494_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14353)) := by
  have c91680 := (publicBatchWrapper89_copies2865 a h).1
  have c91681 := (publicBatchWrapper89_copies2865 a h).2.1
  have c91682 := (publicBatchWrapper89_copies2865 a h).2.2.1
  have c91683 := (publicBatchWrapper89_copies2865 a h).2.2.2.1
  have c91684 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.1
  have c91685 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_16
  simp only [← c91680, ← c91681, ← c91682] at e_1494_16
  have e_1494_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_17
  simp only [← c91683, ← c91684, k2, ← c91685] at e_1494_17
  have hr := e_1494_17
  simp only [e_1494_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1494 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14354)) := by
  have c91686 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.1
  have c91687 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.1
  have c91688 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.1
  have c91689 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.1
  have c91690 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91691 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1494_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_18
  simp only [← c91686, ← c91687, ← c91688] at e_1494_18
  have e_1494_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_19
  simp only [← c91689, ← c91690, k2, ← c91691] at e_1494_19
  have hr := e_1494_19
  simp only [e_1494_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14355)) := by
  have c91692 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91693 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91694 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91695 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91696 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91697 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_0
  simp only [← c91692, ← c91693, ← c91694] at e_1495_0
  have e_1495_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_1
  simp only [← c91695, ← c91696, k2, ← c91697] at e_1495_1
  have hr := e_1495_1
  simp only [e_1495_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14356)) := by
  have c91698 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91699 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91700 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91701 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91702 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91703 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_2
  simp only [← c91698, ← c91699, ← c91700] at e_1495_2
  have e_1495_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_3
  simp only [← c91701, ← c91702, k2, ← c91703] at e_1495_3
  have hr := e_1495_3
  simp only [e_1495_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14357)) := by
  have c91704 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91705 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91706 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91707 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91708 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91709 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_4
  simp only [← c91704, ← c91705, ← c91706] at e_1495_4
  have e_1495_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_5
  simp only [← c91707, ← c91708, k2, ← c91709] at e_1495_5
  have hr := e_1495_5
  simp only [e_1495_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14358)) := by
  have c91710 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91711 := (publicBatchWrapper89_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91712 := (publicBatchWrapper89_copies2866 a h).1
  have c91713 := (publicBatchWrapper89_copies2866 a h).2.1
  have c91714 := (publicBatchWrapper89_copies2866 a h).2.2.1
  have c91715 := (publicBatchWrapper89_copies2866 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_6
  simp only [← c91710, ← c91711, ← c91712] at e_1495_6
  have e_1495_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_7
  simp only [← c91713, ← c91714, k2, ← c91715] at e_1495_7
  have hr := e_1495_7
  simp only [e_1495_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14359)) := by
  have c91716 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.1
  have c91717 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.1
  have c91718 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.1
  have c91719 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.1
  have c91720 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.1
  have c91721 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_8
  simp only [← c91716, ← c91717, ← c91718] at e_1495_8
  have e_1495_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_9
  simp only [← c91719, ← c91720, k2, ← c91721] at e_1495_9
  have hr := e_1495_9
  simp only [e_1495_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14360)) := by
  have c91722 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91723 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91724 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91725 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91726 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91727 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_10
  simp only [← c91722, ← c91723, ← c91724] at e_1495_10
  have e_1495_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_11
  simp only [← c91725, ← c91726, k2, ← c91727] at e_1495_11
  have hr := e_1495_11
  simp only [e_1495_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14361)) := by
  have c91728 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91729 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91730 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91731 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91732 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91733 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_12
  simp only [← c91728, ← c91729, ← c91730] at e_1495_12
  have e_1495_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_13
  simp only [← c91731, ← c91732, k2, ← c91733] at e_1495_13
  have hr := e_1495_13
  simp only [e_1495_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14362)) := by
  have c91734 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91735 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91736 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91737 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91738 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91739 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_14
  simp only [← c91734, ← c91735, ← c91736] at e_1495_14
  have e_1495_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_15
  simp only [← c91737, ← c91738, k2, ← c91739] at e_1495_15
  have hr := e_1495_15
  simp only [e_1495_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14363)) := by
  have c91740 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91741 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91742 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91743 := (publicBatchWrapper89_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91744 := (publicBatchWrapper89_copies2867 a h).1
  have c91745 := (publicBatchWrapper89_copies2867 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_16
  simp only [← c91740, ← c91741, ← c91742] at e_1495_16
  have e_1495_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_17
  simp only [← c91743, ← c91744, k2, ← c91745] at e_1495_17
  have hr := e_1495_17
  simp only [e_1495_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1495 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14364)) := by
  have c91746 := (publicBatchWrapper89_copies2867 a h).2.2.1
  have c91747 := (publicBatchWrapper89_copies2867 a h).2.2.2.1
  have c91748 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.1
  have c91749 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.1
  have c91750 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.1
  have c91751 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1495_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_18
  simp only [← c91746, ← c91747, ← c91748] at e_1495_18
  have e_1495_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_19
  simp only [← c91749, ← c91750, k2, ← c91751] at e_1495_19
  have hr := e_1495_19
  simp only [e_1495_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14365)) := by
  have c91752 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.1
  have c91753 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.1
  have c91754 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91755 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91756 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91757 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_0
  simp only [← c91752, ← c91753, ← c91754] at e_1496_0
  have e_1496_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_1
  simp only [← c91755, ← c91756, k2, ← c91757] at e_1496_1
  have hr := e_1496_1
  simp only [e_1496_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14366)) := by
  have c91758 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91759 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91760 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91761 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91762 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91763 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_2
  simp only [← c91758, ← c91759, ← c91760] at e_1496_2
  have e_1496_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_3
  simp only [← c91761, ← c91762, k2, ← c91763] at e_1496_3
  have hr := e_1496_3
  simp only [e_1496_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14367)) := by
  have c91764 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91765 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91766 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91767 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91768 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91769 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_4
  simp only [← c91764, ← c91765, ← c91766] at e_1496_4
  have e_1496_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_5
  simp only [← c91767, ← c91768, k2, ← c91769] at e_1496_5
  have hr := e_1496_5
  simp only [e_1496_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14368)) := by
  have c91770 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91771 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91772 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91773 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91774 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91775 := (publicBatchWrapper89_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_6
  simp only [← c91770, ← c91771, ← c91772] at e_1496_6
  have e_1496_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_7
  simp only [← c91773, ← c91774, k2, ← c91775] at e_1496_7
  have hr := e_1496_7
  simp only [e_1496_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14369)) := by
  have c91776 := (publicBatchWrapper89_copies2868 a h).1
  have c91777 := (publicBatchWrapper89_copies2868 a h).2.1
  have c91778 := (publicBatchWrapper89_copies2868 a h).2.2.1
  have c91779 := (publicBatchWrapper89_copies2868 a h).2.2.2.1
  have c91780 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.1
  have c91781 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_8
  simp only [← c91776, ← c91777, ← c91778] at e_1496_8
  have e_1496_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_9
  simp only [← c91779, ← c91780, k2, ← c91781] at e_1496_9
  have hr := e_1496_9
  simp only [e_1496_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14410)) := by
  have c91782 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.1
  have c91783 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.1
  have c91784 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.1
  have c91785 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.1
  have c91786 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91787 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_10
  simp only [← c91782, ← c91783, ← c91784] at e_1496_10
  have e_1496_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_11
  simp only [← c91785, ← c91786, k2, ← c91787] at e_1496_11
  have hr := e_1496_11
  simp only [e_1496_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14411)) := by
  have c91788 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91789 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91790 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91791 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91792 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91793 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_12
  simp only [← c91788, ← c91789, ← c91790] at e_1496_12
  have e_1496_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_13
  simp only [← c91791, ← c91792, k2, ← c91793] at e_1496_13
  have hr := e_1496_13
  simp only [e_1496_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14412)) := by
  have c91794 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91795 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91796 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91797 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91798 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91799 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_14
  simp only [← c91794, ← c91795, ← c91796] at e_1496_14
  have e_1496_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_15
  simp only [← c91797, ← c91798, k2, ← c91799] at e_1496_15
  have hr := e_1496_15
  simp only [e_1496_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14413)) := by
  have c91800 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91801 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91802 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91803 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91804 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91805 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_16
  simp only [← c91800, ← c91801, ← c91802] at e_1496_16
  have e_1496_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_17
  simp only [← c91803, ← c91804, k2, ← c91805] at e_1496_17
  have hr := e_1496_17
  simp only [e_1496_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1496 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14414)) := by
  have c91806 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91807 := (publicBatchWrapper89_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91808 := (publicBatchWrapper89_copies2869 a h).1
  have c91809 := (publicBatchWrapper89_copies2869 a h).2.1
  have c91810 := (publicBatchWrapper89_copies2869 a h).2.2.1
  have c91811 := (publicBatchWrapper89_copies2869 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1496_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_18
  simp only [← c91806, ← c91807, ← c91808] at e_1496_18
  have e_1496_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_19
  simp only [← c91809, ← c91810, k2, ← c91811] at e_1496_19
  have hr := e_1496_19
  simp only [e_1496_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14415)) := by
  have c91812 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.1
  have c91813 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.1
  have c91814 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.1
  have c91815 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.1
  have c91816 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.1
  have c91817 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_0
  simp only [← c91812, ← c91813, ← c91814] at e_1497_0
  have e_1497_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_1
  simp only [← c91815, ← c91816, k2, ← c91817] at e_1497_1
  have hr := e_1497_1
  simp only [e_1497_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14416)) := by
  have c91818 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91819 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91820 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91821 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91822 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91823 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_2
  simp only [← c91818, ← c91819, ← c91820] at e_1497_2
  have e_1497_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_3
  simp only [← c91821, ← c91822, k2, ← c91823] at e_1497_3
  have hr := e_1497_3
  simp only [e_1497_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14417)) := by
  have c91824 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91825 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91826 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91827 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91828 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91829 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_4
  simp only [← c91824, ← c91825, ← c91826] at e_1497_4
  have e_1497_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_5
  simp only [← c91827, ← c91828, k2, ← c91829] at e_1497_5
  have hr := e_1497_5
  simp only [e_1497_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14418)) := by
  have c91830 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91831 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91832 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91833 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91834 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91835 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_6
  simp only [← c91830, ← c91831, ← c91832] at e_1497_6
  have e_1497_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_7
  simp only [← c91833, ← c91834, k2, ← c91835] at e_1497_7
  have hr := e_1497_7
  simp only [e_1497_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14419)) := by
  have c91836 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91837 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91838 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91839 := (publicBatchWrapper89_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91840 := (publicBatchWrapper89_copies2870 a h).1
  have c91841 := (publicBatchWrapper89_copies2870 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_8
  simp only [← c91836, ← c91837, ← c91838] at e_1497_8
  have e_1497_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_9
  simp only [← c91839, ← c91840, k2, ← c91841] at e_1497_9
  have hr := e_1497_9
  simp only [e_1497_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14420)) := by
  have c91842 := (publicBatchWrapper89_copies2870 a h).2.2.1
  have c91843 := (publicBatchWrapper89_copies2870 a h).2.2.2.1
  have c91844 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.1
  have c91845 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.1
  have c91846 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.1
  have c91847 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_10
  simp only [← c91842, ← c91843, ← c91844] at e_1497_10
  have e_1497_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_11
  simp only [← c91845, ← c91846, k2, ← c91847] at e_1497_11
  have hr := e_1497_11
  simp only [e_1497_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14421)) := by
  have c91848 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.1
  have c91849 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.1
  have c91850 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91851 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91852 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91853 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_12
  simp only [← c91848, ← c91849, ← c91850] at e_1497_12
  have e_1497_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_13
  simp only [← c91851, ← c91852, k2, ← c91853] at e_1497_13
  have hr := e_1497_13
  simp only [e_1497_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14422)) := by
  have c91854 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91855 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91856 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91857 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91858 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91859 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_14
  simp only [← c91854, ← c91855, ← c91856] at e_1497_14
  have e_1497_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_15
  simp only [← c91857, ← c91858, k2, ← c91859] at e_1497_15
  have hr := e_1497_15
  simp only [e_1497_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14423)) := by
  have c91860 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91861 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91862 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91863 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91864 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91865 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_16
  simp only [← c91860, ← c91861, ← c91862] at e_1497_16
  have e_1497_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_17
  simp only [← c91863, ← c91864, k2, ← c91865] at e_1497_17
  have hr := e_1497_17
  simp only [e_1497_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1497 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14424)) := by
  have c91866 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91867 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91868 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91869 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91870 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91871 := (publicBatchWrapper89_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1497_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_18
  simp only [← c91866, ← c91867, ← c91868] at e_1497_18
  have e_1497_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_19
  simp only [← c91869, ← c91870, k2, ← c91871] at e_1497_19
  have hr := e_1497_19
  simp only [e_1497_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14425)) := by
  have c91872 := (publicBatchWrapper89_copies2871 a h).1
  have c91873 := (publicBatchWrapper89_copies2871 a h).2.1
  have c91874 := (publicBatchWrapper89_copies2871 a h).2.2.1
  have c91875 := (publicBatchWrapper89_copies2871 a h).2.2.2.1
  have c91876 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.1
  have c91877 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_0
  simp only [← c91872, ← c91873, ← c91874] at e_1498_0
  have e_1498_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_1
  simp only [← c91875, ← c91876, k2, ← c91877] at e_1498_1
  have hr := e_1498_1
  simp only [e_1498_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14426)) := by
  have c91878 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.1
  have c91879 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.1
  have c91880 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.1
  have c91881 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.1
  have c91882 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91883 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_2
  simp only [← c91878, ← c91879, ← c91880] at e_1498_2
  have e_1498_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_3
  simp only [← c91881, ← c91882, k2, ← c91883] at e_1498_3
  have hr := e_1498_3
  simp only [e_1498_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14427)) := by
  have c91884 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91885 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91886 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91887 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91888 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91889 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_4
  simp only [← c91884, ← c91885, ← c91886] at e_1498_4
  have e_1498_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_5
  simp only [← c91887, ← c91888, k2, ← c91889] at e_1498_5
  have hr := e_1498_5
  simp only [e_1498_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14428)) := by
  have c91890 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91891 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91892 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91893 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91894 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91895 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_6
  simp only [← c91890, ← c91891, ← c91892] at e_1498_6
  have e_1498_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_7
  simp only [← c91893, ← c91894, k2, ← c91895] at e_1498_7
  have hr := e_1498_7
  simp only [e_1498_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14429)) := by
  have c91896 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91897 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91898 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91899 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91900 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91901 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_8
  simp only [← c91896, ← c91897, ← c91898] at e_1498_8
  have e_1498_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_9
  simp only [← c91899, ← c91900, k2, ← c91901] at e_1498_9
  have hr := e_1498_9
  simp only [e_1498_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14430)) := by
  have c91902 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91903 := (publicBatchWrapper89_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91904 := (publicBatchWrapper89_copies2872 a h).1
  have c91905 := (publicBatchWrapper89_copies2872 a h).2.1
  have c91906 := (publicBatchWrapper89_copies2872 a h).2.2.1
  have c91907 := (publicBatchWrapper89_copies2872 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_10
  simp only [← c91902, ← c91903, ← c91904] at e_1498_10
  have e_1498_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_11
  simp only [← c91905, ← c91906, k2, ← c91907] at e_1498_11
  have hr := e_1498_11
  simp only [e_1498_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14431)) := by
  have c91908 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.1
  have c91909 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.1
  have c91910 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.1
  have c91911 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.1
  have c91912 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.1
  have c91913 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_12
  simp only [← c91908, ← c91909, ← c91910] at e_1498_12
  have e_1498_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_13
  simp only [← c91911, ← c91912, k2, ← c91913] at e_1498_13
  have hr := e_1498_13
  simp only [e_1498_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14432)) := by
  have c91914 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91915 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91916 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91917 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91918 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91919 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_14
  simp only [← c91914, ← c91915, ← c91916] at e_1498_14
  have e_1498_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_15
  simp only [← c91917, ← c91918, k2, ← c91919] at e_1498_15
  have hr := e_1498_15
  simp only [e_1498_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14433)) := by
  have c91920 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91921 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91922 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91923 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91924 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91925 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_16
  simp only [← c91920, ← c91921, ← c91922] at e_1498_16
  have e_1498_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_17
  simp only [← c91923, ← c91924, k2, ← c91925] at e_1498_17
  have hr := e_1498_17
  simp only [e_1498_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1498 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14434)) := by
  have c91926 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91927 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91928 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91929 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91930 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91931 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1498_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_18
  simp only [← c91926, ← c91927, ← c91928] at e_1498_18
  have e_1498_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_19
  simp only [← c91929, ← c91930, k2, ← c91931] at e_1498_19
  have hr := e_1498_19
  simp only [e_1498_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14435)) := by
  have c91932 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91933 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91934 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91935 := (publicBatchWrapper89_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91936 := (publicBatchWrapper89_copies2873 a h).1
  have c91937 := (publicBatchWrapper89_copies2873 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_0
  simp only [← c91932, ← c91933, ← c91934] at e_1499_0
  have e_1499_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_1
  simp only [← c91935, ← c91936, k2, ← c91937] at e_1499_1
  have hr := e_1499_1
  simp only [e_1499_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14436)) := by
  have c91938 := (publicBatchWrapper89_copies2873 a h).2.2.1
  have c91939 := (publicBatchWrapper89_copies2873 a h).2.2.2.1
  have c91940 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.1
  have c91941 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.1
  have c91942 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.1
  have c91943 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_2
  simp only [← c91938, ← c91939, ← c91940] at e_1499_2
  have e_1499_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_3
  simp only [← c91941, ← c91942, k2, ← c91943] at e_1499_3
  have hr := e_1499_3
  simp only [e_1499_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14437)) := by
  have c91944 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.1
  have c91945 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.1
  have c91946 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91947 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91948 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91949 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_4
  simp only [← c91944, ← c91945, ← c91946] at e_1499_4
  have e_1499_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_5
  simp only [← c91947, ← c91948, k2, ← c91949] at e_1499_5
  have hr := e_1499_5
  simp only [e_1499_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14438)) := by
  have c91950 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91951 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91952 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91953 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91954 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91955 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_6
  simp only [← c91950, ← c91951, ← c91952] at e_1499_6
  have e_1499_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_7
  simp only [← c91953, ← c91954, k2, ← c91955] at e_1499_7
  have hr := e_1499_7
  simp only [e_1499_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14439)) := by
  have c91956 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91957 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91958 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91959 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91960 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91961 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_8
  simp only [← c91956, ← c91957, ← c91958] at e_1499_8
  have e_1499_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_9
  simp only [← c91959, ← c91960, k2, ← c91961] at e_1499_9
  have hr := e_1499_9
  simp only [e_1499_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14440)) := by
  have c91962 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91963 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91964 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91965 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91966 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91967 := (publicBatchWrapper89_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_10
  simp only [← c91962, ← c91963, ← c91964] at e_1499_10
  have e_1499_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_11
  simp only [← c91965, ← c91966, k2, ← c91967] at e_1499_11
  have hr := e_1499_11
  simp only [e_1499_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14441)) := by
  have c91968 := (publicBatchWrapper89_copies2874 a h).1
  have c91969 := (publicBatchWrapper89_copies2874 a h).2.1
  have c91970 := (publicBatchWrapper89_copies2874 a h).2.2.1
  have c91971 := (publicBatchWrapper89_copies2874 a h).2.2.2.1
  have c91972 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.1
  have c91973 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_12
  simp only [← c91968, ← c91969, ← c91970] at e_1499_12
  have e_1499_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_13
  simp only [← c91971, ← c91972, k2, ← c91973] at e_1499_13
  have hr := e_1499_13
  simp only [e_1499_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14442)) := by
  have c91974 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.1
  have c91975 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.1
  have c91976 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.1
  have c91977 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.1
  have c91978 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91979 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_14
  simp only [← c91974, ← c91975, ← c91976] at e_1499_14
  have e_1499_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_15
  simp only [← c91977, ← c91978, k2, ← c91979] at e_1499_15
  have hr := e_1499_15
  simp only [e_1499_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14443)) := by
  have c91980 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91981 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91982 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91983 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91984 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91985 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_16
  simp only [← c91980, ← c91981, ← c91982] at e_1499_16
  have e_1499_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_17
  simp only [← c91983, ← c91984, k2, ← c91985] at e_1499_17
  have hr := e_1499_17
  simp only [e_1499_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1499 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14444)) := by
  have c91986 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91987 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91988 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91989 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91990 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91991 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1499_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_18
  simp only [← c91986, ← c91987, ← c91988] at e_1499_18
  have e_1499_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_19
  simp only [← c91989, ← c91990, k2, ← c91991] at e_1499_19
  have hr := e_1499_19
  simp only [e_1499_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14445)) := by
  have c91992 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91993 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91994 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91995 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91996 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91997 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_0
  simp only [← c91992, ← c91993, ← c91994] at e_1500_0
  have e_1500_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_1
  simp only [← c91995, ← c91996, k2, ← c91997] at e_1500_1
  have hr := e_1500_1
  simp only [e_1500_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14446)) := by
  have c91998 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91999 := (publicBatchWrapper89_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92000 := (publicBatchWrapper89_copies2875 a h).1
  have c92001 := (publicBatchWrapper89_copies2875 a h).2.1
  have c92002 := (publicBatchWrapper89_copies2875 a h).2.2.1
  have c92003 := (publicBatchWrapper89_copies2875 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_2
  simp only [← c91998, ← c91999, ← c92000] at e_1500_2
  have e_1500_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_3
  simp only [← c92001, ← c92002, k2, ← c92003] at e_1500_3
  have hr := e_1500_3
  simp only [e_1500_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14447)) := by
  have c92004 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.1
  have c92005 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.1
  have c92006 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.1
  have c92007 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.1
  have c92008 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.1
  have c92009 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_4
  simp only [← c92004, ← c92005, ← c92006] at e_1500_4
  have e_1500_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_5
  simp only [← c92007, ← c92008, k2, ← c92009] at e_1500_5
  have hr := e_1500_5
  simp only [e_1500_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14448)) := by
  have c92010 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92011 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92012 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92013 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92014 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92015 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_6
  simp only [← c92010, ← c92011, ← c92012] at e_1500_6
  have e_1500_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_7
  simp only [← c92013, ← c92014, k2, ← c92015] at e_1500_7
  have hr := e_1500_7
  simp only [e_1500_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14449)) := by
  have c92016 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92017 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92018 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92019 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92020 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92021 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_8
  simp only [← c92016, ← c92017, ← c92018] at e_1500_8
  have e_1500_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_9
  simp only [← c92019, ← c92020, k2, ← c92021] at e_1500_9
  have hr := e_1500_9
  simp only [e_1500_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14450)) := by
  have c92022 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92023 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92024 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92025 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92026 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92027 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_10
  simp only [← c92022, ← c92023, ← c92024] at e_1500_10
  have e_1500_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_11
  simp only [← c92025, ← c92026, k2, ← c92027] at e_1500_11
  have hr := e_1500_11
  simp only [e_1500_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14451)) := by
  have c92028 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92029 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92030 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92031 := (publicBatchWrapper89_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92032 := (publicBatchWrapper89_copies2876 a h).1
  have c92033 := (publicBatchWrapper89_copies2876 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_12
  simp only [← c92028, ← c92029, ← c92030] at e_1500_12
  have e_1500_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_13
  simp only [← c92031, ← c92032, k2, ← c92033] at e_1500_13
  have hr := e_1500_13
  simp only [e_1500_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14452)) := by
  have c92034 := (publicBatchWrapper89_copies2876 a h).2.2.1
  have c92035 := (publicBatchWrapper89_copies2876 a h).2.2.2.1
  have c92036 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.1
  have c92037 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.1
  have c92038 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.1
  have c92039 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_14
  simp only [← c92034, ← c92035, ← c92036] at e_1500_14
  have e_1500_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_15
  simp only [← c92037, ← c92038, k2, ← c92039] at e_1500_15
  have hr := e_1500_15
  simp only [e_1500_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14453)) := by
  have c92040 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.1
  have c92041 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.1
  have c92042 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92043 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92044 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92045 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_16
  simp only [← c92040, ← c92041, ← c92042] at e_1500_16
  have e_1500_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_17
  simp only [← c92043, ← c92044, k2, ← c92045] at e_1500_17
  have hr := e_1500_17
  simp only [e_1500_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1500 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14454)) := by
  have c92046 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92047 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92048 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92049 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92050 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92051 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1500_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_18
  simp only [← c92046, ← c92047, ← c92048] at e_1500_18
  have e_1500_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_19
  simp only [← c92049, ← c92050, k2, ← c92051] at e_1500_19
  have hr := e_1500_19
  simp only [e_1500_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14455)) := by
  have c92052 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92053 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92054 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92055 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92056 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92057 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_0
  simp only [← c92052, ← c92053, ← c92054] at e_1501_0
  have e_1501_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_1
  simp only [← c92055, ← c92056, k2, ← c92057] at e_1501_1
  have hr := e_1501_1
  simp only [e_1501_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14456)) := by
  have c92058 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92059 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92060 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92061 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92062 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92063 := (publicBatchWrapper89_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_2
  simp only [← c92058, ← c92059, ← c92060] at e_1501_2
  have e_1501_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_3
  simp only [← c92061, ← c92062, k2, ← c92063] at e_1501_3
  have hr := e_1501_3
  simp only [e_1501_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14457)) := by
  have c92064 := (publicBatchWrapper89_copies2877 a h).1
  have c92065 := (publicBatchWrapper89_copies2877 a h).2.1
  have c92066 := (publicBatchWrapper89_copies2877 a h).2.2.1
  have c92067 := (publicBatchWrapper89_copies2877 a h).2.2.2.1
  have c92068 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.1
  have c92069 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_4
  simp only [← c92064, ← c92065, ← c92066] at e_1501_4
  have e_1501_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_5
  simp only [← c92067, ← c92068, k2, ← c92069] at e_1501_5
  have hr := e_1501_5
  simp only [e_1501_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14458)) := by
  have c92070 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.1
  have c92071 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.1
  have c92072 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.1
  have c92073 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.1
  have c92074 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92075 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_6
  simp only [← c92070, ← c92071, ← c92072] at e_1501_6
  have e_1501_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_7
  simp only [← c92073, ← c92074, k2, ← c92075] at e_1501_7
  have hr := e_1501_7
  simp only [e_1501_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14459)) := by
  have c92076 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92077 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92078 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92079 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92080 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92081 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_8
  simp only [← c92076, ← c92077, ← c92078] at e_1501_8
  have e_1501_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_9
  simp only [← c92079, ← c92080, k2, ← c92081] at e_1501_9
  have hr := e_1501_9
  simp only [e_1501_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14460)) := by
  have c92082 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92083 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92084 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92085 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92086 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92087 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_10
  simp only [← c92082, ← c92083, ← c92084] at e_1501_10
  have e_1501_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_11
  simp only [← c92085, ← c92086, k2, ← c92087] at e_1501_11
  have hr := e_1501_11
  simp only [e_1501_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14461)) := by
  have c92088 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92089 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92090 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92091 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92092 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92093 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_12
  simp only [← c92088, ← c92089, ← c92090] at e_1501_12
  have e_1501_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_13
  simp only [← c92091, ← c92092, k2, ← c92093] at e_1501_13
  have hr := e_1501_13
  simp only [e_1501_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14462)) := by
  have c92094 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92095 := (publicBatchWrapper89_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92096 := (publicBatchWrapper89_copies2878 a h).1
  have c92097 := (publicBatchWrapper89_copies2878 a h).2.1
  have c92098 := (publicBatchWrapper89_copies2878 a h).2.2.1
  have c92099 := (publicBatchWrapper89_copies2878 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_14
  simp only [← c92094, ← c92095, ← c92096] at e_1501_14
  have e_1501_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_15
  simp only [← c92097, ← c92098, k2, ← c92099] at e_1501_15
  have hr := e_1501_15
  simp only [e_1501_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14463)) := by
  have c92100 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.1
  have c92101 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.1
  have c92102 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.1
  have c92103 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.1
  have c92104 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.1
  have c92105 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_16
  simp only [← c92100, ← c92101, ← c92102] at e_1501_16
  have e_1501_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_17
  simp only [← c92103, ← c92104, k2, ← c92105] at e_1501_17
  have hr := e_1501_17
  simp only [e_1501_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1501 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14464)) := by
  have c92106 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92107 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92108 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92109 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92110 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92111 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1501_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_18
  simp only [← c92106, ← c92107, ← c92108] at e_1501_18
  have e_1501_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_19
  simp only [← c92109, ← c92110, k2, ← c92111] at e_1501_19
  have hr := e_1501_19
  simp only [e_1501_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14465)) := by
  have c92112 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92113 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92114 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92115 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92116 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92117 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_0
  simp only [← c92112, ← c92113, ← c92114] at e_1502_0
  have e_1502_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_1
  simp only [← c92115, ← c92116, k2, ← c92117] at e_1502_1
  have hr := e_1502_1
  simp only [e_1502_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14466)) := by
  have c92118 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92119 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92120 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92121 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92122 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92123 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_2
  simp only [← c92118, ← c92119, ← c92120] at e_1502_2
  have e_1502_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_3
  simp only [← c92121, ← c92122, k2, ← c92123] at e_1502_3
  have hr := e_1502_3
  simp only [e_1502_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14467)) := by
  have c92124 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92125 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92126 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92127 := (publicBatchWrapper89_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92128 := (publicBatchWrapper89_copies2879 a h).1
  have c92129 := (publicBatchWrapper89_copies2879 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_4
  simp only [← c92124, ← c92125, ← c92126] at e_1502_4
  have e_1502_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_5
  simp only [← c92127, ← c92128, k2, ← c92129] at e_1502_5
  have hr := e_1502_5
  simp only [e_1502_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14468)) := by
  have c92130 := (publicBatchWrapper89_copies2879 a h).2.2.1
  have c92131 := (publicBatchWrapper89_copies2879 a h).2.2.2.1
  have c92132 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.1
  have c92133 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.1
  have c92134 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.1
  have c92135 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_6
  simp only [← c92130, ← c92131, ← c92132] at e_1502_6
  have e_1502_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_7
  simp only [← c92133, ← c92134, k2, ← c92135] at e_1502_7
  have hr := e_1502_7
  simp only [e_1502_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14469)) := by
  have c92136 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.1
  have c92137 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.1
  have c92138 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92139 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92140 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92141 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_8
  simp only [← c92136, ← c92137, ← c92138] at e_1502_8
  have e_1502_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_9
  simp only [← c92139, ← c92140, k2, ← c92141] at e_1502_9
  have hr := e_1502_9
  simp only [e_1502_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14470)) := by
  have c92142 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92143 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92144 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92145 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92146 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92147 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_10
  simp only [← c92142, ← c92143, ← c92144] at e_1502_10
  have e_1502_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_11
  simp only [← c92145, ← c92146, k2, ← c92147] at e_1502_11
  have hr := e_1502_11
  simp only [e_1502_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14471)) := by
  have c92148 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92149 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92150 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92151 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92152 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92153 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_12
  simp only [← c92148, ← c92149, ← c92150] at e_1502_12
  have e_1502_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_13
  simp only [← c92151, ← c92152, k2, ← c92153] at e_1502_13
  have hr := e_1502_13
  simp only [e_1502_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14472)) := by
  have c92154 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92155 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92156 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92157 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92158 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92159 := (publicBatchWrapper89_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_14
  simp only [← c92154, ← c92155, ← c92156] at e_1502_14
  have e_1502_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_15
  simp only [← c92157, ← c92158, k2, ← c92159] at e_1502_15
  have hr := e_1502_15
  simp only [e_1502_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14473)) := by
  have c92160 := (publicBatchWrapper89_copies2880 a h).1
  have c92161 := (publicBatchWrapper89_copies2880 a h).2.1
  have c92162 := (publicBatchWrapper89_copies2880 a h).2.2.1
  have c92163 := (publicBatchWrapper89_copies2880 a h).2.2.2.1
  have c92164 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.1
  have c92165 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_16
  simp only [← c92160, ← c92161, ← c92162] at e_1502_16
  have e_1502_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_17
  simp only [← c92163, ← c92164, k2, ← c92165] at e_1502_17
  have hr := e_1502_17
  simp only [e_1502_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1502 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14474)) := by
  have c92166 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.1
  have c92167 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.1
  have c92168 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.1
  have c92169 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.1
  have c92170 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92171 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1502_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_18
  simp only [← c92166, ← c92167, ← c92168] at e_1502_18
  have e_1502_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_19
  simp only [← c92169, ← c92170, k2, ← c92171] at e_1502_19
  have hr := e_1502_19
  simp only [e_1502_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14475)) := by
  have c92172 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92173 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92174 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92175 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92176 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92177 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_0
  simp only [← c92172, ← c92173, ← c92174] at e_1503_0
  have e_1503_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_1
  simp only [← c92175, ← c92176, k2, ← c92177] at e_1503_1
  have hr := e_1503_1
  simp only [e_1503_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14476)) := by
  have c92178 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92179 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92180 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92181 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92182 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92183 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_2
  simp only [← c92178, ← c92179, ← c92180] at e_1503_2
  have e_1503_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_3
  simp only [← c92181, ← c92182, k2, ← c92183] at e_1503_3
  have hr := e_1503_3
  simp only [e_1503_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14477)) := by
  have c92184 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92185 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92186 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92187 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92188 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92189 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_4
  simp only [← c92184, ← c92185, ← c92186] at e_1503_4
  have e_1503_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_5
  simp only [← c92187, ← c92188, k2, ← c92189] at e_1503_5
  have hr := e_1503_5
  simp only [e_1503_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14478)) := by
  have c92190 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92191 := (publicBatchWrapper89_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92192 := (publicBatchWrapper89_copies2881 a h).1
  have c92193 := (publicBatchWrapper89_copies2881 a h).2.1
  have c92194 := (publicBatchWrapper89_copies2881 a h).2.2.1
  have c92195 := (publicBatchWrapper89_copies2881 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_6
  simp only [← c92190, ← c92191, ← c92192] at e_1503_6
  have e_1503_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_7
  simp only [← c92193, ← c92194, k2, ← c92195] at e_1503_7
  have hr := e_1503_7
  simp only [e_1503_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14479)) := by
  have c92196 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.1
  have c92197 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.1
  have c92198 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.1
  have c92199 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.1
  have c92200 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.1
  have c92201 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_8
  simp only [← c92196, ← c92197, ← c92198] at e_1503_8
  have e_1503_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_9
  simp only [← c92199, ← c92200, k2, ← c92201] at e_1503_9
  have hr := e_1503_9
  simp only [e_1503_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14480)) := by
  have c92202 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92203 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92204 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92205 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92206 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92207 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_10
  simp only [← c92202, ← c92203, ← c92204] at e_1503_10
  have e_1503_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_11
  simp only [← c92205, ← c92206, k2, ← c92207] at e_1503_11
  have hr := e_1503_11
  simp only [e_1503_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14481)) := by
  have c92208 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92209 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92210 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92211 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92212 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92213 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_12
  simp only [← c92208, ← c92209, ← c92210] at e_1503_12
  have e_1503_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_13
  simp only [← c92211, ← c92212, k2, ← c92213] at e_1503_13
  have hr := e_1503_13
  simp only [e_1503_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14482)) := by
  have c92214 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92215 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92216 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92217 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92218 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92219 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_14
  simp only [← c92214, ← c92215, ← c92216] at e_1503_14
  have e_1503_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_15
  simp only [← c92217, ← c92218, k2, ← c92219] at e_1503_15
  have hr := e_1503_15
  simp only [e_1503_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14483)) := by
  have c92220 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92221 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92222 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92223 := (publicBatchWrapper89_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92224 := (publicBatchWrapper89_copies2882 a h).1
  have c92225 := (publicBatchWrapper89_copies2882 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_16
  simp only [← c92220, ← c92221, ← c92222] at e_1503_16
  have e_1503_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_17
  simp only [← c92223, ← c92224, k2, ← c92225] at e_1503_17
  have hr := e_1503_17
  simp only [e_1503_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1503 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14484)) := by
  have c92226 := (publicBatchWrapper89_copies2882 a h).2.2.1
  have c92227 := (publicBatchWrapper89_copies2882 a h).2.2.2.1
  have c92228 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.1
  have c92229 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.1
  have c92230 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.1
  have c92231 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1503_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_18
  simp only [← c92226, ← c92227, ← c92228] at e_1503_18
  have e_1503_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_19
  simp only [← c92229, ← c92230, k2, ← c92231] at e_1503_19
  have hr := e_1503_19
  simp only [e_1503_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14485)) := by
  have c92232 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.1
  have c92233 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.1
  have c92234 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92235 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92236 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92237 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_0
  simp only [← c92232, ← c92233, ← c92234] at e_1504_0
  have e_1504_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_1
  simp only [← c92235, ← c92236, k2, ← c92237] at e_1504_1
  have hr := e_1504_1
  simp only [e_1504_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14486)) := by
  have c92238 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92239 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92240 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92241 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92242 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92243 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_2
  simp only [← c92238, ← c92239, ← c92240] at e_1504_2
  have e_1504_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_3
  simp only [← c92241, ← c92242, k2, ← c92243] at e_1504_3
  have hr := e_1504_3
  simp only [e_1504_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14487)) := by
  have c92244 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92245 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92246 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92247 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92248 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92249 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_4
  simp only [← c92244, ← c92245, ← c92246] at e_1504_4
  have e_1504_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_5
  simp only [← c92247, ← c92248, k2, ← c92249] at e_1504_5
  have hr := e_1504_5
  simp only [e_1504_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14488)) := by
  have c92250 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92251 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92252 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92253 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92254 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92255 := (publicBatchWrapper89_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_6
  simp only [← c92250, ← c92251, ← c92252] at e_1504_6
  have e_1504_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_7
  simp only [← c92253, ← c92254, k2, ← c92255] at e_1504_7
  have hr := e_1504_7
  simp only [e_1504_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14489)) := by
  have c92256 := (publicBatchWrapper89_copies2883 a h).1
  have c92257 := (publicBatchWrapper89_copies2883 a h).2.1
  have c92258 := (publicBatchWrapper89_copies2883 a h).2.2.1
  have c92259 := (publicBatchWrapper89_copies2883 a h).2.2.2.1
  have c92260 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.1
  have c92261 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_8
  simp only [← c92256, ← c92257, ← c92258] at e_1504_8
  have e_1504_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_9
  simp only [← c92259, ← c92260, k2, ← c92261] at e_1504_9
  have hr := e_1504_9
  simp only [e_1504_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14490)) := by
  have c92262 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.1
  have c92263 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.1
  have c92264 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.1
  have c92265 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.1
  have c92266 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92267 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_10
  simp only [← c92262, ← c92263, ← c92264] at e_1504_10
  have e_1504_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_11
  simp only [← c92265, ← c92266, k2, ← c92267] at e_1504_11
  have hr := e_1504_11
  simp only [e_1504_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14491)) := by
  have c92268 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92269 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92270 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92271 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92272 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92273 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_12
  simp only [← c92268, ← c92269, ← c92270] at e_1504_12
  have e_1504_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_13
  simp only [← c92271, ← c92272, k2, ← c92273] at e_1504_13
  have hr := e_1504_13
  simp only [e_1504_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14492)) := by
  have c92274 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92275 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92276 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92277 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92278 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92279 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_14
  simp only [← c92274, ← c92275, ← c92276] at e_1504_14
  have e_1504_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_15
  simp only [← c92277, ← c92278, k2, ← c92279] at e_1504_15
  have hr := e_1504_15
  simp only [e_1504_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14493)) := by
  have c92280 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92281 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92282 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92283 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92284 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92285 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_16
  simp only [← c92280, ← c92281, ← c92282] at e_1504_16
  have e_1504_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_17
  simp only [← c92283, ← c92284, k2, ← c92285] at e_1504_17
  have hr := e_1504_17
  simp only [e_1504_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1504 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14494)) := by
  have c92286 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92287 := (publicBatchWrapper89_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92288 := (publicBatchWrapper89_copies2884 a h).1
  have c92289 := (publicBatchWrapper89_copies2884 a h).2.1
  have c92290 := (publicBatchWrapper89_copies2884 a h).2.2.1
  have c92291 := (publicBatchWrapper89_copies2884 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1504_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_18
  simp only [← c92286, ← c92287, ← c92288] at e_1504_18
  have e_1504_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_19
  simp only [← c92289, ← c92290, k2, ← c92291] at e_1504_19
  have hr := e_1504_19
  simp only [e_1504_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14495)) := by
  have c92292 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.1
  have c92293 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.1
  have c92294 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.1
  have c92295 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.1
  have c92296 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.1
  have c92297 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_0
  simp only [← c92292, ← c92293, ← c92294] at e_1505_0
  have e_1505_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_1
  simp only [← c92295, ← c92296, k2, ← c92297] at e_1505_1
  have hr := e_1505_1
  simp only [e_1505_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14496)) := by
  have c92298 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92299 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92300 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92301 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92302 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92303 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_2
  simp only [← c92298, ← c92299, ← c92300] at e_1505_2
  have e_1505_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_3
  simp only [← c92301, ← c92302, k2, ← c92303] at e_1505_3
  have hr := e_1505_3
  simp only [e_1505_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14497)) := by
  have c92304 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92305 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92306 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92307 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92308 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92309 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_4
  simp only [← c92304, ← c92305, ← c92306] at e_1505_4
  have e_1505_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_5
  simp only [← c92307, ← c92308, k2, ← c92309] at e_1505_5
  have hr := e_1505_5
  simp only [e_1505_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14498)) := by
  have c92310 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92311 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92312 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92313 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92314 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92315 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_6
  simp only [← c92310, ← c92311, ← c92312] at e_1505_6
  have e_1505_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_7
  simp only [← c92313, ← c92314, k2, ← c92315] at e_1505_7
  have hr := e_1505_7
  simp only [e_1505_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14499)) := by
  have c92316 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92317 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92318 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92319 := (publicBatchWrapper89_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92320 := (publicBatchWrapper89_copies2885 a h).1
  have c92321 := (publicBatchWrapper89_copies2885 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_8
  simp only [← c92316, ← c92317, ← c92318] at e_1505_8
  have e_1505_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_9
  simp only [← c92319, ← c92320, k2, ← c92321] at e_1505_9
  have hr := e_1505_9
  simp only [e_1505_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14500)) := by
  have c92322 := (publicBatchWrapper89_copies2885 a h).2.2.1
  have c92323 := (publicBatchWrapper89_copies2885 a h).2.2.2.1
  have c92324 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.1
  have c92325 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.1
  have c92326 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.1
  have c92327 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_10
  simp only [← c92322, ← c92323, ← c92324] at e_1505_10
  have e_1505_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_11
  simp only [← c92325, ← c92326, k2, ← c92327] at e_1505_11
  have hr := e_1505_11
  simp only [e_1505_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14501)) := by
  have c92328 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.1
  have c92329 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.1
  have c92330 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92331 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92332 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92333 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_12
  simp only [← c92328, ← c92329, ← c92330] at e_1505_12
  have e_1505_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_13
  simp only [← c92331, ← c92332, k2, ← c92333] at e_1505_13
  have hr := e_1505_13
  simp only [e_1505_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14502)) := by
  have c92334 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92335 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92336 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92337 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92338 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92339 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_14
  simp only [← c92334, ← c92335, ← c92336] at e_1505_14
  have e_1505_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_15
  simp only [← c92337, ← c92338, k2, ← c92339] at e_1505_15
  have hr := e_1505_15
  simp only [e_1505_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14503)) := by
  have c92340 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92341 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92342 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92343 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92344 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92345 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_16
  simp only [← c92340, ← c92341, ← c92342] at e_1505_16
  have e_1505_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_17
  simp only [← c92343, ← c92344, k2, ← c92345] at e_1505_17
  have hr := e_1505_17
  simp only [e_1505_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1505 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14504)) := by
  have c92346 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92347 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92348 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92349 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92350 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92351 := (publicBatchWrapper89_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1505_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_18
  simp only [← c92346, ← c92347, ← c92348] at e_1505_18
  have e_1505_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_19
  simp only [← c92349, ← c92350, k2, ← c92351] at e_1505_19
  have hr := e_1505_19
  simp only [e_1505_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14505)) := by
  have c92352 := (publicBatchWrapper89_copies2886 a h).1
  have c92353 := (publicBatchWrapper89_copies2886 a h).2.1
  have c92354 := (publicBatchWrapper89_copies2886 a h).2.2.1
  have c92355 := (publicBatchWrapper89_copies2886 a h).2.2.2.1
  have c92356 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.1
  have c92357 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_0
  simp only [← c92352, ← c92353, ← c92354] at e_1506_0
  have e_1506_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_1
  simp only [← c92355, ← c92356, k2, ← c92357] at e_1506_1
  have hr := e_1506_1
  simp only [e_1506_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14506)) := by
  have c92358 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.1
  have c92359 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.1
  have c92360 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.1
  have c92361 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.1
  have c92362 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92363 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_2
  simp only [← c92358, ← c92359, ← c92360] at e_1506_2
  have e_1506_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_3
  simp only [← c92361, ← c92362, k2, ← c92363] at e_1506_3
  have hr := e_1506_3
  simp only [e_1506_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14507)) := by
  have c92364 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92365 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92366 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92367 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92368 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92369 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_4
  simp only [← c92364, ← c92365, ← c92366] at e_1506_4
  have e_1506_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_5
  simp only [← c92367, ← c92368, k2, ← c92369] at e_1506_5
  have hr := e_1506_5
  simp only [e_1506_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14508)) := by
  have c92370 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92371 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92372 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92373 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92374 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92375 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_6
  simp only [← c92370, ← c92371, ← c92372] at e_1506_6
  have e_1506_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_7
  simp only [← c92373, ← c92374, k2, ← c92375] at e_1506_7
  have hr := e_1506_7
  simp only [e_1506_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14509)) := by
  have c92376 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92377 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92378 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92379 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92380 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92381 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_8
  simp only [← c92376, ← c92377, ← c92378] at e_1506_8
  have e_1506_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_9
  simp only [← c92379, ← c92380, k2, ← c92381] at e_1506_9
  have hr := e_1506_9
  simp only [e_1506_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14510)) := by
  have c92382 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92383 := (publicBatchWrapper89_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92384 := (publicBatchWrapper89_copies2887 a h).1
  have c92385 := (publicBatchWrapper89_copies2887 a h).2.1
  have c92386 := (publicBatchWrapper89_copies2887 a h).2.2.1
  have c92387 := (publicBatchWrapper89_copies2887 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_10
  simp only [← c92382, ← c92383, ← c92384] at e_1506_10
  have e_1506_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_11
  simp only [← c92385, ← c92386, k2, ← c92387] at e_1506_11
  have hr := e_1506_11
  simp only [e_1506_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14511)) := by
  have c92388 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.1
  have c92389 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.1
  have c92390 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.1
  have c92391 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.1
  have c92392 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.1
  have c92393 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_12
  simp only [← c92388, ← c92389, ← c92390] at e_1506_12
  have e_1506_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_13
  simp only [← c92391, ← c92392, k2, ← c92393] at e_1506_13
  have hr := e_1506_13
  simp only [e_1506_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14512)) := by
  have c92394 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92395 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92396 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92397 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92398 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92399 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_14
  simp only [← c92394, ← c92395, ← c92396] at e_1506_14
  have e_1506_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_15
  simp only [← c92397, ← c92398, k2, ← c92399] at e_1506_15
  have hr := e_1506_15
  simp only [e_1506_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14513)) := by
  have c92400 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92401 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92402 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92403 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92404 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92405 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_16
  simp only [← c92400, ← c92401, ← c92402] at e_1506_16
  have e_1506_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_17
  simp only [← c92403, ← c92404, k2, ← c92405] at e_1506_17
  have hr := e_1506_17
  simp only [e_1506_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1506 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14514)) := by
  have c92406 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92407 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92408 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92409 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92410 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92411 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1506_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_18
  simp only [← c92406, ← c92407, ← c92408] at e_1506_18
  have e_1506_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_19
  simp only [← c92409, ← c92410, k2, ← c92411] at e_1506_19
  have hr := e_1506_19
  simp only [e_1506_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14515)) := by
  have c92412 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92413 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92414 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92415 := (publicBatchWrapper89_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92416 := (publicBatchWrapper89_copies2888 a h).1
  have c92417 := (publicBatchWrapper89_copies2888 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_0
  simp only [← c92412, ← c92413, ← c92414] at e_1507_0
  have e_1507_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_1
  simp only [← c92415, ← c92416, k2, ← c92417] at e_1507_1
  have hr := e_1507_1
  simp only [e_1507_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14516)) := by
  have c92418 := (publicBatchWrapper89_copies2888 a h).2.2.1
  have c92419 := (publicBatchWrapper89_copies2888 a h).2.2.2.1
  have c92420 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.1
  have c92421 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.1
  have c92422 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.1
  have c92423 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_2
  simp only [← c92418, ← c92419, ← c92420] at e_1507_2
  have e_1507_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_3
  simp only [← c92421, ← c92422, k2, ← c92423] at e_1507_3
  have hr := e_1507_3
  simp only [e_1507_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14517)) := by
  have c92424 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.1
  have c92425 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.1
  have c92426 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92427 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92428 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92429 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_4
  simp only [← c92424, ← c92425, ← c92426] at e_1507_4
  have e_1507_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_5
  simp only [← c92427, ← c92428, k2, ← c92429] at e_1507_5
  have hr := e_1507_5
  simp only [e_1507_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14518)) := by
  have c92430 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92431 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92432 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92433 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92434 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92435 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_6
  simp only [← c92430, ← c92431, ← c92432] at e_1507_6
  have e_1507_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_7
  simp only [← c92433, ← c92434, k2, ← c92435] at e_1507_7
  have hr := e_1507_7
  simp only [e_1507_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14519)) := by
  have c92436 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92437 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92438 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92439 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92440 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92441 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_8
  simp only [← c92436, ← c92437, ← c92438] at e_1507_8
  have e_1507_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_9
  simp only [← c92439, ← c92440, k2, ← c92441] at e_1507_9
  have hr := e_1507_9
  simp only [e_1507_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14520)) := by
  have c92442 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92443 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92444 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92445 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92446 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92447 := (publicBatchWrapper89_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_10
  simp only [← c92442, ← c92443, ← c92444] at e_1507_10
  have e_1507_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_11
  simp only [← c92445, ← c92446, k2, ← c92447] at e_1507_11
  have hr := e_1507_11
  simp only [e_1507_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14521)) := by
  have c92448 := (publicBatchWrapper89_copies2889 a h).1
  have c92449 := (publicBatchWrapper89_copies2889 a h).2.1
  have c92450 := (publicBatchWrapper89_copies2889 a h).2.2.1
  have c92451 := (publicBatchWrapper89_copies2889 a h).2.2.2.1
  have c92452 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.1
  have c92453 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_12
  simp only [← c92448, ← c92449, ← c92450] at e_1507_12
  have e_1507_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_13
  simp only [← c92451, ← c92452, k2, ← c92453] at e_1507_13
  have hr := e_1507_13
  simp only [e_1507_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14522)) := by
  have c92454 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.1
  have c92455 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.1
  have c92456 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.1
  have c92457 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.1
  have c92458 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92459 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_14
  simp only [← c92454, ← c92455, ← c92456] at e_1507_14
  have e_1507_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_15
  simp only [← c92457, ← c92458, k2, ← c92459] at e_1507_15
  have hr := e_1507_15
  simp only [e_1507_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14523)) := by
  have c92460 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92461 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92462 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92463 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92464 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92465 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_16
  simp only [← c92460, ← c92461, ← c92462] at e_1507_16
  have e_1507_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_17
  simp only [← c92463, ← c92464, k2, ← c92465] at e_1507_17
  have hr := e_1507_17
  simp only [e_1507_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1507 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14524)) := by
  have c92466 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92467 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92468 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92469 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92470 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92471 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1507_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_18
  simp only [← c92466, ← c92467, ← c92468] at e_1507_18
  have e_1507_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_19
  simp only [← c92469, ← c92470, k2, ← c92471] at e_1507_19
  have hr := e_1507_19
  simp only [e_1507_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14525)) := by
  have c92472 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92473 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92474 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92475 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92476 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92477 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_0
  simp only [← c92472, ← c92473, ← c92474] at e_1508_0
  have e_1508_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_1
  simp only [← c92475, ← c92476, k2, ← c92477] at e_1508_1
  have hr := e_1508_1
  simp only [e_1508_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14526)) := by
  have c92478 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92479 := (publicBatchWrapper89_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92480 := (publicBatchWrapper89_copies2890 a h).1
  have c92481 := (publicBatchWrapper89_copies2890 a h).2.1
  have c92482 := (publicBatchWrapper89_copies2890 a h).2.2.1
  have c92483 := (publicBatchWrapper89_copies2890 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_2
  simp only [← c92478, ← c92479, ← c92480] at e_1508_2
  have e_1508_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_3
  simp only [← c92481, ← c92482, k2, ← c92483] at e_1508_3
  have hr := e_1508_3
  simp only [e_1508_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14527)) := by
  have c92484 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.1
  have c92485 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.1
  have c92486 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.1
  have c92487 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.1
  have c92488 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.1
  have c92489 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_4
  simp only [← c92484, ← c92485, ← c92486] at e_1508_4
  have e_1508_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_5
  simp only [← c92487, ← c92488, k2, ← c92489] at e_1508_5
  have hr := e_1508_5
  simp only [e_1508_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14528)) := by
  have c92490 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92491 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92492 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92493 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92494 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92495 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_6
  simp only [← c92490, ← c92491, ← c92492] at e_1508_6
  have e_1508_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_7
  simp only [← c92493, ← c92494, k2, ← c92495] at e_1508_7
  have hr := e_1508_7
  simp only [e_1508_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14529)) := by
  have c92496 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92497 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92498 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92499 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92500 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92501 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_8
  simp only [← c92496, ← c92497, ← c92498] at e_1508_8
  have e_1508_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_9
  simp only [← c92499, ← c92500, k2, ← c92501] at e_1508_9
  have hr := e_1508_9
  simp only [e_1508_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14530)) := by
  have c92502 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92503 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92504 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92505 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92506 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92507 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_10
  simp only [← c92502, ← c92503, ← c92504] at e_1508_10
  have e_1508_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_11
  simp only [← c92505, ← c92506, k2, ← c92507] at e_1508_11
  have hr := e_1508_11
  simp only [e_1508_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14531)) := by
  have c92508 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92509 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92510 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92511 := (publicBatchWrapper89_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92512 := (publicBatchWrapper89_copies2891 a h).1
  have c92513 := (publicBatchWrapper89_copies2891 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_12
  simp only [← c92508, ← c92509, ← c92510] at e_1508_12
  have e_1508_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_13
  simp only [← c92511, ← c92512, k2, ← c92513] at e_1508_13
  have hr := e_1508_13
  simp only [e_1508_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14532)) := by
  have c92514 := (publicBatchWrapper89_copies2891 a h).2.2.1
  have c92515 := (publicBatchWrapper89_copies2891 a h).2.2.2.1
  have c92516 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.1
  have c92517 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.1
  have c92518 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.1
  have c92519 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_14
  simp only [← c92514, ← c92515, ← c92516] at e_1508_14
  have e_1508_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_15
  simp only [← c92517, ← c92518, k2, ← c92519] at e_1508_15
  have hr := e_1508_15
  simp only [e_1508_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14533)) := by
  have c92520 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.1
  have c92521 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.1
  have c92522 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92523 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92524 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92525 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_16
  simp only [← c92520, ← c92521, ← c92522] at e_1508_16
  have e_1508_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_17
  simp only [← c92523, ← c92524, k2, ← c92525] at e_1508_17
  have hr := e_1508_17
  simp only [e_1508_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1508 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14534)) := by
  have c92526 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92527 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92528 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92529 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92530 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92531 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1508_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_18
  simp only [← c92526, ← c92527, ← c92528] at e_1508_18
  have e_1508_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_19
  simp only [← c92529, ← c92530, k2, ← c92531] at e_1508_19
  have hr := e_1508_19
  simp only [e_1508_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14535)) := by
  have c92532 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92533 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92534 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92535 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92536 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92537 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_0
  simp only [← c92532, ← c92533, ← c92534] at e_1509_0
  have e_1509_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_1
  simp only [← c92535, ← c92536, k2, ← c92537] at e_1509_1
  have hr := e_1509_1
  simp only [e_1509_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14536)) := by
  have c92538 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92539 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92540 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92541 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92542 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92543 := (publicBatchWrapper89_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_2
  simp only [← c92538, ← c92539, ← c92540] at e_1509_2
  have e_1509_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_3
  simp only [← c92541, ← c92542, k2, ← c92543] at e_1509_3
  have hr := e_1509_3
  simp only [e_1509_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14537)) := by
  have c92544 := (publicBatchWrapper89_copies2892 a h).1
  have c92545 := (publicBatchWrapper89_copies2892 a h).2.1
  have c92546 := (publicBatchWrapper89_copies2892 a h).2.2.1
  have c92547 := (publicBatchWrapper89_copies2892 a h).2.2.2.1
  have c92548 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.1
  have c92549 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_4
  simp only [← c92544, ← c92545, ← c92546] at e_1509_4
  have e_1509_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_5
  simp only [← c92547, ← c92548, k2, ← c92549] at e_1509_5
  have hr := e_1509_5
  simp only [e_1509_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14538)) := by
  have c92550 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.1
  have c92551 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.1
  have c92552 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.1
  have c92553 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.1
  have c92554 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92555 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_6
  simp only [← c92550, ← c92551, ← c92552] at e_1509_6
  have e_1509_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_7
  simp only [← c92553, ← c92554, k2, ← c92555] at e_1509_7
  have hr := e_1509_7
  simp only [e_1509_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14539)) := by
  have c92556 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92557 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92558 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92559 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92560 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92561 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_8
  simp only [← c92556, ← c92557, ← c92558] at e_1509_8
  have e_1509_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_9
  simp only [← c92559, ← c92560, k2, ← c92561] at e_1509_9
  have hr := e_1509_9
  simp only [e_1509_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14540)) := by
  have c92562 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92563 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92564 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92565 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92566 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92567 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_10
  simp only [← c92562, ← c92563, ← c92564] at e_1509_10
  have e_1509_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_11
  simp only [← c92565, ← c92566, k2, ← c92567] at e_1509_11
  have hr := e_1509_11
  simp only [e_1509_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14541)) := by
  have c92568 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92569 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92570 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92571 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92572 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92573 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_12
  simp only [← c92568, ← c92569, ← c92570] at e_1509_12
  have e_1509_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_13
  simp only [← c92571, ← c92572, k2, ← c92573] at e_1509_13
  have hr := e_1509_13
  simp only [e_1509_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14542)) := by
  have c92574 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92575 := (publicBatchWrapper89_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92576 := (publicBatchWrapper89_copies2893 a h).1
  have c92577 := (publicBatchWrapper89_copies2893 a h).2.1
  have c92578 := (publicBatchWrapper89_copies2893 a h).2.2.1
  have c92579 := (publicBatchWrapper89_copies2893 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_14
  simp only [← c92574, ← c92575, ← c92576] at e_1509_14
  have e_1509_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_15
  simp only [← c92577, ← c92578, k2, ← c92579] at e_1509_15
  have hr := e_1509_15
  simp only [e_1509_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14543)) := by
  have c92580 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.1
  have c92581 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.1
  have c92582 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.1
  have c92583 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.1
  have c92584 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.1
  have c92585 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_16
  simp only [← c92580, ← c92581, ← c92582] at e_1509_16
  have e_1509_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_17
  simp only [← c92583, ← c92584, k2, ← c92585] at e_1509_17
  have hr := e_1509_17
  simp only [e_1509_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1509 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14544)) := by
  have c92586 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92587 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92588 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92589 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92590 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92591 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1509_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_18
  simp only [← c92586, ← c92587, ← c92588] at e_1509_18
  have e_1509_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_19
  simp only [← c92589, ← c92590, k2, ← c92591] at e_1509_19
  have hr := e_1509_19
  simp only [e_1509_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14545)) := by
  have c92592 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92593 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92594 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92595 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92596 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92597 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_0
  simp only [← c92592, ← c92593, ← c92594] at e_1510_0
  have e_1510_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_1
  simp only [← c92595, ← c92596, k2, ← c92597] at e_1510_1
  have hr := e_1510_1
  simp only [e_1510_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14546)) := by
  have c92598 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92599 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92600 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92601 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92602 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92603 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_2
  simp only [← c92598, ← c92599, ← c92600] at e_1510_2
  have e_1510_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_3
  simp only [← c92601, ← c92602, k2, ← c92603] at e_1510_3
  have hr := e_1510_3
  simp only [e_1510_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14547)) := by
  have c92604 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92605 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92606 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92607 := (publicBatchWrapper89_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92608 := (publicBatchWrapper89_copies2894 a h).1
  have c92609 := (publicBatchWrapper89_copies2894 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_4
  simp only [← c92604, ← c92605, ← c92606] at e_1510_4
  have e_1510_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_5
  simp only [← c92607, ← c92608, k2, ← c92609] at e_1510_5
  have hr := e_1510_5
  simp only [e_1510_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14548)) := by
  have c92610 := (publicBatchWrapper89_copies2894 a h).2.2.1
  have c92611 := (publicBatchWrapper89_copies2894 a h).2.2.2.1
  have c92612 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.1
  have c92613 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.1
  have c92614 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.1
  have c92615 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_6
  simp only [← c92610, ← c92611, ← c92612] at e_1510_6
  have e_1510_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_7
  simp only [← c92613, ← c92614, k2, ← c92615] at e_1510_7
  have hr := e_1510_7
  simp only [e_1510_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14549)) := by
  have c92616 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.1
  have c92617 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.1
  have c92618 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92619 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92620 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92621 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_8
  simp only [← c92616, ← c92617, ← c92618] at e_1510_8
  have e_1510_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_9
  simp only [← c92619, ← c92620, k2, ← c92621] at e_1510_9
  have hr := e_1510_9
  simp only [e_1510_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14550)) := by
  have c92622 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92623 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92624 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92625 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92626 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92627 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_10
  simp only [← c92622, ← c92623, ← c92624] at e_1510_10
  have e_1510_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_11
  simp only [← c92625, ← c92626, k2, ← c92627] at e_1510_11
  have hr := e_1510_11
  simp only [e_1510_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14551)) := by
  have c92628 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92629 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92630 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92631 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92632 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92633 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_12
  simp only [← c92628, ← c92629, ← c92630] at e_1510_12
  have e_1510_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_13
  simp only [← c92631, ← c92632, k2, ← c92633] at e_1510_13
  have hr := e_1510_13
  simp only [e_1510_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14552)) := by
  have c92634 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92635 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92636 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92637 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92638 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92639 := (publicBatchWrapper89_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_14
  simp only [← c92634, ← c92635, ← c92636] at e_1510_14
  have e_1510_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_15
  simp only [← c92637, ← c92638, k2, ← c92639] at e_1510_15
  have hr := e_1510_15
  simp only [e_1510_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14553)) := by
  have c92640 := (publicBatchWrapper89_copies2895 a h).1
  have c92641 := (publicBatchWrapper89_copies2895 a h).2.1
  have c92642 := (publicBatchWrapper89_copies2895 a h).2.2.1
  have c92643 := (publicBatchWrapper89_copies2895 a h).2.2.2.1
  have c92644 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.1
  have c92645 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_16
  simp only [← c92640, ← c92641, ← c92642] at e_1510_16
  have e_1510_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_17
  simp only [← c92643, ← c92644, k2, ← c92645] at e_1510_17
  have hr := e_1510_17
  simp only [e_1510_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1510 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14554)) := by
  have c92646 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.1
  have c92647 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.1
  have c92648 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.1
  have c92649 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.1
  have c92650 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92651 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1510_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_18
  simp only [← c92646, ← c92647, ← c92648] at e_1510_18
  have e_1510_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_19
  simp only [← c92649, ← c92650, k2, ← c92651] at e_1510_19
  have hr := e_1510_19
  simp only [e_1510_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14555)) := by
  have c92652 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92653 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92654 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92655 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92656 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92657 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_0
  simp only [← c92652, ← c92653, ← c92654] at e_1511_0
  have e_1511_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_1
  simp only [← c92655, ← c92656, k2, ← c92657] at e_1511_1
  have hr := e_1511_1
  simp only [e_1511_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14556)) := by
  have c92658 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92659 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92660 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92661 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92662 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92663 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_2
  simp only [← c92658, ← c92659, ← c92660] at e_1511_2
  have e_1511_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_3
  simp only [← c92661, ← c92662, k2, ← c92663] at e_1511_3
  have hr := e_1511_3
  simp only [e_1511_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14557)) := by
  have c92664 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92665 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92666 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92667 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92668 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92669 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_4
  simp only [← c92664, ← c92665, ← c92666] at e_1511_4
  have e_1511_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_5
  simp only [← c92667, ← c92668, k2, ← c92669] at e_1511_5
  have hr := e_1511_5
  simp only [e_1511_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14558)) := by
  have c92670 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92671 := (publicBatchWrapper89_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92672 := (publicBatchWrapper89_copies2896 a h).1
  have c92673 := (publicBatchWrapper89_copies2896 a h).2.1
  have c92674 := (publicBatchWrapper89_copies2896 a h).2.2.1
  have c92675 := (publicBatchWrapper89_copies2896 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_6
  simp only [← c92670, ← c92671, ← c92672] at e_1511_6
  have e_1511_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_7
  simp only [← c92673, ← c92674, k2, ← c92675] at e_1511_7
  have hr := e_1511_7
  simp only [e_1511_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14559)) := by
  have c92676 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.1
  have c92677 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.1
  have c92678 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.1
  have c92679 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.1
  have c92680 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.1
  have c92681 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_8
  simp only [← c92676, ← c92677, ← c92678] at e_1511_8
  have e_1511_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_9
  simp only [← c92679, ← c92680, k2, ← c92681] at e_1511_9
  have hr := e_1511_9
  simp only [e_1511_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14560)) := by
  have c92682 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92683 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92684 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92685 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92686 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92687 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_10
  simp only [← c92682, ← c92683, ← c92684] at e_1511_10
  have e_1511_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_11
  simp only [← c92685, ← c92686, k2, ← c92687] at e_1511_11
  have hr := e_1511_11
  simp only [e_1511_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14561)) := by
  have c92688 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92689 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92690 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92691 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92692 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92693 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_12
  simp only [← c92688, ← c92689, ← c92690] at e_1511_12
  have e_1511_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_13
  simp only [← c92691, ← c92692, k2, ← c92693] at e_1511_13
  have hr := e_1511_13
  simp only [e_1511_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14562)) := by
  have c92694 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92695 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92696 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92697 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92698 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92699 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_14
  simp only [← c92694, ← c92695, ← c92696] at e_1511_14
  have e_1511_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_15
  simp only [← c92697, ← c92698, k2, ← c92699] at e_1511_15
  have hr := e_1511_15
  simp only [e_1511_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14563)) := by
  have c92700 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92701 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92702 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92703 := (publicBatchWrapper89_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92704 := (publicBatchWrapper89_copies2897 a h).1
  have c92705 := (publicBatchWrapper89_copies2897 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_16
  simp only [← c92700, ← c92701, ← c92702] at e_1511_16
  have e_1511_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_17
  simp only [← c92703, ← c92704, k2, ← c92705] at e_1511_17
  have hr := e_1511_17
  simp only [e_1511_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1511 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14564)) := by
  have c92706 := (publicBatchWrapper89_copies2897 a h).2.2.1
  have c92707 := (publicBatchWrapper89_copies2897 a h).2.2.2.1
  have c92708 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.1
  have c92709 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.1
  have c92710 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.1
  have c92711 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1511_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_18
  simp only [← c92706, ← c92707, ← c92708] at e_1511_18
  have e_1511_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_19
  simp only [← c92709, ← c92710, k2, ← c92711] at e_1511_19
  have hr := e_1511_19
  simp only [e_1511_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14565)) := by
  have c92712 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.1
  have c92713 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.1
  have c92714 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92715 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92716 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92717 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_0
  simp only [← c92712, ← c92713, ← c92714] at e_1512_0
  have e_1512_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_1
  simp only [← c92715, ← c92716, k2, ← c92717] at e_1512_1
  have hr := e_1512_1
  simp only [e_1512_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14566)) := by
  have c92718 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92719 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92720 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92721 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92722 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92723 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_2
  simp only [← c92718, ← c92719, ← c92720] at e_1512_2
  have e_1512_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_3
  simp only [← c92721, ← c92722, k2, ← c92723] at e_1512_3
  have hr := e_1512_3
  simp only [e_1512_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14567)) := by
  have c92724 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92725 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92726 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92727 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92728 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92729 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_4
  simp only [← c92724, ← c92725, ← c92726] at e_1512_4
  have e_1512_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_5
  simp only [← c92727, ← c92728, k2, ← c92729] at e_1512_5
  have hr := e_1512_5
  simp only [e_1512_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14568)) := by
  have c92730 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92731 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92732 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92733 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92734 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92735 := (publicBatchWrapper89_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_6
  simp only [← c92730, ← c92731, ← c92732] at e_1512_6
  have e_1512_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_7
  simp only [← c92733, ← c92734, k2, ← c92735] at e_1512_7
  have hr := e_1512_7
  simp only [e_1512_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14569)) := by
  have c92736 := (publicBatchWrapper89_copies2898 a h).1
  have c92737 := (publicBatchWrapper89_copies2898 a h).2.1
  have c92738 := (publicBatchWrapper89_copies2898 a h).2.2.1
  have c92739 := (publicBatchWrapper89_copies2898 a h).2.2.2.1
  have c92740 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.1
  have c92741 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_8
  simp only [← c92736, ← c92737, ← c92738] at e_1512_8
  have e_1512_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_9
  simp only [← c92739, ← c92740, k2, ← c92741] at e_1512_9
  have hr := e_1512_9
  simp only [e_1512_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14610)) := by
  have c92742 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.1
  have c92743 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.1
  have c92744 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.1
  have c92745 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.1
  have c92746 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92747 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_10
  simp only [← c92742, ← c92743, ← c92744] at e_1512_10
  have e_1512_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_11
  simp only [← c92745, ← c92746, k2, ← c92747] at e_1512_11
  have hr := e_1512_11
  simp only [e_1512_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14611)) := by
  have c92748 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92749 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92750 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92751 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92752 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92753 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_12
  simp only [← c92748, ← c92749, ← c92750] at e_1512_12
  have e_1512_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_13
  simp only [← c92751, ← c92752, k2, ← c92753] at e_1512_13
  have hr := e_1512_13
  simp only [e_1512_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14612)) := by
  have c92754 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92755 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92756 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92757 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92758 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92759 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_14
  simp only [← c92754, ← c92755, ← c92756] at e_1512_14
  have e_1512_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_15
  simp only [← c92757, ← c92758, k2, ← c92759] at e_1512_15
  have hr := e_1512_15
  simp only [e_1512_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14613)) := by
  have c92760 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92761 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92762 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92763 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92764 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92765 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_16
  simp only [← c92760, ← c92761, ← c92762] at e_1512_16
  have e_1512_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_17
  simp only [← c92763, ← c92764, k2, ← c92765] at e_1512_17
  have hr := e_1512_17
  simp only [e_1512_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1512 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14614)) := by
  have c92766 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92767 := (publicBatchWrapper89_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92768 := (publicBatchWrapper89_copies2899 a h).1
  have c92769 := (publicBatchWrapper89_copies2899 a h).2.1
  have c92770 := (publicBatchWrapper89_copies2899 a h).2.2.1
  have c92771 := (publicBatchWrapper89_copies2899 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1512_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_18
  simp only [← c92766, ← c92767, ← c92768] at e_1512_18
  have e_1512_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_19
  simp only [← c92769, ← c92770, k2, ← c92771] at e_1512_19
  have hr := e_1512_19
  simp only [e_1512_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14615)) := by
  have c92772 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.1
  have c92773 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.1
  have c92774 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.1
  have c92775 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.1
  have c92776 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.1
  have c92777 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_0
  simp only [← c92772, ← c92773, ← c92774] at e_1513_0
  have e_1513_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_1
  simp only [← c92775, ← c92776, k2, ← c92777] at e_1513_1
  have hr := e_1513_1
  simp only [e_1513_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14616)) := by
  have c92778 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92779 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92780 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92781 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92782 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92783 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_2
  simp only [← c92778, ← c92779, ← c92780] at e_1513_2
  have e_1513_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_3
  simp only [← c92781, ← c92782, k2, ← c92783] at e_1513_3
  have hr := e_1513_3
  simp only [e_1513_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14617)) := by
  have c92784 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92785 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92786 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92787 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92788 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92789 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_4
  simp only [← c92784, ← c92785, ← c92786] at e_1513_4
  have e_1513_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_5
  simp only [← c92787, ← c92788, k2, ← c92789] at e_1513_5
  have hr := e_1513_5
  simp only [e_1513_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14618)) := by
  have c92790 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92791 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92792 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92793 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92794 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92795 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_6
  simp only [← c92790, ← c92791, ← c92792] at e_1513_6
  have e_1513_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_7
  simp only [← c92793, ← c92794, k2, ← c92795] at e_1513_7
  have hr := e_1513_7
  simp only [e_1513_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14619)) := by
  have c92796 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92797 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92798 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92799 := (publicBatchWrapper89_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92800 := (publicBatchWrapper89_copies2900 a h).1
  have c92801 := (publicBatchWrapper89_copies2900 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_8
  simp only [← c92796, ← c92797, ← c92798] at e_1513_8
  have e_1513_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_9
  simp only [← c92799, ← c92800, k2, ← c92801] at e_1513_9
  have hr := e_1513_9
  simp only [e_1513_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14620)) := by
  have c92802 := (publicBatchWrapper89_copies2900 a h).2.2.1
  have c92803 := (publicBatchWrapper89_copies2900 a h).2.2.2.1
  have c92804 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.1
  have c92805 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.1
  have c92806 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.1
  have c92807 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_10
  simp only [← c92802, ← c92803, ← c92804] at e_1513_10
  have e_1513_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_11
  simp only [← c92805, ← c92806, k2, ← c92807] at e_1513_11
  have hr := e_1513_11
  simp only [e_1513_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14621)) := by
  have c92808 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.1
  have c92809 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.1
  have c92810 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92811 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92812 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92813 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_12
  simp only [← c92808, ← c92809, ← c92810] at e_1513_12
  have e_1513_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_13
  simp only [← c92811, ← c92812, k2, ← c92813] at e_1513_13
  have hr := e_1513_13
  simp only [e_1513_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14622)) := by
  have c92814 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92815 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92816 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92817 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92818 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92819 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_14
  simp only [← c92814, ← c92815, ← c92816] at e_1513_14
  have e_1513_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_15
  simp only [← c92817, ← c92818, k2, ← c92819] at e_1513_15
  have hr := e_1513_15
  simp only [e_1513_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14623)) := by
  have c92820 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92821 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92822 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92823 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92824 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92825 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_16
  simp only [← c92820, ← c92821, ← c92822] at e_1513_16
  have e_1513_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_17
  simp only [← c92823, ← c92824, k2, ← c92825] at e_1513_17
  have hr := e_1513_17
  simp only [e_1513_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1513 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14624)) := by
  have c92826 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92827 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92828 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92829 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92830 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92831 := (publicBatchWrapper89_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1513_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_18
  simp only [← c92826, ← c92827, ← c92828] at e_1513_18
  have e_1513_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_19
  simp only [← c92829, ← c92830, k2, ← c92831] at e_1513_19
  have hr := e_1513_19
  simp only [e_1513_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14625)) := by
  have c92832 := (publicBatchWrapper89_copies2901 a h).1
  have c92833 := (publicBatchWrapper89_copies2901 a h).2.1
  have c92834 := (publicBatchWrapper89_copies2901 a h).2.2.1
  have c92835 := (publicBatchWrapper89_copies2901 a h).2.2.2.1
  have c92836 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.1
  have c92837 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_0
  simp only [← c92832, ← c92833, ← c92834] at e_1514_0
  have e_1514_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_1
  simp only [← c92835, ← c92836, k2, ← c92837] at e_1514_1
  have hr := e_1514_1
  simp only [e_1514_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14626)) := by
  have c92838 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.1
  have c92839 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.1
  have c92840 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.1
  have c92841 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.1
  have c92842 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92843 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_2
  simp only [← c92838, ← c92839, ← c92840] at e_1514_2
  have e_1514_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_3
  simp only [← c92841, ← c92842, k2, ← c92843] at e_1514_3
  have hr := e_1514_3
  simp only [e_1514_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14627)) := by
  have c92844 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92845 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92846 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92847 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92848 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92849 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_4
  simp only [← c92844, ← c92845, ← c92846] at e_1514_4
  have e_1514_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_5
  simp only [← c92847, ← c92848, k2, ← c92849] at e_1514_5
  have hr := e_1514_5
  simp only [e_1514_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14628)) := by
  have c92850 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92851 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92852 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92853 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92854 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92855 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_6
  simp only [← c92850, ← c92851, ← c92852] at e_1514_6
  have e_1514_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_7
  simp only [← c92853, ← c92854, k2, ← c92855] at e_1514_7
  have hr := e_1514_7
  simp only [e_1514_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14629)) := by
  have c92856 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92857 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92858 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92859 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92860 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92861 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_8
  simp only [← c92856, ← c92857, ← c92858] at e_1514_8
  have e_1514_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_9
  simp only [← c92859, ← c92860, k2, ← c92861] at e_1514_9
  have hr := e_1514_9
  simp only [e_1514_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14630)) := by
  have c92862 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92863 := (publicBatchWrapper89_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92864 := (publicBatchWrapper89_copies2902 a h).1
  have c92865 := (publicBatchWrapper89_copies2902 a h).2.1
  have c92866 := (publicBatchWrapper89_copies2902 a h).2.2.1
  have c92867 := (publicBatchWrapper89_copies2902 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_10
  simp only [← c92862, ← c92863, ← c92864] at e_1514_10
  have e_1514_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_11
  simp only [← c92865, ← c92866, k2, ← c92867] at e_1514_11
  have hr := e_1514_11
  simp only [e_1514_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14631)) := by
  have c92868 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.1
  have c92869 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.1
  have c92870 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.1
  have c92871 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.1
  have c92872 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.1
  have c92873 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_12
  simp only [← c92868, ← c92869, ← c92870] at e_1514_12
  have e_1514_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_13
  simp only [← c92871, ← c92872, k2, ← c92873] at e_1514_13
  have hr := e_1514_13
  simp only [e_1514_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14632)) := by
  have c92874 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92875 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92876 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92877 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92878 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92879 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_14
  simp only [← c92874, ← c92875, ← c92876] at e_1514_14
  have e_1514_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_15
  simp only [← c92877, ← c92878, k2, ← c92879] at e_1514_15
  have hr := e_1514_15
  simp only [e_1514_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14633)) := by
  have c92880 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92881 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92882 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92883 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92884 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92885 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_16
  simp only [← c92880, ← c92881, ← c92882] at e_1514_16
  have e_1514_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_17
  simp only [← c92883, ← c92884, k2, ← c92885] at e_1514_17
  have hr := e_1514_17
  simp only [e_1514_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1514 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14634)) := by
  have c92886 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92887 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92888 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92889 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92890 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92891 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1514_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_18
  simp only [← c92886, ← c92887, ← c92888] at e_1514_18
  have e_1514_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_19
  simp only [← c92889, ← c92890, k2, ← c92891] at e_1514_19
  have hr := e_1514_19
  simp only [e_1514_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14635)) := by
  have c92892 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92893 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92894 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92895 := (publicBatchWrapper89_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92896 := (publicBatchWrapper89_copies2903 a h).1
  have c92897 := (publicBatchWrapper89_copies2903 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_0
  simp only [← c92892, ← c92893, ← c92894] at e_1515_0
  have e_1515_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_1
  simp only [← c92895, ← c92896, k2, ← c92897] at e_1515_1
  have hr := e_1515_1
  simp only [e_1515_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14636)) := by
  have c92898 := (publicBatchWrapper89_copies2903 a h).2.2.1
  have c92899 := (publicBatchWrapper89_copies2903 a h).2.2.2.1
  have c92900 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.1
  have c92901 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.1
  have c92902 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.1
  have c92903 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_2
  simp only [← c92898, ← c92899, ← c92900] at e_1515_2
  have e_1515_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_3
  simp only [← c92901, ← c92902, k2, ← c92903] at e_1515_3
  have hr := e_1515_3
  simp only [e_1515_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14637)) := by
  have c92904 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.1
  have c92905 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.1
  have c92906 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92907 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92908 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92909 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_4
  simp only [← c92904, ← c92905, ← c92906] at e_1515_4
  have e_1515_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_5
  simp only [← c92907, ← c92908, k2, ← c92909] at e_1515_5
  have hr := e_1515_5
  simp only [e_1515_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14638)) := by
  have c92910 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92911 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92912 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92913 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92914 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92915 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_6
  simp only [← c92910, ← c92911, ← c92912] at e_1515_6
  have e_1515_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_7
  simp only [← c92913, ← c92914, k2, ← c92915] at e_1515_7
  have hr := e_1515_7
  simp only [e_1515_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14639)) := by
  have c92916 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92917 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92918 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92919 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92920 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92921 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_8
  simp only [← c92916, ← c92917, ← c92918] at e_1515_8
  have e_1515_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_9
  simp only [← c92919, ← c92920, k2, ← c92921] at e_1515_9
  have hr := e_1515_9
  simp only [e_1515_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14640)) := by
  have c92922 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92923 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92924 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92925 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92926 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92927 := (publicBatchWrapper89_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_10
  simp only [← c92922, ← c92923, ← c92924] at e_1515_10
  have e_1515_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_11
  simp only [← c92925, ← c92926, k2, ← c92927] at e_1515_11
  have hr := e_1515_11
  simp only [e_1515_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14641)) := by
  have c92928 := (publicBatchWrapper89_copies2904 a h).1
  have c92929 := (publicBatchWrapper89_copies2904 a h).2.1
  have c92930 := (publicBatchWrapper89_copies2904 a h).2.2.1
  have c92931 := (publicBatchWrapper89_copies2904 a h).2.2.2.1
  have c92932 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.1
  have c92933 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_12
  simp only [← c92928, ← c92929, ← c92930] at e_1515_12
  have e_1515_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_13
  simp only [← c92931, ← c92932, k2, ← c92933] at e_1515_13
  have hr := e_1515_13
  simp only [e_1515_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14642)) := by
  have c92934 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.1
  have c92935 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.1
  have c92936 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.1
  have c92937 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.1
  have c92938 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92939 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_14
  simp only [← c92934, ← c92935, ← c92936] at e_1515_14
  have e_1515_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_15
  simp only [← c92937, ← c92938, k2, ← c92939] at e_1515_15
  have hr := e_1515_15
  simp only [e_1515_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14643)) := by
  have c92940 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92941 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92942 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92943 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92944 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92945 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_16
  simp only [← c92940, ← c92941, ← c92942] at e_1515_16
  have e_1515_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_17
  simp only [← c92943, ← c92944, k2, ← c92945] at e_1515_17
  have hr := e_1515_17
  simp only [e_1515_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1515 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14644)) := by
  have c92946 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92947 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92948 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92949 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92950 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92951 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1515_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_18
  simp only [← c92946, ← c92947, ← c92948] at e_1515_18
  have e_1515_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_19
  simp only [← c92949, ← c92950, k2, ← c92951] at e_1515_19
  have hr := e_1515_19
  simp only [e_1515_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14645)) := by
  have c92952 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92953 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92954 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92955 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92956 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92957 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_0
  simp only [← c92952, ← c92953, ← c92954] at e_1516_0
  have e_1516_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_1
  simp only [← c92955, ← c92956, k2, ← c92957] at e_1516_1
  have hr := e_1516_1
  simp only [e_1516_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14646)) := by
  have c92958 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92959 := (publicBatchWrapper89_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92960 := (publicBatchWrapper89_copies2905 a h).1
  have c92961 := (publicBatchWrapper89_copies2905 a h).2.1
  have c92962 := (publicBatchWrapper89_copies2905 a h).2.2.1
  have c92963 := (publicBatchWrapper89_copies2905 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_2
  simp only [← c92958, ← c92959, ← c92960] at e_1516_2
  have e_1516_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_3
  simp only [← c92961, ← c92962, k2, ← c92963] at e_1516_3
  have hr := e_1516_3
  simp only [e_1516_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14647)) := by
  have c92964 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.1
  have c92965 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.1
  have c92966 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.1
  have c92967 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.1
  have c92968 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.1
  have c92969 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_4
  simp only [← c92964, ← c92965, ← c92966] at e_1516_4
  have e_1516_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_5
  simp only [← c92967, ← c92968, k2, ← c92969] at e_1516_5
  have hr := e_1516_5
  simp only [e_1516_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14648)) := by
  have c92970 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92971 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92972 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92973 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92974 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92975 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_6
  simp only [← c92970, ← c92971, ← c92972] at e_1516_6
  have e_1516_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_7
  simp only [← c92973, ← c92974, k2, ← c92975] at e_1516_7
  have hr := e_1516_7
  simp only [e_1516_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14649)) := by
  have c92976 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92977 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92978 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92979 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92980 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92981 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_8
  simp only [← c92976, ← c92977, ← c92978] at e_1516_8
  have e_1516_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_9
  simp only [← c92979, ← c92980, k2, ← c92981] at e_1516_9
  have hr := e_1516_9
  simp only [e_1516_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14650)) := by
  have c92982 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92983 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92984 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92985 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92986 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92987 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_10
  simp only [← c92982, ← c92983, ← c92984] at e_1516_10
  have e_1516_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_11
  simp only [← c92985, ← c92986, k2, ← c92987] at e_1516_11
  have hr := e_1516_11
  simp only [e_1516_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14651)) := by
  have c92988 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92989 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92990 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92991 := (publicBatchWrapper89_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92992 := (publicBatchWrapper89_copies2906 a h).1
  have c92993 := (publicBatchWrapper89_copies2906 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_12
  simp only [← c92988, ← c92989, ← c92990] at e_1516_12
  have e_1516_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_13
  simp only [← c92991, ← c92992, k2, ← c92993] at e_1516_13
  have hr := e_1516_13
  simp only [e_1516_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14652)) := by
  have c92994 := (publicBatchWrapper89_copies2906 a h).2.2.1
  have c92995 := (publicBatchWrapper89_copies2906 a h).2.2.2.1
  have c92996 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.1
  have c92997 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.1
  have c92998 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.1
  have c92999 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_14
  simp only [← c92994, ← c92995, ← c92996] at e_1516_14
  have e_1516_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_15
  simp only [← c92997, ← c92998, k2, ← c92999] at e_1516_15
  have hr := e_1516_15
  simp only [e_1516_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14653)) := by
  have c93000 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.1
  have c93001 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.1
  have c93002 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93003 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93004 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93005 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_16
  simp only [← c93000, ← c93001, ← c93002] at e_1516_16
  have e_1516_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_17
  simp only [← c93003, ← c93004, k2, ← c93005] at e_1516_17
  have hr := e_1516_17
  simp only [e_1516_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1516 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14654)) := by
  have c93006 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93007 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93008 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93009 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93010 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93011 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1516_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_18
  simp only [← c93006, ← c93007, ← c93008] at e_1516_18
  have e_1516_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_19
  simp only [← c93009, ← c93010, k2, ← c93011] at e_1516_19
  have hr := e_1516_19
  simp only [e_1516_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14655)) := by
  have c93012 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93013 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93014 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93015 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93016 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93017 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_0
  simp only [← c93012, ← c93013, ← c93014] at e_1517_0
  have e_1517_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_1
  simp only [← c93015, ← c93016, k2, ← c93017] at e_1517_1
  have hr := e_1517_1
  simp only [e_1517_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14656)) := by
  have c93018 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93019 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93020 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93021 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93022 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93023 := (publicBatchWrapper89_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_2
  simp only [← c93018, ← c93019, ← c93020] at e_1517_2
  have e_1517_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_3
  simp only [← c93021, ← c93022, k2, ← c93023] at e_1517_3
  have hr := e_1517_3
  simp only [e_1517_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14657)) := by
  have c93024 := (publicBatchWrapper89_copies2907 a h).1
  have c93025 := (publicBatchWrapper89_copies2907 a h).2.1
  have c93026 := (publicBatchWrapper89_copies2907 a h).2.2.1
  have c93027 := (publicBatchWrapper89_copies2907 a h).2.2.2.1
  have c93028 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.1
  have c93029 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_4
  simp only [← c93024, ← c93025, ← c93026] at e_1517_4
  have e_1517_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_5
  simp only [← c93027, ← c93028, k2, ← c93029] at e_1517_5
  have hr := e_1517_5
  simp only [e_1517_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14658)) := by
  have c93030 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.1
  have c93031 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.1
  have c93032 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.1
  have c93033 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.1
  have c93034 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93035 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_6
  simp only [← c93030, ← c93031, ← c93032] at e_1517_6
  have e_1517_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_7
  simp only [← c93033, ← c93034, k2, ← c93035] at e_1517_7
  have hr := e_1517_7
  simp only [e_1517_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14659)) := by
  have c93036 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93037 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93038 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93039 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93040 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93041 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_8
  simp only [← c93036, ← c93037, ← c93038] at e_1517_8
  have e_1517_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_9
  simp only [← c93039, ← c93040, k2, ← c93041] at e_1517_9
  have hr := e_1517_9
  simp only [e_1517_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14660)) := by
  have c93042 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93043 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93044 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93045 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93046 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93047 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_10
  simp only [← c93042, ← c93043, ← c93044] at e_1517_10
  have e_1517_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_11
  simp only [← c93045, ← c93046, k2, ← c93047] at e_1517_11
  have hr := e_1517_11
  simp only [e_1517_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14661)) := by
  have c93048 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93049 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93050 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93051 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93052 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93053 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_12
  simp only [← c93048, ← c93049, ← c93050] at e_1517_12
  have e_1517_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_13
  simp only [← c93051, ← c93052, k2, ← c93053] at e_1517_13
  have hr := e_1517_13
  simp only [e_1517_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14662)) := by
  have c93054 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93055 := (publicBatchWrapper89_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93056 := (publicBatchWrapper89_copies2908 a h).1
  have c93057 := (publicBatchWrapper89_copies2908 a h).2.1
  have c93058 := (publicBatchWrapper89_copies2908 a h).2.2.1
  have c93059 := (publicBatchWrapper89_copies2908 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_14
  simp only [← c93054, ← c93055, ← c93056] at e_1517_14
  have e_1517_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_15
  simp only [← c93057, ← c93058, k2, ← c93059] at e_1517_15
  have hr := e_1517_15
  simp only [e_1517_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14663)) := by
  have c93060 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.1
  have c93061 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.1
  have c93062 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.1
  have c93063 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.1
  have c93064 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.1
  have c93065 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_16
  simp only [← c93060, ← c93061, ← c93062] at e_1517_16
  have e_1517_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_17
  simp only [← c93063, ← c93064, k2, ← c93065] at e_1517_17
  have hr := e_1517_17
  simp only [e_1517_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1517 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14664)) := by
  have c93066 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93067 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93068 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93069 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93070 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93071 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1517_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_18
  simp only [← c93066, ← c93067, ← c93068] at e_1517_18
  have e_1517_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_19
  simp only [← c93069, ← c93070, k2, ← c93071] at e_1517_19
  have hr := e_1517_19
  simp only [e_1517_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14665)) := by
  have c93072 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93073 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93074 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93075 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93076 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93077 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_0
  simp only [← c93072, ← c93073, ← c93074] at e_1518_0
  have e_1518_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_1
  simp only [← c93075, ← c93076, k2, ← c93077] at e_1518_1
  have hr := e_1518_1
  simp only [e_1518_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14666)) := by
  have c93078 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93079 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93080 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93081 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93082 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93083 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_2
  simp only [← c93078, ← c93079, ← c93080] at e_1518_2
  have e_1518_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_3
  simp only [← c93081, ← c93082, k2, ← c93083] at e_1518_3
  have hr := e_1518_3
  simp only [e_1518_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14667)) := by
  have c93084 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93085 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93086 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93087 := (publicBatchWrapper89_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93088 := (publicBatchWrapper89_copies2909 a h).1
  have c93089 := (publicBatchWrapper89_copies2909 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_4
  simp only [← c93084, ← c93085, ← c93086] at e_1518_4
  have e_1518_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_5
  simp only [← c93087, ← c93088, k2, ← c93089] at e_1518_5
  have hr := e_1518_5
  simp only [e_1518_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14668)) := by
  have c93090 := (publicBatchWrapper89_copies2909 a h).2.2.1
  have c93091 := (publicBatchWrapper89_copies2909 a h).2.2.2.1
  have c93092 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.1
  have c93093 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.1
  have c93094 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.1
  have c93095 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_6
  simp only [← c93090, ← c93091, ← c93092] at e_1518_6
  have e_1518_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_7
  simp only [← c93093, ← c93094, k2, ← c93095] at e_1518_7
  have hr := e_1518_7
  simp only [e_1518_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14669)) := by
  have c93096 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.1
  have c93097 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.1
  have c93098 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93099 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93100 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93101 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_8
  simp only [← c93096, ← c93097, ← c93098] at e_1518_8
  have e_1518_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_9
  simp only [← c93099, ← c93100, k2, ← c93101] at e_1518_9
  have hr := e_1518_9
  simp only [e_1518_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14670)) := by
  have c93102 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93103 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93104 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93105 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93106 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93107 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_10
  simp only [← c93102, ← c93103, ← c93104] at e_1518_10
  have e_1518_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_11
  simp only [← c93105, ← c93106, k2, ← c93107] at e_1518_11
  have hr := e_1518_11
  simp only [e_1518_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14671)) := by
  have c93108 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93109 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93110 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93111 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93112 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93113 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_12
  simp only [← c93108, ← c93109, ← c93110] at e_1518_12
  have e_1518_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_13
  simp only [← c93111, ← c93112, k2, ← c93113] at e_1518_13
  have hr := e_1518_13
  simp only [e_1518_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14672)) := by
  have c93114 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93115 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93116 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93117 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93118 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93119 := (publicBatchWrapper89_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_14
  simp only [← c93114, ← c93115, ← c93116] at e_1518_14
  have e_1518_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_15
  simp only [← c93117, ← c93118, k2, ← c93119] at e_1518_15
  have hr := e_1518_15
  simp only [e_1518_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14673)) := by
  have c93120 := (publicBatchWrapper89_copies2910 a h).1
  have c93121 := (publicBatchWrapper89_copies2910 a h).2.1
  have c93122 := (publicBatchWrapper89_copies2910 a h).2.2.1
  have c93123 := (publicBatchWrapper89_copies2910 a h).2.2.2.1
  have c93124 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.1
  have c93125 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_16
  simp only [← c93120, ← c93121, ← c93122] at e_1518_16
  have e_1518_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_17
  simp only [← c93123, ← c93124, k2, ← c93125] at e_1518_17
  have hr := e_1518_17
  simp only [e_1518_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1518 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14674)) := by
  have c93126 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.1
  have c93127 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.1
  have c93128 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.1
  have c93129 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.1
  have c93130 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93131 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1518_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_18
  simp only [← c93126, ← c93127, ← c93128] at e_1518_18
  have e_1518_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_19
  simp only [← c93129, ← c93130, k2, ← c93131] at e_1518_19
  have hr := e_1518_19
  simp only [e_1518_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14675)) := by
  have c93132 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93133 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93134 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93135 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93136 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93137 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_0
  simp only [← c93132, ← c93133, ← c93134] at e_1519_0
  have e_1519_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_1
  simp only [← c93135, ← c93136, k2, ← c93137] at e_1519_1
  have hr := e_1519_1
  simp only [e_1519_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14676)) := by
  have c93138 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93139 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93140 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93141 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93142 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93143 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_2
  simp only [← c93138, ← c93139, ← c93140] at e_1519_2
  have e_1519_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_3
  simp only [← c93141, ← c93142, k2, ← c93143] at e_1519_3
  have hr := e_1519_3
  simp only [e_1519_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14677)) := by
  have c93144 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93145 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93146 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93147 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93148 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93149 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_4
  simp only [← c93144, ← c93145, ← c93146] at e_1519_4
  have e_1519_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_5
  simp only [← c93147, ← c93148, k2, ← c93149] at e_1519_5
  have hr := e_1519_5
  simp only [e_1519_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14678)) := by
  have c93150 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93151 := (publicBatchWrapper89_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93152 := (publicBatchWrapper89_copies2911 a h).1
  have c93153 := (publicBatchWrapper89_copies2911 a h).2.1
  have c93154 := (publicBatchWrapper89_copies2911 a h).2.2.1
  have c93155 := (publicBatchWrapper89_copies2911 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_6
  simp only [← c93150, ← c93151, ← c93152] at e_1519_6
  have e_1519_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_7
  simp only [← c93153, ← c93154, k2, ← c93155] at e_1519_7
  have hr := e_1519_7
  simp only [e_1519_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14679)) := by
  have c93156 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.1
  have c93157 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.1
  have c93158 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.1
  have c93159 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.1
  have c93160 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.1
  have c93161 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_8
  simp only [← c93156, ← c93157, ← c93158] at e_1519_8
  have e_1519_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_9
  simp only [← c93159, ← c93160, k2, ← c93161] at e_1519_9
  have hr := e_1519_9
  simp only [e_1519_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14680)) := by
  have c93162 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93163 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93164 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93165 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93166 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93167 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_10
  simp only [← c93162, ← c93163, ← c93164] at e_1519_10
  have e_1519_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_11
  simp only [← c93165, ← c93166, k2, ← c93167] at e_1519_11
  have hr := e_1519_11
  simp only [e_1519_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14681)) := by
  have c93168 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93169 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93170 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93171 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93172 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93173 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_12
  simp only [← c93168, ← c93169, ← c93170] at e_1519_12
  have e_1519_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_13
  simp only [← c93171, ← c93172, k2, ← c93173] at e_1519_13
  have hr := e_1519_13
  simp only [e_1519_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14682)) := by
  have c93174 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93175 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93176 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93177 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93178 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93179 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_14
  simp only [← c93174, ← c93175, ← c93176] at e_1519_14
  have e_1519_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_15
  simp only [← c93177, ← c93178, k2, ← c93179] at e_1519_15
  have hr := e_1519_15
  simp only [e_1519_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14683)) := by
  have c93180 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93181 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93182 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93183 := (publicBatchWrapper89_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93184 := (publicBatchWrapper89_copies2912 a h).1
  have c93185 := (publicBatchWrapper89_copies2912 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_16
  simp only [← c93180, ← c93181, ← c93182] at e_1519_16
  have e_1519_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_17
  simp only [← c93183, ← c93184, k2, ← c93185] at e_1519_17
  have hr := e_1519_17
  simp only [e_1519_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1519 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14684)) := by
  have c93186 := (publicBatchWrapper89_copies2912 a h).2.2.1
  have c93187 := (publicBatchWrapper89_copies2912 a h).2.2.2.1
  have c93188 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.1
  have c93189 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.1
  have c93190 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.1
  have c93191 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1519_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_18
  simp only [← c93186, ← c93187, ← c93188] at e_1519_18
  have e_1519_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_19
  simp only [← c93189, ← c93190, k2, ← c93191] at e_1519_19
  have hr := e_1519_19
  simp only [e_1519_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14685)) := by
  have c93192 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.1
  have c93193 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.1
  have c93194 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93195 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93196 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93197 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_0
  simp only [← c93192, ← c93193, ← c93194] at e_1520_0
  have e_1520_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_1
  simp only [← c93195, ← c93196, k2, ← c93197] at e_1520_1
  have hr := e_1520_1
  simp only [e_1520_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14686)) := by
  have c93198 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93199 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93200 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93201 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93202 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93203 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_2
  simp only [← c93198, ← c93199, ← c93200] at e_1520_2
  have e_1520_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_3
  simp only [← c93201, ← c93202, k2, ← c93203] at e_1520_3
  have hr := e_1520_3
  simp only [e_1520_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14687)) := by
  have c93204 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93205 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93206 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93207 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93208 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93209 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_4
  simp only [← c93204, ← c93205, ← c93206] at e_1520_4
  have e_1520_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_5
  simp only [← c93207, ← c93208, k2, ← c93209] at e_1520_5
  have hr := e_1520_5
  simp only [e_1520_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14688)) := by
  have c93210 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93211 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93212 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93213 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93214 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93215 := (publicBatchWrapper89_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_6
  simp only [← c93210, ← c93211, ← c93212] at e_1520_6
  have e_1520_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_7
  simp only [← c93213, ← c93214, k2, ← c93215] at e_1520_7
  have hr := e_1520_7
  simp only [e_1520_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14689)) := by
  have c93216 := (publicBatchWrapper89_copies2913 a h).1
  have c93217 := (publicBatchWrapper89_copies2913 a h).2.1
  have c93218 := (publicBatchWrapper89_copies2913 a h).2.2.1
  have c93219 := (publicBatchWrapper89_copies2913 a h).2.2.2.1
  have c93220 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.1
  have c93221 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_8
  simp only [← c93216, ← c93217, ← c93218] at e_1520_8
  have e_1520_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_9
  simp only [← c93219, ← c93220, k2, ← c93221] at e_1520_9
  have hr := e_1520_9
  simp only [e_1520_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14690)) := by
  have c93222 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.1
  have c93223 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.1
  have c93224 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.1
  have c93225 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.1
  have c93226 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93227 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_10
  simp only [← c93222, ← c93223, ← c93224] at e_1520_10
  have e_1520_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_11
  simp only [← c93225, ← c93226, k2, ← c93227] at e_1520_11
  have hr := e_1520_11
  simp only [e_1520_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14691)) := by
  have c93228 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93229 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93230 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93231 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93232 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93233 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_12
  simp only [← c93228, ← c93229, ← c93230] at e_1520_12
  have e_1520_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_13
  simp only [← c93231, ← c93232, k2, ← c93233] at e_1520_13
  have hr := e_1520_13
  simp only [e_1520_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14692)) := by
  have c93234 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93235 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93236 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93237 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93238 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93239 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_14
  simp only [← c93234, ← c93235, ← c93236] at e_1520_14
  have e_1520_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_15
  simp only [← c93237, ← c93238, k2, ← c93239] at e_1520_15
  have hr := e_1520_15
  simp only [e_1520_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14693)) := by
  have c93240 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93241 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93242 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93243 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93244 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93245 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_16
  simp only [← c93240, ← c93241, ← c93242] at e_1520_16
  have e_1520_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_17
  simp only [← c93243, ← c93244, k2, ← c93245] at e_1520_17
  have hr := e_1520_17
  simp only [e_1520_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1520 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14694)) := by
  have c93246 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93247 := (publicBatchWrapper89_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93248 := (publicBatchWrapper89_copies2914 a h).1
  have c93249 := (publicBatchWrapper89_copies2914 a h).2.1
  have c93250 := (publicBatchWrapper89_copies2914 a h).2.2.1
  have c93251 := (publicBatchWrapper89_copies2914 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1520_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_18
  simp only [← c93246, ← c93247, ← c93248] at e_1520_18
  have e_1520_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_19
  simp only [← c93249, ← c93250, k2, ← c93251] at e_1520_19
  have hr := e_1520_19
  simp only [e_1520_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14695)) := by
  have c93252 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.1
  have c93253 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.1
  have c93254 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.1
  have c93255 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.1
  have c93256 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.1
  have c93257 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_0
  simp only [← c93252, ← c93253, ← c93254] at e_1521_0
  have e_1521_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_1
  simp only [← c93255, ← c93256, k2, ← c93257] at e_1521_1
  have hr := e_1521_1
  simp only [e_1521_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14696)) := by
  have c93258 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93259 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93260 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93261 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93262 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93263 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_2
  simp only [← c93258, ← c93259, ← c93260] at e_1521_2
  have e_1521_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_3
  simp only [← c93261, ← c93262, k2, ← c93263] at e_1521_3
  have hr := e_1521_3
  simp only [e_1521_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14697)) := by
  have c93264 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93265 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93266 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93267 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93268 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93269 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_4
  simp only [← c93264, ← c93265, ← c93266] at e_1521_4
  have e_1521_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_5
  simp only [← c93267, ← c93268, k2, ← c93269] at e_1521_5
  have hr := e_1521_5
  simp only [e_1521_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14698)) := by
  have c93270 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93271 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93272 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93273 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93274 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93275 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_6
  simp only [← c93270, ← c93271, ← c93272] at e_1521_6
  have e_1521_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_7
  simp only [← c93273, ← c93274, k2, ← c93275] at e_1521_7
  have hr := e_1521_7
  simp only [e_1521_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14699)) := by
  have c93276 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93277 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93278 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93279 := (publicBatchWrapper89_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93280 := (publicBatchWrapper89_copies2915 a h).1
  have c93281 := (publicBatchWrapper89_copies2915 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_8
  simp only [← c93276, ← c93277, ← c93278] at e_1521_8
  have e_1521_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_9
  simp only [← c93279, ← c93280, k2, ← c93281] at e_1521_9
  have hr := e_1521_9
  simp only [e_1521_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14700)) := by
  have c93282 := (publicBatchWrapper89_copies2915 a h).2.2.1
  have c93283 := (publicBatchWrapper89_copies2915 a h).2.2.2.1
  have c93284 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.1
  have c93285 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.1
  have c93286 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.1
  have c93287 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_10
  simp only [← c93282, ← c93283, ← c93284] at e_1521_10
  have e_1521_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_11
  simp only [← c93285, ← c93286, k2, ← c93287] at e_1521_11
  have hr := e_1521_11
  simp only [e_1521_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14701)) := by
  have c93288 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.1
  have c93289 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.1
  have c93290 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93291 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93292 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93293 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_12
  simp only [← c93288, ← c93289, ← c93290] at e_1521_12
  have e_1521_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_13
  simp only [← c93291, ← c93292, k2, ← c93293] at e_1521_13
  have hr := e_1521_13
  simp only [e_1521_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14702)) := by
  have c93294 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93295 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93296 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93297 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93298 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93299 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_14
  simp only [← c93294, ← c93295, ← c93296] at e_1521_14
  have e_1521_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_15
  simp only [← c93297, ← c93298, k2, ← c93299] at e_1521_15
  have hr := e_1521_15
  simp only [e_1521_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14703)) := by
  have c93300 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93301 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93302 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93303 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93304 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93305 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_16
  simp only [← c93300, ← c93301, ← c93302] at e_1521_16
  have e_1521_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_17
  simp only [← c93303, ← c93304, k2, ← c93305] at e_1521_17
  have hr := e_1521_17
  simp only [e_1521_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1521 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14704)) := by
  have c93306 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93307 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93308 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93309 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93310 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93311 := (publicBatchWrapper89_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1521_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_18
  simp only [← c93306, ← c93307, ← c93308] at e_1521_18
  have e_1521_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_19
  simp only [← c93309, ← c93310, k2, ← c93311] at e_1521_19
  have hr := e_1521_19
  simp only [e_1521_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14705)) := by
  have c93312 := (publicBatchWrapper89_copies2916 a h).1
  have c93313 := (publicBatchWrapper89_copies2916 a h).2.1
  have c93314 := (publicBatchWrapper89_copies2916 a h).2.2.1
  have c93315 := (publicBatchWrapper89_copies2916 a h).2.2.2.1
  have c93316 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.1
  have c93317 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_0
  simp only [← c93312, ← c93313, ← c93314] at e_1522_0
  have e_1522_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_1
  simp only [← c93315, ← c93316, k2, ← c93317] at e_1522_1
  have hr := e_1522_1
  simp only [e_1522_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14706)) := by
  have c93318 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.1
  have c93319 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.1
  have c93320 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.1
  have c93321 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.1
  have c93322 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93323 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_2
  simp only [← c93318, ← c93319, ← c93320] at e_1522_2
  have e_1522_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_3
  simp only [← c93321, ← c93322, k2, ← c93323] at e_1522_3
  have hr := e_1522_3
  simp only [e_1522_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14707)) := by
  have c93324 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93325 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93326 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93327 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93328 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93329 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_4
  simp only [← c93324, ← c93325, ← c93326] at e_1522_4
  have e_1522_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_5
  simp only [← c93327, ← c93328, k2, ← c93329] at e_1522_5
  have hr := e_1522_5
  simp only [e_1522_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14708)) := by
  have c93330 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93331 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93332 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93333 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93334 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93335 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_6
  simp only [← c93330, ← c93331, ← c93332] at e_1522_6
  have e_1522_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_7
  simp only [← c93333, ← c93334, k2, ← c93335] at e_1522_7
  have hr := e_1522_7
  simp only [e_1522_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14709)) := by
  have c93336 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93337 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93338 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93339 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93340 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93341 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_8
  simp only [← c93336, ← c93337, ← c93338] at e_1522_8
  have e_1522_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_9
  simp only [← c93339, ← c93340, k2, ← c93341] at e_1522_9
  have hr := e_1522_9
  simp only [e_1522_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14710)) := by
  have c93342 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93343 := (publicBatchWrapper89_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93344 := (publicBatchWrapper89_copies2917 a h).1
  have c93345 := (publicBatchWrapper89_copies2917 a h).2.1
  have c93346 := (publicBatchWrapper89_copies2917 a h).2.2.1
  have c93347 := (publicBatchWrapper89_copies2917 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_10
  simp only [← c93342, ← c93343, ← c93344] at e_1522_10
  have e_1522_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_11
  simp only [← c93345, ← c93346, k2, ← c93347] at e_1522_11
  have hr := e_1522_11
  simp only [e_1522_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14711)) := by
  have c93348 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.1
  have c93349 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.1
  have c93350 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.1
  have c93351 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.1
  have c93352 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.1
  have c93353 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_12
  simp only [← c93348, ← c93349, ← c93350] at e_1522_12
  have e_1522_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_13
  simp only [← c93351, ← c93352, k2, ← c93353] at e_1522_13
  have hr := e_1522_13
  simp only [e_1522_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14712)) := by
  have c93354 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93355 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93356 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93357 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93358 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93359 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_14
  simp only [← c93354, ← c93355, ← c93356] at e_1522_14
  have e_1522_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_15
  simp only [← c93357, ← c93358, k2, ← c93359] at e_1522_15
  have hr := e_1522_15
  simp only [e_1522_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14713)) := by
  have c93360 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93361 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93362 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93363 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93364 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93365 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_16
  simp only [← c93360, ← c93361, ← c93362] at e_1522_16
  have e_1522_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_17
  simp only [← c93363, ← c93364, k2, ← c93365] at e_1522_17
  have hr := e_1522_17
  simp only [e_1522_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1522 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14714)) := by
  have c93366 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93367 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93368 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93369 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93370 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93371 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1522_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_18
  simp only [← c93366, ← c93367, ← c93368] at e_1522_18
  have e_1522_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_19
  simp only [← c93369, ← c93370, k2, ← c93371] at e_1522_19
  have hr := e_1522_19
  simp only [e_1522_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14715)) := by
  have c93372 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93373 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93374 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93375 := (publicBatchWrapper89_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93376 := (publicBatchWrapper89_copies2918 a h).1
  have c93377 := (publicBatchWrapper89_copies2918 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_0
  simp only [← c93372, ← c93373, ← c93374] at e_1523_0
  have e_1523_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_1
  simp only [← c93375, ← c93376, k2, ← c93377] at e_1523_1
  have hr := e_1523_1
  simp only [e_1523_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14716)) := by
  have c93378 := (publicBatchWrapper89_copies2918 a h).2.2.1
  have c93379 := (publicBatchWrapper89_copies2918 a h).2.2.2.1
  have c93380 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.1
  have c93381 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.1
  have c93382 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.1
  have c93383 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_2
  simp only [← c93378, ← c93379, ← c93380] at e_1523_2
  have e_1523_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_3
  simp only [← c93381, ← c93382, k2, ← c93383] at e_1523_3
  have hr := e_1523_3
  simp only [e_1523_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14717)) := by
  have c93384 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.1
  have c93385 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.1
  have c93386 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93387 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93388 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93389 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_4
  simp only [← c93384, ← c93385, ← c93386] at e_1523_4
  have e_1523_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_5
  simp only [← c93387, ← c93388, k2, ← c93389] at e_1523_5
  have hr := e_1523_5
  simp only [e_1523_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14718)) := by
  have c93390 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93391 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93392 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93393 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93394 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93395 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_6
  simp only [← c93390, ← c93391, ← c93392] at e_1523_6
  have e_1523_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_7
  simp only [← c93393, ← c93394, k2, ← c93395] at e_1523_7
  have hr := e_1523_7
  simp only [e_1523_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14719)) := by
  have c93396 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93397 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93398 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93399 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93400 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93401 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_8
  simp only [← c93396, ← c93397, ← c93398] at e_1523_8
  have e_1523_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_9
  simp only [← c93399, ← c93400, k2, ← c93401] at e_1523_9
  have hr := e_1523_9
  simp only [e_1523_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14720)) := by
  have c93402 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93403 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93404 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93405 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93406 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93407 := (publicBatchWrapper89_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_10
  simp only [← c93402, ← c93403, ← c93404] at e_1523_10
  have e_1523_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_11
  simp only [← c93405, ← c93406, k2, ← c93407] at e_1523_11
  have hr := e_1523_11
  simp only [e_1523_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14721)) := by
  have c93408 := (publicBatchWrapper89_copies2919 a h).1
  have c93409 := (publicBatchWrapper89_copies2919 a h).2.1
  have c93410 := (publicBatchWrapper89_copies2919 a h).2.2.1
  have c93411 := (publicBatchWrapper89_copies2919 a h).2.2.2.1
  have c93412 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.1
  have c93413 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_12
  simp only [← c93408, ← c93409, ← c93410] at e_1523_12
  have e_1523_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_13
  simp only [← c93411, ← c93412, k2, ← c93413] at e_1523_13
  have hr := e_1523_13
  simp only [e_1523_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14722)) := by
  have c93414 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.1
  have c93415 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.1
  have c93416 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.1
  have c93417 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.1
  have c93418 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93419 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_14
  simp only [← c93414, ← c93415, ← c93416] at e_1523_14
  have e_1523_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_15
  simp only [← c93417, ← c93418, k2, ← c93419] at e_1523_15
  have hr := e_1523_15
  simp only [e_1523_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14723)) := by
  have c93420 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93421 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93422 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93423 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93424 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93425 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_16
  simp only [← c93420, ← c93421, ← c93422] at e_1523_16
  have e_1523_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_17
  simp only [← c93423, ← c93424, k2, ← c93425] at e_1523_17
  have hr := e_1523_17
  simp only [e_1523_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1523 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14724)) := by
  have c93426 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93427 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93428 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93429 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93430 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93431 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1523_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_18
  simp only [← c93426, ← c93427, ← c93428] at e_1523_18
  have e_1523_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_19
  simp only [← c93429, ← c93430, k2, ← c93431] at e_1523_19
  have hr := e_1523_19
  simp only [e_1523_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14725)) := by
  have c93432 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93433 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93434 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93435 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93436 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93437 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_0
  simp only [← c93432, ← c93433, ← c93434] at e_1524_0
  have e_1524_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_1
  simp only [← c93435, ← c93436, k2, ← c93437] at e_1524_1
  have hr := e_1524_1
  simp only [e_1524_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14726)) := by
  have c93438 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93439 := (publicBatchWrapper89_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93440 := (publicBatchWrapper89_copies2920 a h).1
  have c93441 := (publicBatchWrapper89_copies2920 a h).2.1
  have c93442 := (publicBatchWrapper89_copies2920 a h).2.2.1
  have c93443 := (publicBatchWrapper89_copies2920 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_2
  simp only [← c93438, ← c93439, ← c93440] at e_1524_2
  have e_1524_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_3
  simp only [← c93441, ← c93442, k2, ← c93443] at e_1524_3
  have hr := e_1524_3
  simp only [e_1524_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14727)) := by
  have c93444 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.1
  have c93445 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.1
  have c93446 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.1
  have c93447 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.1
  have c93448 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.1
  have c93449 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_4
  simp only [← c93444, ← c93445, ← c93446] at e_1524_4
  have e_1524_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_5
  simp only [← c93447, ← c93448, k2, ← c93449] at e_1524_5
  have hr := e_1524_5
  simp only [e_1524_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14728)) := by
  have c93450 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93451 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93452 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93453 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93454 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93455 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_6
  simp only [← c93450, ← c93451, ← c93452] at e_1524_6
  have e_1524_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_7
  simp only [← c93453, ← c93454, k2, ← c93455] at e_1524_7
  have hr := e_1524_7
  simp only [e_1524_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14729)) := by
  have c93456 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93457 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93458 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93459 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93460 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93461 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_8
  simp only [← c93456, ← c93457, ← c93458] at e_1524_8
  have e_1524_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_9
  simp only [← c93459, ← c93460, k2, ← c93461] at e_1524_9
  have hr := e_1524_9
  simp only [e_1524_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14730)) := by
  have c93462 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93463 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93464 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93465 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93466 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93467 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_10
  simp only [← c93462, ← c93463, ← c93464] at e_1524_10
  have e_1524_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_11
  simp only [← c93465, ← c93466, k2, ← c93467] at e_1524_11
  have hr := e_1524_11
  simp only [e_1524_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14731)) := by
  have c93468 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93469 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93470 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93471 := (publicBatchWrapper89_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93472 := (publicBatchWrapper89_copies2921 a h).1
  have c93473 := (publicBatchWrapper89_copies2921 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_12
  simp only [← c93468, ← c93469, ← c93470] at e_1524_12
  have e_1524_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_13
  simp only [← c93471, ← c93472, k2, ← c93473] at e_1524_13
  have hr := e_1524_13
  simp only [e_1524_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14732)) := by
  have c93474 := (publicBatchWrapper89_copies2921 a h).2.2.1
  have c93475 := (publicBatchWrapper89_copies2921 a h).2.2.2.1
  have c93476 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.1
  have c93477 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.1
  have c93478 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.1
  have c93479 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_14
  simp only [← c93474, ← c93475, ← c93476] at e_1524_14
  have e_1524_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_15
  simp only [← c93477, ← c93478, k2, ← c93479] at e_1524_15
  have hr := e_1524_15
  simp only [e_1524_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14733)) := by
  have c93480 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.1
  have c93481 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.1
  have c93482 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93483 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93484 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93485 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_16
  simp only [← c93480, ← c93481, ← c93482] at e_1524_16
  have e_1524_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_17
  simp only [← c93483, ← c93484, k2, ← c93485] at e_1524_17
  have hr := e_1524_17
  simp only [e_1524_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1524 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14734)) := by
  have c93486 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93487 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93488 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93489 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93490 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93491 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1524_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_18
  simp only [← c93486, ← c93487, ← c93488] at e_1524_18
  have e_1524_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_19
  simp only [← c93489, ← c93490, k2, ← c93491] at e_1524_19
  have hr := e_1524_19
  simp only [e_1524_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14735)) := by
  have c93492 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93493 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93494 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93495 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93496 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93497 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_0
  simp only [← c93492, ← c93493, ← c93494] at e_1525_0
  have e_1525_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_1
  simp only [← c93495, ← c93496, k2, ← c93497] at e_1525_1
  have hr := e_1525_1
  simp only [e_1525_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14736)) := by
  have c93498 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93499 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93500 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93501 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93502 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93503 := (publicBatchWrapper89_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_2
  simp only [← c93498, ← c93499, ← c93500] at e_1525_2
  have e_1525_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_3
  simp only [← c93501, ← c93502, k2, ← c93503] at e_1525_3
  have hr := e_1525_3
  simp only [e_1525_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14737)) := by
  have c93504 := (publicBatchWrapper89_copies2922 a h).1
  have c93505 := (publicBatchWrapper89_copies2922 a h).2.1
  have c93506 := (publicBatchWrapper89_copies2922 a h).2.2.1
  have c93507 := (publicBatchWrapper89_copies2922 a h).2.2.2.1
  have c93508 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.1
  have c93509 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_4
  simp only [← c93504, ← c93505, ← c93506] at e_1525_4
  have e_1525_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_5
  simp only [← c93507, ← c93508, k2, ← c93509] at e_1525_5
  have hr := e_1525_5
  simp only [e_1525_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14738)) := by
  have c93510 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.1
  have c93511 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.1
  have c93512 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.1
  have c93513 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.1
  have c93514 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93515 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_6
  simp only [← c93510, ← c93511, ← c93512] at e_1525_6
  have e_1525_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_7
  simp only [← c93513, ← c93514, k2, ← c93515] at e_1525_7
  have hr := e_1525_7
  simp only [e_1525_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14739)) := by
  have c93516 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93517 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93518 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93519 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93520 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93521 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_8
  simp only [← c93516, ← c93517, ← c93518] at e_1525_8
  have e_1525_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_9
  simp only [← c93519, ← c93520, k2, ← c93521] at e_1525_9
  have hr := e_1525_9
  simp only [e_1525_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14740)) := by
  have c93522 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93523 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93524 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93525 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93526 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93527 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_10
  simp only [← c93522, ← c93523, ← c93524] at e_1525_10
  have e_1525_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_11
  simp only [← c93525, ← c93526, k2, ← c93527] at e_1525_11
  have hr := e_1525_11
  simp only [e_1525_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14741)) := by
  have c93528 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93529 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93530 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93531 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93532 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93533 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_12
  simp only [← c93528, ← c93529, ← c93530] at e_1525_12
  have e_1525_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_13
  simp only [← c93531, ← c93532, k2, ← c93533] at e_1525_13
  have hr := e_1525_13
  simp only [e_1525_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14742)) := by
  have c93534 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93535 := (publicBatchWrapper89_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93536 := (publicBatchWrapper89_copies2923 a h).1
  have c93537 := (publicBatchWrapper89_copies2923 a h).2.1
  have c93538 := (publicBatchWrapper89_copies2923 a h).2.2.1
  have c93539 := (publicBatchWrapper89_copies2923 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_14
  simp only [← c93534, ← c93535, ← c93536] at e_1525_14
  have e_1525_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_15
  simp only [← c93537, ← c93538, k2, ← c93539] at e_1525_15
  have hr := e_1525_15
  simp only [e_1525_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14743)) := by
  have c93540 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.1
  have c93541 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.1
  have c93542 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.1
  have c93543 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.1
  have c93544 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.1
  have c93545 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_16
  simp only [← c93540, ← c93541, ← c93542] at e_1525_16
  have e_1525_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_17
  simp only [← c93543, ← c93544, k2, ← c93545] at e_1525_17
  have hr := e_1525_17
  simp only [e_1525_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1525 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14744)) := by
  have c93546 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93547 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93548 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93549 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93550 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93551 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1525_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_18
  simp only [← c93546, ← c93547, ← c93548] at e_1525_18
  have e_1525_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_19
  simp only [← c93549, ← c93550, k2, ← c93551] at e_1525_19
  have hr := e_1525_19
  simp only [e_1525_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14745)) := by
  have c93552 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93553 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93554 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93555 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93556 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93557 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_0
  simp only [← c93552, ← c93553, ← c93554] at e_1526_0
  have e_1526_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_1
  simp only [← c93555, ← c93556, k2, ← c93557] at e_1526_1
  have hr := e_1526_1
  simp only [e_1526_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14746)) := by
  have c93558 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93559 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93560 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93561 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93562 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93563 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_2
  simp only [← c93558, ← c93559, ← c93560] at e_1526_2
  have e_1526_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_3
  simp only [← c93561, ← c93562, k2, ← c93563] at e_1526_3
  have hr := e_1526_3
  simp only [e_1526_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14747)) := by
  have c93564 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93565 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93566 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93567 := (publicBatchWrapper89_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93568 := (publicBatchWrapper89_copies2924 a h).1
  have c93569 := (publicBatchWrapper89_copies2924 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_4
  simp only [← c93564, ← c93565, ← c93566] at e_1526_4
  have e_1526_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_5
  simp only [← c93567, ← c93568, k2, ← c93569] at e_1526_5
  have hr := e_1526_5
  simp only [e_1526_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14748)) := by
  have c93570 := (publicBatchWrapper89_copies2924 a h).2.2.1
  have c93571 := (publicBatchWrapper89_copies2924 a h).2.2.2.1
  have c93572 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.1
  have c93573 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.1
  have c93574 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.1
  have c93575 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_6
  simp only [← c93570, ← c93571, ← c93572] at e_1526_6
  have e_1526_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_7
  simp only [← c93573, ← c93574, k2, ← c93575] at e_1526_7
  have hr := e_1526_7
  simp only [e_1526_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14749)) := by
  have c93576 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.1
  have c93577 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.1
  have c93578 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93579 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93580 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93581 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_8
  simp only [← c93576, ← c93577, ← c93578] at e_1526_8
  have e_1526_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_9
  simp only [← c93579, ← c93580, k2, ← c93581] at e_1526_9
  have hr := e_1526_9
  simp only [e_1526_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14750)) := by
  have c93582 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93583 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93584 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93585 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93586 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93587 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_10
  simp only [← c93582, ← c93583, ← c93584] at e_1526_10
  have e_1526_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_11
  simp only [← c93585, ← c93586, k2, ← c93587] at e_1526_11
  have hr := e_1526_11
  simp only [e_1526_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14751)) := by
  have c93588 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93589 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93590 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93591 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93592 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93593 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_12
  simp only [← c93588, ← c93589, ← c93590] at e_1526_12
  have e_1526_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_13
  simp only [← c93591, ← c93592, k2, ← c93593] at e_1526_13
  have hr := e_1526_13
  simp only [e_1526_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14752)) := by
  have c93594 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93595 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93596 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93597 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93598 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93599 := (publicBatchWrapper89_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_14
  simp only [← c93594, ← c93595, ← c93596] at e_1526_14
  have e_1526_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_15
  simp only [← c93597, ← c93598, k2, ← c93599] at e_1526_15
  have hr := e_1526_15
  simp only [e_1526_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14753)) := by
  have c93600 := (publicBatchWrapper89_copies2925 a h).1
  have c93601 := (publicBatchWrapper89_copies2925 a h).2.1
  have c93602 := (publicBatchWrapper89_copies2925 a h).2.2.1
  have c93603 := (publicBatchWrapper89_copies2925 a h).2.2.2.1
  have c93604 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.1
  have c93605 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_16
  simp only [← c93600, ← c93601, ← c93602] at e_1526_16
  have e_1526_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_17
  simp only [← c93603, ← c93604, k2, ← c93605] at e_1526_17
  have hr := e_1526_17
  simp only [e_1526_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1526 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14754)) := by
  have c93606 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.1
  have c93607 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.1
  have c93608 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.1
  have c93609 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.1
  have c93610 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93611 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1526_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_18
  simp only [← c93606, ← c93607, ← c93608] at e_1526_18
  have e_1526_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_19
  simp only [← c93609, ← c93610, k2, ← c93611] at e_1526_19
  have hr := e_1526_19
  simp only [e_1526_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14755)) := by
  have c93612 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93613 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93614 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93615 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93616 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93617 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_0
  simp only [← c93612, ← c93613, ← c93614] at e_1527_0
  have e_1527_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_1
  simp only [← c93615, ← c93616, k2, ← c93617] at e_1527_1
  have hr := e_1527_1
  simp only [e_1527_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14756)) := by
  have c93618 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93619 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93620 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93621 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93622 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93623 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_2
  simp only [← c93618, ← c93619, ← c93620] at e_1527_2
  have e_1527_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_3
  simp only [← c93621, ← c93622, k2, ← c93623] at e_1527_3
  have hr := e_1527_3
  simp only [e_1527_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14757)) := by
  have c93624 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93625 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93626 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93627 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93628 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93629 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_4
  simp only [← c93624, ← c93625, ← c93626] at e_1527_4
  have e_1527_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_5
  simp only [← c93627, ← c93628, k2, ← c93629] at e_1527_5
  have hr := e_1527_5
  simp only [e_1527_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14758)) := by
  have c93630 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93631 := (publicBatchWrapper89_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93632 := (publicBatchWrapper89_copies2926 a h).1
  have c93633 := (publicBatchWrapper89_copies2926 a h).2.1
  have c93634 := (publicBatchWrapper89_copies2926 a h).2.2.1
  have c93635 := (publicBatchWrapper89_copies2926 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_6
  simp only [← c93630, ← c93631, ← c93632] at e_1527_6
  have e_1527_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_7
  simp only [← c93633, ← c93634, k2, ← c93635] at e_1527_7
  have hr := e_1527_7
  simp only [e_1527_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14759)) := by
  have c93636 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.1
  have c93637 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.1
  have c93638 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.1
  have c93639 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.1
  have c93640 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.1
  have c93641 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_8
  simp only [← c93636, ← c93637, ← c93638] at e_1527_8
  have e_1527_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_9
  simp only [← c93639, ← c93640, k2, ← c93641] at e_1527_9
  have hr := e_1527_9
  simp only [e_1527_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14760)) := by
  have c93642 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93643 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93644 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93645 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93646 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93647 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_10
  simp only [← c93642, ← c93643, ← c93644] at e_1527_10
  have e_1527_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_11
  simp only [← c93645, ← c93646, k2, ← c93647] at e_1527_11
  have hr := e_1527_11
  simp only [e_1527_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14761)) := by
  have c93648 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93649 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93650 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93651 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93652 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93653 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_12
  simp only [← c93648, ← c93649, ← c93650] at e_1527_12
  have e_1527_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_13
  simp only [← c93651, ← c93652, k2, ← c93653] at e_1527_13
  have hr := e_1527_13
  simp only [e_1527_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14762)) := by
  have c93654 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93655 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93656 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93657 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93658 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93659 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_14
  simp only [← c93654, ← c93655, ← c93656] at e_1527_14
  have e_1527_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_15
  simp only [← c93657, ← c93658, k2, ← c93659] at e_1527_15
  have hr := e_1527_15
  simp only [e_1527_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14763)) := by
  have c93660 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93661 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93662 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93663 := (publicBatchWrapper89_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93664 := (publicBatchWrapper89_copies2927 a h).1
  have c93665 := (publicBatchWrapper89_copies2927 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_16
  simp only [← c93660, ← c93661, ← c93662] at e_1527_16
  have e_1527_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_17
  simp only [← c93663, ← c93664, k2, ← c93665] at e_1527_17
  have hr := e_1527_17
  simp only [e_1527_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1527 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14764)) := by
  have c93666 := (publicBatchWrapper89_copies2927 a h).2.2.1
  have c93667 := (publicBatchWrapper89_copies2927 a h).2.2.2.1
  have c93668 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.1
  have c93669 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.1
  have c93670 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.1
  have c93671 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1527_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_18
  simp only [← c93666, ← c93667, ← c93668] at e_1527_18
  have e_1527_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_19
  simp only [← c93669, ← c93670, k2, ← c93671] at e_1527_19
  have hr := e_1527_19
  simp only [e_1527_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14765)) := by
  have c93672 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.1
  have c93673 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.1
  have c93674 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93675 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93676 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93677 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_0
  simp only [← c93672, ← c93673, ← c93674] at e_1528_0
  have e_1528_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_1
  simp only [← c93675, ← c93676, k2, ← c93677] at e_1528_1
  have hr := e_1528_1
  simp only [e_1528_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14766)) := by
  have c93678 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93679 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93680 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93681 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93682 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93683 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_2
  simp only [← c93678, ← c93679, ← c93680] at e_1528_2
  have e_1528_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_3
  simp only [← c93681, ← c93682, k2, ← c93683] at e_1528_3
  have hr := e_1528_3
  simp only [e_1528_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14767)) := by
  have c93684 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93685 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93686 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93687 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93688 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93689 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_4
  simp only [← c93684, ← c93685, ← c93686] at e_1528_4
  have e_1528_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_5
  simp only [← c93687, ← c93688, k2, ← c93689] at e_1528_5
  have hr := e_1528_5
  simp only [e_1528_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14768)) := by
  have c93690 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93691 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93692 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93693 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93694 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93695 := (publicBatchWrapper89_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_6
  simp only [← c93690, ← c93691, ← c93692] at e_1528_6
  have e_1528_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_7
  simp only [← c93693, ← c93694, k2, ← c93695] at e_1528_7
  have hr := e_1528_7
  simp only [e_1528_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14769)) := by
  have c93696 := (publicBatchWrapper89_copies2928 a h).1
  have c93697 := (publicBatchWrapper89_copies2928 a h).2.1
  have c93698 := (publicBatchWrapper89_copies2928 a h).2.2.1
  have c93699 := (publicBatchWrapper89_copies2928 a h).2.2.2.1
  have c93700 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.1
  have c93701 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_8
  simp only [← c93696, ← c93697, ← c93698] at e_1528_8
  have e_1528_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_9
  simp only [← c93699, ← c93700, k2, ← c93701] at e_1528_9
  have hr := e_1528_9
  simp only [e_1528_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14810)) := by
  have c93702 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.1
  have c93703 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.1
  have c93704 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.1
  have c93705 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.1
  have c93706 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93707 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_10
  simp only [← c93702, ← c93703, ← c93704] at e_1528_10
  have e_1528_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_11
  simp only [← c93705, ← c93706, k2, ← c93707] at e_1528_11
  have hr := e_1528_11
  simp only [e_1528_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14811)) := by
  have c93708 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93709 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93710 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93711 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93712 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93713 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_12
  simp only [← c93708, ← c93709, ← c93710] at e_1528_12
  have e_1528_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_13
  simp only [← c93711, ← c93712, k2, ← c93713] at e_1528_13
  have hr := e_1528_13
  simp only [e_1528_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14812)) := by
  have c93714 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93715 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93716 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93717 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93718 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93719 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_14
  simp only [← c93714, ← c93715, ← c93716] at e_1528_14
  have e_1528_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_15
  simp only [← c93717, ← c93718, k2, ← c93719] at e_1528_15
  have hr := e_1528_15
  simp only [e_1528_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14813)) := by
  have c93720 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93721 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93722 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93723 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93724 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93725 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_16
  simp only [← c93720, ← c93721, ← c93722] at e_1528_16
  have e_1528_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_17
  simp only [← c93723, ← c93724, k2, ← c93725] at e_1528_17
  have hr := e_1528_17
  simp only [e_1528_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1528 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14814)) := by
  have c93726 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93727 := (publicBatchWrapper89_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93728 := (publicBatchWrapper89_copies2929 a h).1
  have c93729 := (publicBatchWrapper89_copies2929 a h).2.1
  have c93730 := (publicBatchWrapper89_copies2929 a h).2.2.1
  have c93731 := (publicBatchWrapper89_copies2929 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1528_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_18
  simp only [← c93726, ← c93727, ← c93728] at e_1528_18
  have e_1528_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_19
  simp only [← c93729, ← c93730, k2, ← c93731] at e_1528_19
  have hr := e_1528_19
  simp only [e_1528_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14815)) := by
  have c93732 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.1
  have c93733 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.1
  have c93734 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.1
  have c93735 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.1
  have c93736 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.1
  have c93737 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_0
  simp only [← c93732, ← c93733, ← c93734] at e_1529_0
  have e_1529_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_1
  simp only [← c93735, ← c93736, k2, ← c93737] at e_1529_1
  have hr := e_1529_1
  simp only [e_1529_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14816)) := by
  have c93738 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93739 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93740 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93741 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93742 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93743 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_2
  simp only [← c93738, ← c93739, ← c93740] at e_1529_2
  have e_1529_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_3
  simp only [← c93741, ← c93742, k2, ← c93743] at e_1529_3
  have hr := e_1529_3
  simp only [e_1529_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14817)) := by
  have c93744 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93745 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93746 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93747 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93748 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93749 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_4
  simp only [← c93744, ← c93745, ← c93746] at e_1529_4
  have e_1529_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_5
  simp only [← c93747, ← c93748, k2, ← c93749] at e_1529_5
  have hr := e_1529_5
  simp only [e_1529_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14818)) := by
  have c93750 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93751 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93752 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93753 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93754 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93755 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_6
  simp only [← c93750, ← c93751, ← c93752] at e_1529_6
  have e_1529_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_7
  simp only [← c93753, ← c93754, k2, ← c93755] at e_1529_7
  have hr := e_1529_7
  simp only [e_1529_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14819)) := by
  have c93756 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93757 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93758 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93759 := (publicBatchWrapper89_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93760 := (publicBatchWrapper89_copies2930 a h).1
  have c93761 := (publicBatchWrapper89_copies2930 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_8
  simp only [← c93756, ← c93757, ← c93758] at e_1529_8
  have e_1529_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_9
  simp only [← c93759, ← c93760, k2, ← c93761] at e_1529_9
  have hr := e_1529_9
  simp only [e_1529_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14820)) := by
  have c93762 := (publicBatchWrapper89_copies2930 a h).2.2.1
  have c93763 := (publicBatchWrapper89_copies2930 a h).2.2.2.1
  have c93764 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.1
  have c93765 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.1
  have c93766 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.1
  have c93767 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_10
  simp only [← c93762, ← c93763, ← c93764] at e_1529_10
  have e_1529_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_11
  simp only [← c93765, ← c93766, k2, ← c93767] at e_1529_11
  have hr := e_1529_11
  simp only [e_1529_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14821)) := by
  have c93768 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.1
  have c93769 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.1
  have c93770 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93771 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93772 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93773 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_12
  simp only [← c93768, ← c93769, ← c93770] at e_1529_12
  have e_1529_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_13
  simp only [← c93771, ← c93772, k2, ← c93773] at e_1529_13
  have hr := e_1529_13
  simp only [e_1529_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14822)) := by
  have c93774 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93775 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93776 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93777 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93778 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93779 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_14
  simp only [← c93774, ← c93775, ← c93776] at e_1529_14
  have e_1529_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_15
  simp only [← c93777, ← c93778, k2, ← c93779] at e_1529_15
  have hr := e_1529_15
  simp only [e_1529_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14823)) := by
  have c93780 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93781 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93782 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93783 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93784 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93785 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_16
  simp only [← c93780, ← c93781, ← c93782] at e_1529_16
  have e_1529_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_17
  simp only [← c93783, ← c93784, k2, ← c93785] at e_1529_17
  have hr := e_1529_17
  simp only [e_1529_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1529 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14824)) := by
  have c93786 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93787 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93788 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93789 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93790 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93791 := (publicBatchWrapper89_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1529_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_18
  simp only [← c93786, ← c93787, ← c93788] at e_1529_18
  have e_1529_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_19
  simp only [← c93789, ← c93790, k2, ← c93791] at e_1529_19
  have hr := e_1529_19
  simp only [e_1529_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14825)) := by
  have c93792 := (publicBatchWrapper89_copies2931 a h).1
  have c93793 := (publicBatchWrapper89_copies2931 a h).2.1
  have c93794 := (publicBatchWrapper89_copies2931 a h).2.2.1
  have c93795 := (publicBatchWrapper89_copies2931 a h).2.2.2.1
  have c93796 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.1
  have c93797 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_0
  simp only [← c93792, ← c93793, ← c93794] at e_1530_0
  have e_1530_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_1
  simp only [← c93795, ← c93796, k2, ← c93797] at e_1530_1
  have hr := e_1530_1
  simp only [e_1530_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14826)) := by
  have c93798 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.1
  have c93799 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.1
  have c93800 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.1
  have c93801 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.1
  have c93802 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93803 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_2
  simp only [← c93798, ← c93799, ← c93800] at e_1530_2
  have e_1530_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_3
  simp only [← c93801, ← c93802, k2, ← c93803] at e_1530_3
  have hr := e_1530_3
  simp only [e_1530_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14827)) := by
  have c93804 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93805 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93806 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93807 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93808 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93809 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_4
  simp only [← c93804, ← c93805, ← c93806] at e_1530_4
  have e_1530_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_5
  simp only [← c93807, ← c93808, k2, ← c93809] at e_1530_5
  have hr := e_1530_5
  simp only [e_1530_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14828)) := by
  have c93810 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93811 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93812 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93813 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93814 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93815 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_6
  simp only [← c93810, ← c93811, ← c93812] at e_1530_6
  have e_1530_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_7
  simp only [← c93813, ← c93814, k2, ← c93815] at e_1530_7
  have hr := e_1530_7
  simp only [e_1530_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14829)) := by
  have c93816 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93817 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93818 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93819 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93820 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93821 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_8
  simp only [← c93816, ← c93817, ← c93818] at e_1530_8
  have e_1530_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_9
  simp only [← c93819, ← c93820, k2, ← c93821] at e_1530_9
  have hr := e_1530_9
  simp only [e_1530_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14830)) := by
  have c93822 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93823 := (publicBatchWrapper89_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93824 := (publicBatchWrapper89_copies2932 a h).1
  have c93825 := (publicBatchWrapper89_copies2932 a h).2.1
  have c93826 := (publicBatchWrapper89_copies2932 a h).2.2.1
  have c93827 := (publicBatchWrapper89_copies2932 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_10
  simp only [← c93822, ← c93823, ← c93824] at e_1530_10
  have e_1530_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_11
  simp only [← c93825, ← c93826, k2, ← c93827] at e_1530_11
  have hr := e_1530_11
  simp only [e_1530_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14831)) := by
  have c93828 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.1
  have c93829 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.1
  have c93830 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.1
  have c93831 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.1
  have c93832 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.1
  have c93833 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_12
  simp only [← c93828, ← c93829, ← c93830] at e_1530_12
  have e_1530_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_13
  simp only [← c93831, ← c93832, k2, ← c93833] at e_1530_13
  have hr := e_1530_13
  simp only [e_1530_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14832)) := by
  have c93834 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93835 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93836 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93837 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93838 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93839 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_14
  simp only [← c93834, ← c93835, ← c93836] at e_1530_14
  have e_1530_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_15
  simp only [← c93837, ← c93838, k2, ← c93839] at e_1530_15
  have hr := e_1530_15
  simp only [e_1530_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14833)) := by
  have c93840 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93841 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93842 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93843 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93844 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93845 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_16
  simp only [← c93840, ← c93841, ← c93842] at e_1530_16
  have e_1530_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_17
  simp only [← c93843, ← c93844, k2, ← c93845] at e_1530_17
  have hr := e_1530_17
  simp only [e_1530_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1530 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14834)) := by
  have c93846 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93847 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93848 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93849 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93850 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93851 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1530_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_18
  simp only [← c93846, ← c93847, ← c93848] at e_1530_18
  have e_1530_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_19
  simp only [← c93849, ← c93850, k2, ← c93851] at e_1530_19
  have hr := e_1530_19
  simp only [e_1530_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14835)) := by
  have c93852 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93853 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93854 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93855 := (publicBatchWrapper89_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93856 := (publicBatchWrapper89_copies2933 a h).1
  have c93857 := (publicBatchWrapper89_copies2933 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_0
  simp only [← c93852, ← c93853, ← c93854] at e_1531_0
  have e_1531_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_1
  simp only [← c93855, ← c93856, k2, ← c93857] at e_1531_1
  have hr := e_1531_1
  simp only [e_1531_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14836)) := by
  have c93858 := (publicBatchWrapper89_copies2933 a h).2.2.1
  have c93859 := (publicBatchWrapper89_copies2933 a h).2.2.2.1
  have c93860 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.1
  have c93861 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.1
  have c93862 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.1
  have c93863 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_2
  simp only [← c93858, ← c93859, ← c93860] at e_1531_2
  have e_1531_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_3
  simp only [← c93861, ← c93862, k2, ← c93863] at e_1531_3
  have hr := e_1531_3
  simp only [e_1531_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14837)) := by
  have c93864 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.1
  have c93865 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.1
  have c93866 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93867 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93868 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93869 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_4
  simp only [← c93864, ← c93865, ← c93866] at e_1531_4
  have e_1531_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_5
  simp only [← c93867, ← c93868, k2, ← c93869] at e_1531_5
  have hr := e_1531_5
  simp only [e_1531_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14838)) := by
  have c93870 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93871 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93872 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93873 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93874 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93875 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_6
  simp only [← c93870, ← c93871, ← c93872] at e_1531_6
  have e_1531_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_7
  simp only [← c93873, ← c93874, k2, ← c93875] at e_1531_7
  have hr := e_1531_7
  simp only [e_1531_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14839)) := by
  have c93876 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93877 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93878 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93879 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93880 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93881 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_8
  simp only [← c93876, ← c93877, ← c93878] at e_1531_8
  have e_1531_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_9
  simp only [← c93879, ← c93880, k2, ← c93881] at e_1531_9
  have hr := e_1531_9
  simp only [e_1531_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14840)) := by
  have c93882 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93883 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93884 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93885 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93886 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93887 := (publicBatchWrapper89_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_10
  simp only [← c93882, ← c93883, ← c93884] at e_1531_10
  have e_1531_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_11
  simp only [← c93885, ← c93886, k2, ← c93887] at e_1531_11
  have hr := e_1531_11
  simp only [e_1531_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14841)) := by
  have c93888 := (publicBatchWrapper89_copies2934 a h).1
  have c93889 := (publicBatchWrapper89_copies2934 a h).2.1
  have c93890 := (publicBatchWrapper89_copies2934 a h).2.2.1
  have c93891 := (publicBatchWrapper89_copies2934 a h).2.2.2.1
  have c93892 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.1
  have c93893 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_12
  simp only [← c93888, ← c93889, ← c93890] at e_1531_12
  have e_1531_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_13
  simp only [← c93891, ← c93892, k2, ← c93893] at e_1531_13
  have hr := e_1531_13
  simp only [e_1531_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14842)) := by
  have c93894 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.1
  have c93895 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.1
  have c93896 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.1
  have c93897 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.1
  have c93898 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93899 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_14
  simp only [← c93894, ← c93895, ← c93896] at e_1531_14
  have e_1531_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_15
  simp only [← c93897, ← c93898, k2, ← c93899] at e_1531_15
  have hr := e_1531_15
  simp only [e_1531_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14843)) := by
  have c93900 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93901 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93902 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93903 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93904 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93905 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_16
  simp only [← c93900, ← c93901, ← c93902] at e_1531_16
  have e_1531_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_17
  simp only [← c93903, ← c93904, k2, ← c93905] at e_1531_17
  have hr := e_1531_17
  simp only [e_1531_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1531 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14844)) := by
  have c93906 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93907 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93908 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93909 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93910 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93911 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1531_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_18
  simp only [← c93906, ← c93907, ← c93908] at e_1531_18
  have e_1531_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_19
  simp only [← c93909, ← c93910, k2, ← c93911] at e_1531_19
  have hr := e_1531_19
  simp only [e_1531_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14845)) := by
  have c93912 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93913 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93914 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93915 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93916 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93917 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_0
  simp only [← c93912, ← c93913, ← c93914] at e_1532_0
  have e_1532_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_1
  simp only [← c93915, ← c93916, k2, ← c93917] at e_1532_1
  have hr := e_1532_1
  simp only [e_1532_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14846)) := by
  have c93918 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93919 := (publicBatchWrapper89_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93920 := (publicBatchWrapper89_copies2935 a h).1
  have c93921 := (publicBatchWrapper89_copies2935 a h).2.1
  have c93922 := (publicBatchWrapper89_copies2935 a h).2.2.1
  have c93923 := (publicBatchWrapper89_copies2935 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_2
  simp only [← c93918, ← c93919, ← c93920] at e_1532_2
  have e_1532_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_3
  simp only [← c93921, ← c93922, k2, ← c93923] at e_1532_3
  have hr := e_1532_3
  simp only [e_1532_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14847)) := by
  have c93924 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.1
  have c93925 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.1
  have c93926 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.1
  have c93927 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.1
  have c93928 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.1
  have c93929 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_4
  simp only [← c93924, ← c93925, ← c93926] at e_1532_4
  have e_1532_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_5
  simp only [← c93927, ← c93928, k2, ← c93929] at e_1532_5
  have hr := e_1532_5
  simp only [e_1532_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14848)) := by
  have c93930 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93931 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93932 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93933 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93934 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93935 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_6
  simp only [← c93930, ← c93931, ← c93932] at e_1532_6
  have e_1532_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_7
  simp only [← c93933, ← c93934, k2, ← c93935] at e_1532_7
  have hr := e_1532_7
  simp only [e_1532_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14849)) := by
  have c93936 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93937 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93938 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93939 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93940 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93941 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_8
  simp only [← c93936, ← c93937, ← c93938] at e_1532_8
  have e_1532_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_9
  simp only [← c93939, ← c93940, k2, ← c93941] at e_1532_9
  have hr := e_1532_9
  simp only [e_1532_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14850)) := by
  have c93942 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93943 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93944 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93945 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93946 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93947 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_10
  simp only [← c93942, ← c93943, ← c93944] at e_1532_10
  have e_1532_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_11
  simp only [← c93945, ← c93946, k2, ← c93947] at e_1532_11
  have hr := e_1532_11
  simp only [e_1532_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14851)) := by
  have c93948 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93949 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93950 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93951 := (publicBatchWrapper89_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93952 := (publicBatchWrapper89_copies2936 a h).1
  have c93953 := (publicBatchWrapper89_copies2936 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_12
  simp only [← c93948, ← c93949, ← c93950] at e_1532_12
  have e_1532_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_13
  simp only [← c93951, ← c93952, k2, ← c93953] at e_1532_13
  have hr := e_1532_13
  simp only [e_1532_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14852)) := by
  have c93954 := (publicBatchWrapper89_copies2936 a h).2.2.1
  have c93955 := (publicBatchWrapper89_copies2936 a h).2.2.2.1
  have c93956 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.1
  have c93957 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.1
  have c93958 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.1
  have c93959 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_14
  simp only [← c93954, ← c93955, ← c93956] at e_1532_14
  have e_1532_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_15
  simp only [← c93957, ← c93958, k2, ← c93959] at e_1532_15
  have hr := e_1532_15
  simp only [e_1532_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14853)) := by
  have c93960 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.1
  have c93961 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.1
  have c93962 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93963 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93964 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93965 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_16
  simp only [← c93960, ← c93961, ← c93962] at e_1532_16
  have e_1532_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_17
  simp only [← c93963, ← c93964, k2, ← c93965] at e_1532_17
  have hr := e_1532_17
  simp only [e_1532_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1532 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14854)) := by
  have c93966 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93967 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93968 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93969 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93970 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93971 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1532_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_18
  simp only [← c93966, ← c93967, ← c93968] at e_1532_18
  have e_1532_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_19
  simp only [← c93969, ← c93970, k2, ← c93971] at e_1532_19
  have hr := e_1532_19
  simp only [e_1532_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14855)) := by
  have c93972 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93973 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93974 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93975 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93976 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93977 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_0
  simp only [← c93972, ← c93973, ← c93974] at e_1533_0
  have e_1533_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_1
  simp only [← c93975, ← c93976, k2, ← c93977] at e_1533_1
  have hr := e_1533_1
  simp only [e_1533_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14856)) := by
  have c93978 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93979 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93980 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93981 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93982 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93983 := (publicBatchWrapper89_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_2
  simp only [← c93978, ← c93979, ← c93980] at e_1533_2
  have e_1533_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_3
  simp only [← c93981, ← c93982, k2, ← c93983] at e_1533_3
  have hr := e_1533_3
  simp only [e_1533_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14857)) := by
  have c93984 := (publicBatchWrapper89_copies2937 a h).1
  have c93985 := (publicBatchWrapper89_copies2937 a h).2.1
  have c93986 := (publicBatchWrapper89_copies2937 a h).2.2.1
  have c93987 := (publicBatchWrapper89_copies2937 a h).2.2.2.1
  have c93988 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.1
  have c93989 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_4
  simp only [← c93984, ← c93985, ← c93986] at e_1533_4
  have e_1533_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_5
  simp only [← c93987, ← c93988, k2, ← c93989] at e_1533_5
  have hr := e_1533_5
  simp only [e_1533_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14858)) := by
  have c93990 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.1
  have c93991 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.1
  have c93992 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.1
  have c93993 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.1
  have c93994 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93995 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_6
  simp only [← c93990, ← c93991, ← c93992] at e_1533_6
  have e_1533_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_7
  simp only [← c93993, ← c93994, k2, ← c93995] at e_1533_7
  have hr := e_1533_7
  simp only [e_1533_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14859)) := by
  have c93996 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93997 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93998 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93999 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94000 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94001 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_8
  simp only [← c93996, ← c93997, ← c93998] at e_1533_8
  have e_1533_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_9
  simp only [← c93999, ← c94000, k2, ← c94001] at e_1533_9
  have hr := e_1533_9
  simp only [e_1533_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14860)) := by
  have c94002 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94003 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94004 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94005 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94006 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94007 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_10
  simp only [← c94002, ← c94003, ← c94004] at e_1533_10
  have e_1533_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_11
  simp only [← c94005, ← c94006, k2, ← c94007] at e_1533_11
  have hr := e_1533_11
  simp only [e_1533_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14861)) := by
  have c94008 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94009 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94010 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94011 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94012 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94013 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_12
  simp only [← c94008, ← c94009, ← c94010] at e_1533_12
  have e_1533_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_13
  simp only [← c94011, ← c94012, k2, ← c94013] at e_1533_13
  have hr := e_1533_13
  simp only [e_1533_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14862)) := by
  have c94014 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94015 := (publicBatchWrapper89_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94016 := (publicBatchWrapper89_copies2938 a h).1
  have c94017 := (publicBatchWrapper89_copies2938 a h).2.1
  have c94018 := (publicBatchWrapper89_copies2938 a h).2.2.1
  have c94019 := (publicBatchWrapper89_copies2938 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_14
  simp only [← c94014, ← c94015, ← c94016] at e_1533_14
  have e_1533_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_15
  simp only [← c94017, ← c94018, k2, ← c94019] at e_1533_15
  have hr := e_1533_15
  simp only [e_1533_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14863)) := by
  have c94020 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.1
  have c94021 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.1
  have c94022 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.1
  have c94023 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.1
  have c94024 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.1
  have c94025 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_16
  simp only [← c94020, ← c94021, ← c94022] at e_1533_16
  have e_1533_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_17
  simp only [← c94023, ← c94024, k2, ← c94025] at e_1533_17
  have hr := e_1533_17
  simp only [e_1533_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1533 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14864)) := by
  have c94026 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94027 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94028 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94029 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94030 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94031 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1533_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_18
  simp only [← c94026, ← c94027, ← c94028] at e_1533_18
  have e_1533_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_19
  simp only [← c94029, ← c94030, k2, ← c94031] at e_1533_19
  have hr := e_1533_19
  simp only [e_1533_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14865)) := by
  have c94032 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94033 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94034 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94035 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94036 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94037 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_0
  simp only [← c94032, ← c94033, ← c94034] at e_1534_0
  have e_1534_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_1
  simp only [← c94035, ← c94036, k2, ← c94037] at e_1534_1
  have hr := e_1534_1
  simp only [e_1534_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14866)) := by
  have c94038 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94039 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94040 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94041 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94042 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94043 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_2
  simp only [← c94038, ← c94039, ← c94040] at e_1534_2
  have e_1534_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_3
  simp only [← c94041, ← c94042, k2, ← c94043] at e_1534_3
  have hr := e_1534_3
  simp only [e_1534_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14867)) := by
  have c94044 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94045 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94046 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94047 := (publicBatchWrapper89_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94048 := (publicBatchWrapper89_copies2939 a h).1
  have c94049 := (publicBatchWrapper89_copies2939 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_4
  simp only [← c94044, ← c94045, ← c94046] at e_1534_4
  have e_1534_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_5
  simp only [← c94047, ← c94048, k2, ← c94049] at e_1534_5
  have hr := e_1534_5
  simp only [e_1534_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14868)) := by
  have c94050 := (publicBatchWrapper89_copies2939 a h).2.2.1
  have c94051 := (publicBatchWrapper89_copies2939 a h).2.2.2.1
  have c94052 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.1
  have c94053 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.1
  have c94054 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.1
  have c94055 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_6
  simp only [← c94050, ← c94051, ← c94052] at e_1534_6
  have e_1534_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_7
  simp only [← c94053, ← c94054, k2, ← c94055] at e_1534_7
  have hr := e_1534_7
  simp only [e_1534_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14869)) := by
  have c94056 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.1
  have c94057 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.1
  have c94058 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94059 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94060 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94061 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_8
  simp only [← c94056, ← c94057, ← c94058] at e_1534_8
  have e_1534_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_9
  simp only [← c94059, ← c94060, k2, ← c94061] at e_1534_9
  have hr := e_1534_9
  simp only [e_1534_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14870)) := by
  have c94062 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94063 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94064 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94065 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94066 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94067 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_10
  simp only [← c94062, ← c94063, ← c94064] at e_1534_10
  have e_1534_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_11
  simp only [← c94065, ← c94066, k2, ← c94067] at e_1534_11
  have hr := e_1534_11
  simp only [e_1534_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14871)) := by
  have c94068 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94069 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94070 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94071 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94072 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94073 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_12
  simp only [← c94068, ← c94069, ← c94070] at e_1534_12
  have e_1534_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_13
  simp only [← c94071, ← c94072, k2, ← c94073] at e_1534_13
  have hr := e_1534_13
  simp only [e_1534_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14872)) := by
  have c94074 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94075 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94076 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94077 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94078 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94079 := (publicBatchWrapper89_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_14
  simp only [← c94074, ← c94075, ← c94076] at e_1534_14
  have e_1534_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_15
  simp only [← c94077, ← c94078, k2, ← c94079] at e_1534_15
  have hr := e_1534_15
  simp only [e_1534_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14873)) := by
  have c94080 := (publicBatchWrapper89_copies2940 a h).1
  have c94081 := (publicBatchWrapper89_copies2940 a h).2.1
  have c94082 := (publicBatchWrapper89_copies2940 a h).2.2.1
  have c94083 := (publicBatchWrapper89_copies2940 a h).2.2.2.1
  have c94084 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.1
  have c94085 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_16
  simp only [← c94080, ← c94081, ← c94082] at e_1534_16
  have e_1534_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_17
  simp only [← c94083, ← c94084, k2, ← c94085] at e_1534_17
  have hr := e_1534_17
  simp only [e_1534_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1534 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14874)) := by
  have c94086 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.1
  have c94087 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.1
  have c94088 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.1
  have c94089 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.1
  have c94090 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94091 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1534_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_18
  simp only [← c94086, ← c94087, ← c94088] at e_1534_18
  have e_1534_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_19
  simp only [← c94089, ← c94090, k2, ← c94091] at e_1534_19
  have hr := e_1534_19
  simp only [e_1534_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14875)) := by
  have c94092 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94093 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94094 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94095 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94096 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94097 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_0
  simp only [← c94092, ← c94093, ← c94094] at e_1535_0
  have e_1535_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_1
  simp only [← c94095, ← c94096, k2, ← c94097] at e_1535_1
  have hr := e_1535_1
  simp only [e_1535_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14876)) := by
  have c94098 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94099 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94100 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94101 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94102 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94103 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_2
  simp only [← c94098, ← c94099, ← c94100] at e_1535_2
  have e_1535_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_3
  simp only [← c94101, ← c94102, k2, ← c94103] at e_1535_3
  have hr := e_1535_3
  simp only [e_1535_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14877)) := by
  have c94104 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94105 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94106 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94107 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94108 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94109 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_4
  simp only [← c94104, ← c94105, ← c94106] at e_1535_4
  have e_1535_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_5
  simp only [← c94107, ← c94108, k2, ← c94109] at e_1535_5
  have hr := e_1535_5
  simp only [e_1535_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14878)) := by
  have c94110 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94111 := (publicBatchWrapper89_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94112 := (publicBatchWrapper89_copies2941 a h).1
  have c94113 := (publicBatchWrapper89_copies2941 a h).2.1
  have c94114 := (publicBatchWrapper89_copies2941 a h).2.2.1
  have c94115 := (publicBatchWrapper89_copies2941 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_6
  simp only [← c94110, ← c94111, ← c94112] at e_1535_6
  have e_1535_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_7
  simp only [← c94113, ← c94114, k2, ← c94115] at e_1535_7
  have hr := e_1535_7
  simp only [e_1535_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14879)) := by
  have c94116 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.1
  have c94117 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.1
  have c94118 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.1
  have c94119 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.1
  have c94120 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.1
  have c94121 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_8
  simp only [← c94116, ← c94117, ← c94118] at e_1535_8
  have e_1535_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_9
  simp only [← c94119, ← c94120, k2, ← c94121] at e_1535_9
  have hr := e_1535_9
  simp only [e_1535_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14880)) := by
  have c94122 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94123 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94124 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94125 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94126 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94127 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_10
  simp only [← c94122, ← c94123, ← c94124] at e_1535_10
  have e_1535_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_11
  simp only [← c94125, ← c94126, k2, ← c94127] at e_1535_11
  have hr := e_1535_11
  simp only [e_1535_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14881)) := by
  have c94128 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94129 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94130 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94131 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94132 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94133 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_12
  simp only [← c94128, ← c94129, ← c94130] at e_1535_12
  have e_1535_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_13
  simp only [← c94131, ← c94132, k2, ← c94133] at e_1535_13
  have hr := e_1535_13
  simp only [e_1535_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14882)) := by
  have c94134 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94135 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94136 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94137 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94138 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94139 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_14
  simp only [← c94134, ← c94135, ← c94136] at e_1535_14
  have e_1535_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_15
  simp only [← c94137, ← c94138, k2, ← c94139] at e_1535_15
  have hr := e_1535_15
  simp only [e_1535_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14883)) := by
  have c94140 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94141 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94142 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94143 := (publicBatchWrapper89_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94144 := (publicBatchWrapper89_copies2942 a h).1
  have c94145 := (publicBatchWrapper89_copies2942 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_16
  simp only [← c94140, ← c94141, ← c94142] at e_1535_16
  have e_1535_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_17
  simp only [← c94143, ← c94144, k2, ← c94145] at e_1535_17
  have hr := e_1535_17
  simp only [e_1535_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1535 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14884)) := by
  have c94146 := (publicBatchWrapper89_copies2942 a h).2.2.1
  have c94147 := (publicBatchWrapper89_copies2942 a h).2.2.2.1
  have c94148 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.1
  have c94149 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.1
  have c94150 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.1
  have c94151 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1535_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_18
  simp only [← c94146, ← c94147, ← c94148] at e_1535_18
  have e_1535_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_19
  simp only [← c94149, ← c94150, k2, ← c94151] at e_1535_19
  have hr := e_1535_19
  simp only [e_1535_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14885)) := by
  have c94152 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.1
  have c94153 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.1
  have c94154 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94155 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94156 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94157 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_0
  simp only [← c94152, ← c94153, ← c94154] at e_1536_0
  have e_1536_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_1
  simp only [← c94155, ← c94156, k2, ← c94157] at e_1536_1
  have hr := e_1536_1
  simp only [e_1536_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14886)) := by
  have c94158 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94159 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94160 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94161 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94162 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94163 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_2
  simp only [← c94158, ← c94159, ← c94160] at e_1536_2
  have e_1536_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_3
  simp only [← c94161, ← c94162, k2, ← c94163] at e_1536_3
  have hr := e_1536_3
  simp only [e_1536_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14887)) := by
  have c94164 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94165 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94166 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94167 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94168 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94169 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_4
  simp only [← c94164, ← c94165, ← c94166] at e_1536_4
  have e_1536_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_5
  simp only [← c94167, ← c94168, k2, ← c94169] at e_1536_5
  have hr := e_1536_5
  simp only [e_1536_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14888)) := by
  have c94170 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94171 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94172 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94173 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94174 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94175 := (publicBatchWrapper89_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_6
  simp only [← c94170, ← c94171, ← c94172] at e_1536_6
  have e_1536_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_7
  simp only [← c94173, ← c94174, k2, ← c94175] at e_1536_7
  have hr := e_1536_7
  simp only [e_1536_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14889)) := by
  have c94176 := (publicBatchWrapper89_copies2943 a h).1
  have c94177 := (publicBatchWrapper89_copies2943 a h).2.1
  have c94178 := (publicBatchWrapper89_copies2943 a h).2.2.1
  have c94179 := (publicBatchWrapper89_copies2943 a h).2.2.2.1
  have c94180 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.1
  have c94181 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_8
  simp only [← c94176, ← c94177, ← c94178] at e_1536_8
  have e_1536_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_9
  simp only [← c94179, ← c94180, k2, ← c94181] at e_1536_9
  have hr := e_1536_9
  simp only [e_1536_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14890)) := by
  have c94182 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.1
  have c94183 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.1
  have c94184 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.1
  have c94185 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.1
  have c94186 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94187 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_10
  simp only [← c94182, ← c94183, ← c94184] at e_1536_10
  have e_1536_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_11
  simp only [← c94185, ← c94186, k2, ← c94187] at e_1536_11
  have hr := e_1536_11
  simp only [e_1536_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14891)) := by
  have c94188 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94189 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94190 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94191 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94192 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94193 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_12
  simp only [← c94188, ← c94189, ← c94190] at e_1536_12
  have e_1536_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_13
  simp only [← c94191, ← c94192, k2, ← c94193] at e_1536_13
  have hr := e_1536_13
  simp only [e_1536_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14892)) := by
  have c94194 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94195 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94196 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94197 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94198 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94199 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_14
  simp only [← c94194, ← c94195, ← c94196] at e_1536_14
  have e_1536_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_15
  simp only [← c94197, ← c94198, k2, ← c94199] at e_1536_15
  have hr := e_1536_15
  simp only [e_1536_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14893)) := by
  have c94200 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94201 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94202 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94203 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94204 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94205 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_16
  simp only [← c94200, ← c94201, ← c94202] at e_1536_16
  have e_1536_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_17
  simp only [← c94203, ← c94204, k2, ← c94205] at e_1536_17
  have hr := e_1536_17
  simp only [e_1536_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1536 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14894)) := by
  have c94206 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94207 := (publicBatchWrapper89_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94208 := (publicBatchWrapper89_copies2944 a h).1
  have c94209 := (publicBatchWrapper89_copies2944 a h).2.1
  have c94210 := (publicBatchWrapper89_copies2944 a h).2.2.1
  have c94211 := (publicBatchWrapper89_copies2944 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1536_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_18
  simp only [← c94206, ← c94207, ← c94208] at e_1536_18
  have e_1536_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_19
  simp only [← c94209, ← c94210, k2, ← c94211] at e_1536_19
  have hr := e_1536_19
  simp only [e_1536_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14895)) := by
  have c94212 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.1
  have c94213 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.1
  have c94214 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.1
  have c94215 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.1
  have c94216 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.1
  have c94217 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_0
  simp only [← c94212, ← c94213, ← c94214] at e_1537_0
  have e_1537_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_1
  simp only [← c94215, ← c94216, k2, ← c94217] at e_1537_1
  have hr := e_1537_1
  simp only [e_1537_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14896)) := by
  have c94218 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94219 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94220 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94221 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94222 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94223 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_2
  simp only [← c94218, ← c94219, ← c94220] at e_1537_2
  have e_1537_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_3
  simp only [← c94221, ← c94222, k2, ← c94223] at e_1537_3
  have hr := e_1537_3
  simp only [e_1537_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14897)) := by
  have c94224 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94225 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94226 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94227 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94228 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94229 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_4
  simp only [← c94224, ← c94225, ← c94226] at e_1537_4
  have e_1537_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_5
  simp only [← c94227, ← c94228, k2, ← c94229] at e_1537_5
  have hr := e_1537_5
  simp only [e_1537_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14898)) := by
  have c94230 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94231 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94232 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94233 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94234 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94235 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_6
  simp only [← c94230, ← c94231, ← c94232] at e_1537_6
  have e_1537_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_7
  simp only [← c94233, ← c94234, k2, ← c94235] at e_1537_7
  have hr := e_1537_7
  simp only [e_1537_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14899)) := by
  have c94236 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94237 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94238 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94239 := (publicBatchWrapper89_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94240 := (publicBatchWrapper89_copies2945 a h).1
  have c94241 := (publicBatchWrapper89_copies2945 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_8
  simp only [← c94236, ← c94237, ← c94238] at e_1537_8
  have e_1537_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_9
  simp only [← c94239, ← c94240, k2, ← c94241] at e_1537_9
  have hr := e_1537_9
  simp only [e_1537_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14900)) := by
  have c94242 := (publicBatchWrapper89_copies2945 a h).2.2.1
  have c94243 := (publicBatchWrapper89_copies2945 a h).2.2.2.1
  have c94244 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.1
  have c94245 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.1
  have c94246 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.1
  have c94247 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_10
  simp only [← c94242, ← c94243, ← c94244] at e_1537_10
  have e_1537_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_11
  simp only [← c94245, ← c94246, k2, ← c94247] at e_1537_11
  have hr := e_1537_11
  simp only [e_1537_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14901)) := by
  have c94248 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.1
  have c94249 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.1
  have c94250 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94251 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94252 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94253 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_12
  simp only [← c94248, ← c94249, ← c94250] at e_1537_12
  have e_1537_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_13
  simp only [← c94251, ← c94252, k2, ← c94253] at e_1537_13
  have hr := e_1537_13
  simp only [e_1537_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14902)) := by
  have c94254 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94255 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94256 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94257 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94258 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94259 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_14
  simp only [← c94254, ← c94255, ← c94256] at e_1537_14
  have e_1537_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_15
  simp only [← c94257, ← c94258, k2, ← c94259] at e_1537_15
  have hr := e_1537_15
  simp only [e_1537_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14903)) := by
  have c94260 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94261 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94262 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94263 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94264 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94265 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_16
  simp only [← c94260, ← c94261, ← c94262] at e_1537_16
  have e_1537_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_17
  simp only [← c94263, ← c94264, k2, ← c94265] at e_1537_17
  have hr := e_1537_17
  simp only [e_1537_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1537 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14904)) := by
  have c94266 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94267 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94268 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94269 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94270 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94271 := (publicBatchWrapper89_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1537_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_18
  simp only [← c94266, ← c94267, ← c94268] at e_1537_18
  have e_1537_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_19
  simp only [← c94269, ← c94270, k2, ← c94271] at e_1537_19
  have hr := e_1537_19
  simp only [e_1537_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14905)) := by
  have c94272 := (publicBatchWrapper89_copies2946 a h).1
  have c94273 := (publicBatchWrapper89_copies2946 a h).2.1
  have c94274 := (publicBatchWrapper89_copies2946 a h).2.2.1
  have c94275 := (publicBatchWrapper89_copies2946 a h).2.2.2.1
  have c94276 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.1
  have c94277 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_0
  simp only [← c94272, ← c94273, ← c94274] at e_1538_0
  have e_1538_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_1
  simp only [← c94275, ← c94276, k2, ← c94277] at e_1538_1
  have hr := e_1538_1
  simp only [e_1538_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14906)) := by
  have c94278 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.1
  have c94279 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.1
  have c94280 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.1
  have c94281 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.1
  have c94282 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94283 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_2
  simp only [← c94278, ← c94279, ← c94280] at e_1538_2
  have e_1538_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_3
  simp only [← c94281, ← c94282, k2, ← c94283] at e_1538_3
  have hr := e_1538_3
  simp only [e_1538_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14907)) := by
  have c94284 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94285 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94286 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94287 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94288 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94289 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_4
  simp only [← c94284, ← c94285, ← c94286] at e_1538_4
  have e_1538_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_5
  simp only [← c94287, ← c94288, k2, ← c94289] at e_1538_5
  have hr := e_1538_5
  simp only [e_1538_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14908)) := by
  have c94290 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94291 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94292 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94293 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94294 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94295 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_6
  simp only [← c94290, ← c94291, ← c94292] at e_1538_6
  have e_1538_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_7
  simp only [← c94293, ← c94294, k2, ← c94295] at e_1538_7
  have hr := e_1538_7
  simp only [e_1538_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14909)) := by
  have c94296 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94297 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94298 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94299 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94300 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94301 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_8
  simp only [← c94296, ← c94297, ← c94298] at e_1538_8
  have e_1538_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_9
  simp only [← c94299, ← c94300, k2, ← c94301] at e_1538_9
  have hr := e_1538_9
  simp only [e_1538_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14910)) := by
  have c94302 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94303 := (publicBatchWrapper89_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94304 := (publicBatchWrapper89_copies2947 a h).1
  have c94305 := (publicBatchWrapper89_copies2947 a h).2.1
  have c94306 := (publicBatchWrapper89_copies2947 a h).2.2.1
  have c94307 := (publicBatchWrapper89_copies2947 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_10
  simp only [← c94302, ← c94303, ← c94304] at e_1538_10
  have e_1538_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_11
  simp only [← c94305, ← c94306, k2, ← c94307] at e_1538_11
  have hr := e_1538_11
  simp only [e_1538_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14911)) := by
  have c94308 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.1
  have c94309 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.1
  have c94310 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.1
  have c94311 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.1
  have c94312 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.1
  have c94313 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_12
  simp only [← c94308, ← c94309, ← c94310] at e_1538_12
  have e_1538_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_13
  simp only [← c94311, ← c94312, k2, ← c94313] at e_1538_13
  have hr := e_1538_13
  simp only [e_1538_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14912)) := by
  have c94314 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94315 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94316 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94317 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94318 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94319 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_14
  simp only [← c94314, ← c94315, ← c94316] at e_1538_14
  have e_1538_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_15
  simp only [← c94317, ← c94318, k2, ← c94319] at e_1538_15
  have hr := e_1538_15
  simp only [e_1538_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14913)) := by
  have c94320 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94321 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94322 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94323 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94324 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94325 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_16
  simp only [← c94320, ← c94321, ← c94322] at e_1538_16
  have e_1538_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_17
  simp only [← c94323, ← c94324, k2, ← c94325] at e_1538_17
  have hr := e_1538_17
  simp only [e_1538_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1538 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14914)) := by
  have c94326 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94327 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94328 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94329 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94330 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94331 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1538_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_18
  simp only [← c94326, ← c94327, ← c94328] at e_1538_18
  have e_1538_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_19
  simp only [← c94329, ← c94330, k2, ← c94331] at e_1538_19
  have hr := e_1538_19
  simp only [e_1538_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14915)) := by
  have c94332 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94333 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94334 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94335 := (publicBatchWrapper89_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94336 := (publicBatchWrapper89_copies2948 a h).1
  have c94337 := (publicBatchWrapper89_copies2948 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_0
  simp only [← c94332, ← c94333, ← c94334] at e_1539_0
  have e_1539_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_1
  simp only [← c94335, ← c94336, k2, ← c94337] at e_1539_1
  have hr := e_1539_1
  simp only [e_1539_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14916)) := by
  have c94338 := (publicBatchWrapper89_copies2948 a h).2.2.1
  have c94339 := (publicBatchWrapper89_copies2948 a h).2.2.2.1
  have c94340 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.1
  have c94341 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.1
  have c94342 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.1
  have c94343 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_2
  simp only [← c94338, ← c94339, ← c94340] at e_1539_2
  have e_1539_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_3
  simp only [← c94341, ← c94342, k2, ← c94343] at e_1539_3
  have hr := e_1539_3
  simp only [e_1539_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14917)) := by
  have c94344 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.1
  have c94345 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.1
  have c94346 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94347 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94348 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94349 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_4
  simp only [← c94344, ← c94345, ← c94346] at e_1539_4
  have e_1539_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_5
  simp only [← c94347, ← c94348, k2, ← c94349] at e_1539_5
  have hr := e_1539_5
  simp only [e_1539_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14918)) := by
  have c94350 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94351 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94352 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94353 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94354 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94355 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_6
  simp only [← c94350, ← c94351, ← c94352] at e_1539_6
  have e_1539_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_7
  simp only [← c94353, ← c94354, k2, ← c94355] at e_1539_7
  have hr := e_1539_7
  simp only [e_1539_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14919)) := by
  have c94356 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94357 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94358 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94359 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94360 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94361 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_8
  simp only [← c94356, ← c94357, ← c94358] at e_1539_8
  have e_1539_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_9
  simp only [← c94359, ← c94360, k2, ← c94361] at e_1539_9
  have hr := e_1539_9
  simp only [e_1539_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14920)) := by
  have c94362 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94363 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94364 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94365 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94366 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94367 := (publicBatchWrapper89_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_10
  simp only [← c94362, ← c94363, ← c94364] at e_1539_10
  have e_1539_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_11
  simp only [← c94365, ← c94366, k2, ← c94367] at e_1539_11
  have hr := e_1539_11
  simp only [e_1539_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14921)) := by
  have c94368 := (publicBatchWrapper89_copies2949 a h).1
  have c94369 := (publicBatchWrapper89_copies2949 a h).2.1
  have c94370 := (publicBatchWrapper89_copies2949 a h).2.2.1
  have c94371 := (publicBatchWrapper89_copies2949 a h).2.2.2.1
  have c94372 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.1
  have c94373 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_12
  simp only [← c94368, ← c94369, ← c94370] at e_1539_12
  have e_1539_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_13
  simp only [← c94371, ← c94372, k2, ← c94373] at e_1539_13
  have hr := e_1539_13
  simp only [e_1539_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14922)) := by
  have c94374 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.1
  have c94375 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.1
  have c94376 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.1
  have c94377 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.1
  have c94378 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94379 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_14
  simp only [← c94374, ← c94375, ← c94376] at e_1539_14
  have e_1539_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_15
  simp only [← c94377, ← c94378, k2, ← c94379] at e_1539_15
  have hr := e_1539_15
  simp only [e_1539_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14923)) := by
  have c94380 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94381 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94382 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94383 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94384 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94385 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_16
  simp only [← c94380, ← c94381, ← c94382] at e_1539_16
  have e_1539_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_17
  simp only [← c94383, ← c94384, k2, ← c94385] at e_1539_17
  have hr := e_1539_17
  simp only [e_1539_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1539 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14924)) := by
  have c94386 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94387 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94388 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94389 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94390 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94391 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1539_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_18
  simp only [← c94386, ← c94387, ← c94388] at e_1539_18
  have e_1539_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_19
  simp only [← c94389, ← c94390, k2, ← c94391] at e_1539_19
  have hr := e_1539_19
  simp only [e_1539_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14925)) := by
  have c94392 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94393 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94394 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94395 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94396 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94397 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_0
  simp only [← c94392, ← c94393, ← c94394] at e_1540_0
  have e_1540_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_1
  simp only [← c94395, ← c94396, k2, ← c94397] at e_1540_1
  have hr := e_1540_1
  simp only [e_1540_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14926)) := by
  have c94398 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94399 := (publicBatchWrapper89_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94400 := (publicBatchWrapper89_copies2950 a h).1
  have c94401 := (publicBatchWrapper89_copies2950 a h).2.1
  have c94402 := (publicBatchWrapper89_copies2950 a h).2.2.1
  have c94403 := (publicBatchWrapper89_copies2950 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_2
  simp only [← c94398, ← c94399, ← c94400] at e_1540_2
  have e_1540_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_3
  simp only [← c94401, ← c94402, k2, ← c94403] at e_1540_3
  have hr := e_1540_3
  simp only [e_1540_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14927)) := by
  have c94404 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.1
  have c94405 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.1
  have c94406 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.1
  have c94407 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.1
  have c94408 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.1
  have c94409 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_4
  simp only [← c94404, ← c94405, ← c94406] at e_1540_4
  have e_1540_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_5
  simp only [← c94407, ← c94408, k2, ← c94409] at e_1540_5
  have hr := e_1540_5
  simp only [e_1540_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14928)) := by
  have c94410 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94411 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94412 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94413 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94414 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94415 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_6
  simp only [← c94410, ← c94411, ← c94412] at e_1540_6
  have e_1540_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_7
  simp only [← c94413, ← c94414, k2, ← c94415] at e_1540_7
  have hr := e_1540_7
  simp only [e_1540_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14929)) := by
  have c94416 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94417 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94418 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94419 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94420 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94421 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_8
  simp only [← c94416, ← c94417, ← c94418] at e_1540_8
  have e_1540_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_9
  simp only [← c94419, ← c94420, k2, ← c94421] at e_1540_9
  have hr := e_1540_9
  simp only [e_1540_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14930)) := by
  have c94422 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94423 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94424 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94425 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94426 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94427 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_10
  simp only [← c94422, ← c94423, ← c94424] at e_1540_10
  have e_1540_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_11
  simp only [← c94425, ← c94426, k2, ← c94427] at e_1540_11
  have hr := e_1540_11
  simp only [e_1540_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14931)) := by
  have c94428 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94429 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94430 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94431 := (publicBatchWrapper89_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94432 := (publicBatchWrapper89_copies2951 a h).1
  have c94433 := (publicBatchWrapper89_copies2951 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_12
  simp only [← c94428, ← c94429, ← c94430] at e_1540_12
  have e_1540_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_13
  simp only [← c94431, ← c94432, k2, ← c94433] at e_1540_13
  have hr := e_1540_13
  simp only [e_1540_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14932)) := by
  have c94434 := (publicBatchWrapper89_copies2951 a h).2.2.1
  have c94435 := (publicBatchWrapper89_copies2951 a h).2.2.2.1
  have c94436 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.1
  have c94437 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.1
  have c94438 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.1
  have c94439 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_14
  simp only [← c94434, ← c94435, ← c94436] at e_1540_14
  have e_1540_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_15
  simp only [← c94437, ← c94438, k2, ← c94439] at e_1540_15
  have hr := e_1540_15
  simp only [e_1540_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14933)) := by
  have c94440 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.1
  have c94441 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.1
  have c94442 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94443 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94444 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94445 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_16
  simp only [← c94440, ← c94441, ← c94442] at e_1540_16
  have e_1540_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_17
  simp only [← c94443, ← c94444, k2, ← c94445] at e_1540_17
  have hr := e_1540_17
  simp only [e_1540_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1540 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14934)) := by
  have c94446 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94447 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94448 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94449 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94450 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94451 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1540_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_18
  simp only [← c94446, ← c94447, ← c94448] at e_1540_18
  have e_1540_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_19
  simp only [← c94449, ← c94450, k2, ← c94451] at e_1540_19
  have hr := e_1540_19
  simp only [e_1540_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14935)) := by
  have c94452 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94453 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94454 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94455 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94456 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94457 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_0
  simp only [← c94452, ← c94453, ← c94454] at e_1541_0
  have e_1541_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_1
  simp only [← c94455, ← c94456, k2, ← c94457] at e_1541_1
  have hr := e_1541_1
  simp only [e_1541_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14936)) := by
  have c94458 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94459 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94460 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94461 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94462 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94463 := (publicBatchWrapper89_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_2
  simp only [← c94458, ← c94459, ← c94460] at e_1541_2
  have e_1541_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_3
  simp only [← c94461, ← c94462, k2, ← c94463] at e_1541_3
  have hr := e_1541_3
  simp only [e_1541_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14937)) := by
  have c94464 := (publicBatchWrapper89_copies2952 a h).1
  have c94465 := (publicBatchWrapper89_copies2952 a h).2.1
  have c94466 := (publicBatchWrapper89_copies2952 a h).2.2.1
  have c94467 := (publicBatchWrapper89_copies2952 a h).2.2.2.1
  have c94468 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.1
  have c94469 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_4
  simp only [← c94464, ← c94465, ← c94466] at e_1541_4
  have e_1541_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_5
  simp only [← c94467, ← c94468, k2, ← c94469] at e_1541_5
  have hr := e_1541_5
  simp only [e_1541_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14938)) := by
  have c94470 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.1
  have c94471 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.1
  have c94472 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.1
  have c94473 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.1
  have c94474 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94475 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_6
  simp only [← c94470, ← c94471, ← c94472] at e_1541_6
  have e_1541_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_7
  simp only [← c94473, ← c94474, k2, ← c94475] at e_1541_7
  have hr := e_1541_7
  simp only [e_1541_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14939)) := by
  have c94476 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94477 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94478 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94479 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94480 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94481 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_8
  simp only [← c94476, ← c94477, ← c94478] at e_1541_8
  have e_1541_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_9
  simp only [← c94479, ← c94480, k2, ← c94481] at e_1541_9
  have hr := e_1541_9
  simp only [e_1541_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14940)) := by
  have c94482 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94483 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94484 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94485 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94486 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94487 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_10
  simp only [← c94482, ← c94483, ← c94484] at e_1541_10
  have e_1541_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_11
  simp only [← c94485, ← c94486, k2, ← c94487] at e_1541_11
  have hr := e_1541_11
  simp only [e_1541_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14941)) := by
  have c94488 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94489 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94490 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94491 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94492 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94493 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_12
  simp only [← c94488, ← c94489, ← c94490] at e_1541_12
  have e_1541_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_13
  simp only [← c94491, ← c94492, k2, ← c94493] at e_1541_13
  have hr := e_1541_13
  simp only [e_1541_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14942)) := by
  have c94494 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94495 := (publicBatchWrapper89_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94496 := (publicBatchWrapper89_copies2953 a h).1
  have c94497 := (publicBatchWrapper89_copies2953 a h).2.1
  have c94498 := (publicBatchWrapper89_copies2953 a h).2.2.1
  have c94499 := (publicBatchWrapper89_copies2953 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_14
  simp only [← c94494, ← c94495, ← c94496] at e_1541_14
  have e_1541_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_15
  simp only [← c94497, ← c94498, k2, ← c94499] at e_1541_15
  have hr := e_1541_15
  simp only [e_1541_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14943)) := by
  have c94500 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.1
  have c94501 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.1
  have c94502 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.1
  have c94503 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.1
  have c94504 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.1
  have c94505 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_16
  simp only [← c94500, ← c94501, ← c94502] at e_1541_16
  have e_1541_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_17
  simp only [← c94503, ← c94504, k2, ← c94505] at e_1541_17
  have hr := e_1541_17
  simp only [e_1541_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1541 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14944)) := by
  have c94506 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94507 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94508 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94509 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94510 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94511 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1541_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_18
  simp only [← c94506, ← c94507, ← c94508] at e_1541_18
  have e_1541_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_19
  simp only [← c94509, ← c94510, k2, ← c94511] at e_1541_19
  have hr := e_1541_19
  simp only [e_1541_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14945)) := by
  have c94512 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94513 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94514 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94515 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94516 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94517 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_0
  simp only [← c94512, ← c94513, ← c94514] at e_1542_0
  have e_1542_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_1
  simp only [← c94515, ← c94516, k2, ← c94517] at e_1542_1
  have hr := e_1542_1
  simp only [e_1542_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14946)) := by
  have c94518 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94519 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94520 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94521 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94522 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94523 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_2
  simp only [← c94518, ← c94519, ← c94520] at e_1542_2
  have e_1542_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_3
  simp only [← c94521, ← c94522, k2, ← c94523] at e_1542_3
  have hr := e_1542_3
  simp only [e_1542_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14947)) := by
  have c94524 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94525 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94526 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94527 := (publicBatchWrapper89_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94528 := (publicBatchWrapper89_copies2954 a h).1
  have c94529 := (publicBatchWrapper89_copies2954 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_4
  simp only [← c94524, ← c94525, ← c94526] at e_1542_4
  have e_1542_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_5
  simp only [← c94527, ← c94528, k2, ← c94529] at e_1542_5
  have hr := e_1542_5
  simp only [e_1542_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14948)) := by
  have c94530 := (publicBatchWrapper89_copies2954 a h).2.2.1
  have c94531 := (publicBatchWrapper89_copies2954 a h).2.2.2.1
  have c94532 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.1
  have c94533 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.1
  have c94534 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.1
  have c94535 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_6
  simp only [← c94530, ← c94531, ← c94532] at e_1542_6
  have e_1542_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_7
  simp only [← c94533, ← c94534, k2, ← c94535] at e_1542_7
  have hr := e_1542_7
  simp only [e_1542_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14949)) := by
  have c94536 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.1
  have c94537 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.1
  have c94538 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94539 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94540 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94541 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_8
  simp only [← c94536, ← c94537, ← c94538] at e_1542_8
  have e_1542_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_9
  simp only [← c94539, ← c94540, k2, ← c94541] at e_1542_9
  have hr := e_1542_9
  simp only [e_1542_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14950)) := by
  have c94542 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94543 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94544 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94545 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94546 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94547 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_10
  simp only [← c94542, ← c94543, ← c94544] at e_1542_10
  have e_1542_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_11
  simp only [← c94545, ← c94546, k2, ← c94547] at e_1542_11
  have hr := e_1542_11
  simp only [e_1542_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14951)) := by
  have c94548 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94549 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94550 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94551 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94552 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94553 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_12
  simp only [← c94548, ← c94549, ← c94550] at e_1542_12
  have e_1542_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_13
  simp only [← c94551, ← c94552, k2, ← c94553] at e_1542_13
  have hr := e_1542_13
  simp only [e_1542_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14952)) := by
  have c94554 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94555 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94556 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94557 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94558 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94559 := (publicBatchWrapper89_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_14
  simp only [← c94554, ← c94555, ← c94556] at e_1542_14
  have e_1542_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_15
  simp only [← c94557, ← c94558, k2, ← c94559] at e_1542_15
  have hr := e_1542_15
  simp only [e_1542_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14953)) := by
  have c94560 := (publicBatchWrapper89_copies2955 a h).1
  have c94561 := (publicBatchWrapper89_copies2955 a h).2.1
  have c94562 := (publicBatchWrapper89_copies2955 a h).2.2.1
  have c94563 := (publicBatchWrapper89_copies2955 a h).2.2.2.1
  have c94564 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.1
  have c94565 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_16
  simp only [← c94560, ← c94561, ← c94562] at e_1542_16
  have e_1542_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_17
  simp only [← c94563, ← c94564, k2, ← c94565] at e_1542_17
  have hr := e_1542_17
  simp only [e_1542_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1542 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14954)) := by
  have c94566 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.1
  have c94567 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.1
  have c94568 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.1
  have c94569 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.1
  have c94570 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94571 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1542_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_18
  simp only [← c94566, ← c94567, ← c94568] at e_1542_18
  have e_1542_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_19
  simp only [← c94569, ← c94570, k2, ← c94571] at e_1542_19
  have hr := e_1542_19
  simp only [e_1542_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14955)) := by
  have c94572 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94573 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94574 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94575 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94576 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94577 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_0
  simp only [← c94572, ← c94573, ← c94574] at e_1543_0
  have e_1543_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_1
  simp only [← c94575, ← c94576, k2, ← c94577] at e_1543_1
  have hr := e_1543_1
  simp only [e_1543_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14956)) := by
  have c94578 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94579 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94580 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94581 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94582 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94583 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_2
  simp only [← c94578, ← c94579, ← c94580] at e_1543_2
  have e_1543_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_3
  simp only [← c94581, ← c94582, k2, ← c94583] at e_1543_3
  have hr := e_1543_3
  simp only [e_1543_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14957)) := by
  have c94584 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94585 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94586 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94587 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94588 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94589 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_4
  simp only [← c94584, ← c94585, ← c94586] at e_1543_4
  have e_1543_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_5
  simp only [← c94587, ← c94588, k2, ← c94589] at e_1543_5
  have hr := e_1543_5
  simp only [e_1543_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14958)) := by
  have c94590 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94591 := (publicBatchWrapper89_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94592 := (publicBatchWrapper89_copies2956 a h).1
  have c94593 := (publicBatchWrapper89_copies2956 a h).2.1
  have c94594 := (publicBatchWrapper89_copies2956 a h).2.2.1
  have c94595 := (publicBatchWrapper89_copies2956 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_6
  simp only [← c94590, ← c94591, ← c94592] at e_1543_6
  have e_1543_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_7
  simp only [← c94593, ← c94594, k2, ← c94595] at e_1543_7
  have hr := e_1543_7
  simp only [e_1543_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14959)) := by
  have c94596 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.1
  have c94597 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.1
  have c94598 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.1
  have c94599 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.1
  have c94600 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.1
  have c94601 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_8
  simp only [← c94596, ← c94597, ← c94598] at e_1543_8
  have e_1543_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_9
  simp only [← c94599, ← c94600, k2, ← c94601] at e_1543_9
  have hr := e_1543_9
  simp only [e_1543_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14960)) := by
  have c94602 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94603 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94604 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94605 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94606 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94607 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_10
  simp only [← c94602, ← c94603, ← c94604] at e_1543_10
  have e_1543_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_11
  simp only [← c94605, ← c94606, k2, ← c94607] at e_1543_11
  have hr := e_1543_11
  simp only [e_1543_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14961)) := by
  have c94608 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94609 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94610 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94611 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94612 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94613 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_12
  simp only [← c94608, ← c94609, ← c94610] at e_1543_12
  have e_1543_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_13
  simp only [← c94611, ← c94612, k2, ← c94613] at e_1543_13
  have hr := e_1543_13
  simp only [e_1543_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14962)) := by
  have c94614 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94615 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94616 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94617 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94618 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94619 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_14
  simp only [← c94614, ← c94615, ← c94616] at e_1543_14
  have e_1543_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_15
  simp only [← c94617, ← c94618, k2, ← c94619] at e_1543_15
  have hr := e_1543_15
  simp only [e_1543_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14963)) := by
  have c94620 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94621 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94622 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94623 := (publicBatchWrapper89_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94624 := (publicBatchWrapper89_copies2957 a h).1
  have c94625 := (publicBatchWrapper89_copies2957 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_16
  simp only [← c94620, ← c94621, ← c94622] at e_1543_16
  have e_1543_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_17
  simp only [← c94623, ← c94624, k2, ← c94625] at e_1543_17
  have hr := e_1543_17
  simp only [e_1543_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1543 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14964)) := by
  have c94626 := (publicBatchWrapper89_copies2957 a h).2.2.1
  have c94627 := (publicBatchWrapper89_copies2957 a h).2.2.2.1
  have c94628 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.1
  have c94629 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.1
  have c94630 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.1
  have c94631 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1543_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_18
  simp only [← c94626, ← c94627, ← c94628] at e_1543_18
  have e_1543_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_19
  simp only [← c94629, ← c94630, k2, ← c94631] at e_1543_19
  have hr := e_1543_19
  simp only [e_1543_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14965)) := by
  have c94632 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.1
  have c94633 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.1
  have c94634 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94635 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94636 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94637 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_0
  simp only [← c94632, ← c94633, ← c94634] at e_1544_0
  have e_1544_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_1
  simp only [← c94635, ← c94636, k2, ← c94637] at e_1544_1
  have hr := e_1544_1
  simp only [e_1544_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14966)) := by
  have c94638 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94639 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94640 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94641 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94642 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94643 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_2
  simp only [← c94638, ← c94639, ← c94640] at e_1544_2
  have e_1544_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_3
  simp only [← c94641, ← c94642, k2, ← c94643] at e_1544_3
  have hr := e_1544_3
  simp only [e_1544_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14967)) := by
  have c94644 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94645 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94646 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94647 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94648 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94649 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_4
  simp only [← c94644, ← c94645, ← c94646] at e_1544_4
  have e_1544_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_5
  simp only [← c94647, ← c94648, k2, ← c94649] at e_1544_5
  have hr := e_1544_5
  simp only [e_1544_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14968)) := by
  have c94650 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94651 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94652 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94653 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94654 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94655 := (publicBatchWrapper89_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_6
  simp only [← c94650, ← c94651, ← c94652] at e_1544_6
  have e_1544_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_7
  simp only [← c94653, ← c94654, k2, ← c94655] at e_1544_7
  have hr := e_1544_7
  simp only [e_1544_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14969)) := by
  have c94656 := (publicBatchWrapper89_copies2958 a h).1
  have c94657 := (publicBatchWrapper89_copies2958 a h).2.1
  have c94658 := (publicBatchWrapper89_copies2958 a h).2.2.1
  have c94659 := (publicBatchWrapper89_copies2958 a h).2.2.2.1
  have c94660 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.1
  have c94661 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_8
  simp only [← c94656, ← c94657, ← c94658] at e_1544_8
  have e_1544_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_9
  simp only [← c94659, ← c94660, k2, ← c94661] at e_1544_9
  have hr := e_1544_9
  simp only [e_1544_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15010)) := by
  have c94662 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.1
  have c94663 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.1
  have c94664 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.1
  have c94665 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.1
  have c94666 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94667 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_10
  simp only [← c94662, ← c94663, ← c94664] at e_1544_10
  have e_1544_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_11
  simp only [← c94665, ← c94666, k2, ← c94667] at e_1544_11
  have hr := e_1544_11
  simp only [e_1544_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15011)) := by
  have c94668 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94669 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94670 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94671 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94672 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94673 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_12
  simp only [← c94668, ← c94669, ← c94670] at e_1544_12
  have e_1544_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_13
  simp only [← c94671, ← c94672, k2, ← c94673] at e_1544_13
  have hr := e_1544_13
  simp only [e_1544_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15012)) := by
  have c94674 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94675 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94676 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94677 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94678 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94679 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_14
  simp only [← c94674, ← c94675, ← c94676] at e_1544_14
  have e_1544_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_15
  simp only [← c94677, ← c94678, k2, ← c94679] at e_1544_15
  have hr := e_1544_15
  simp only [e_1544_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15013)) := by
  have c94680 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94681 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94682 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94683 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94684 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94685 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_16
  simp only [← c94680, ← c94681, ← c94682] at e_1544_16
  have e_1544_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_17
  simp only [← c94683, ← c94684, k2, ← c94685] at e_1544_17
  have hr := e_1544_17
  simp only [e_1544_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1544 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15014)) := by
  have c94686 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94687 := (publicBatchWrapper89_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94688 := (publicBatchWrapper89_copies2959 a h).1
  have c94689 := (publicBatchWrapper89_copies2959 a h).2.1
  have c94690 := (publicBatchWrapper89_copies2959 a h).2.2.1
  have c94691 := (publicBatchWrapper89_copies2959 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1544_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_18
  simp only [← c94686, ← c94687, ← c94688] at e_1544_18
  have e_1544_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_19
  simp only [← c94689, ← c94690, k2, ← c94691] at e_1544_19
  have hr := e_1544_19
  simp only [e_1544_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15015)) := by
  have c94692 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.1
  have c94693 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.1
  have c94694 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.1
  have c94695 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.1
  have c94696 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.1
  have c94697 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_0
  simp only [← c94692, ← c94693, ← c94694] at e_1545_0
  have e_1545_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_1
  simp only [← c94695, ← c94696, k2, ← c94697] at e_1545_1
  have hr := e_1545_1
  simp only [e_1545_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15016)) := by
  have c94698 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94699 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94700 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94701 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94702 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94703 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_2
  simp only [← c94698, ← c94699, ← c94700] at e_1545_2
  have e_1545_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_3
  simp only [← c94701, ← c94702, k2, ← c94703] at e_1545_3
  have hr := e_1545_3
  simp only [e_1545_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15017)) := by
  have c94704 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94705 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94706 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94707 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94708 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94709 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_4
  simp only [← c94704, ← c94705, ← c94706] at e_1545_4
  have e_1545_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_5
  simp only [← c94707, ← c94708, k2, ← c94709] at e_1545_5
  have hr := e_1545_5
  simp only [e_1545_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15018)) := by
  have c94710 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94711 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94712 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94713 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94714 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94715 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_6
  simp only [← c94710, ← c94711, ← c94712] at e_1545_6
  have e_1545_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_7
  simp only [← c94713, ← c94714, k2, ← c94715] at e_1545_7
  have hr := e_1545_7
  simp only [e_1545_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15019)) := by
  have c94716 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94717 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94718 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94719 := (publicBatchWrapper89_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94720 := (publicBatchWrapper89_copies2960 a h).1
  have c94721 := (publicBatchWrapper89_copies2960 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_8
  simp only [← c94716, ← c94717, ← c94718] at e_1545_8
  have e_1545_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_9
  simp only [← c94719, ← c94720, k2, ← c94721] at e_1545_9
  have hr := e_1545_9
  simp only [e_1545_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15020)) := by
  have c94722 := (publicBatchWrapper89_copies2960 a h).2.2.1
  have c94723 := (publicBatchWrapper89_copies2960 a h).2.2.2.1
  have c94724 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.1
  have c94725 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.1
  have c94726 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.1
  have c94727 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_10
  simp only [← c94722, ← c94723, ← c94724] at e_1545_10
  have e_1545_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_11
  simp only [← c94725, ← c94726, k2, ← c94727] at e_1545_11
  have hr := e_1545_11
  simp only [e_1545_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15021)) := by
  have c94728 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.1
  have c94729 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.1
  have c94730 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94731 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94732 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94733 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_12
  simp only [← c94728, ← c94729, ← c94730] at e_1545_12
  have e_1545_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_13
  simp only [← c94731, ← c94732, k2, ← c94733] at e_1545_13
  have hr := e_1545_13
  simp only [e_1545_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15022)) := by
  have c94734 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94735 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94736 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94737 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94738 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94739 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_14
  simp only [← c94734, ← c94735, ← c94736] at e_1545_14
  have e_1545_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_15
  simp only [← c94737, ← c94738, k2, ← c94739] at e_1545_15
  have hr := e_1545_15
  simp only [e_1545_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15023)) := by
  have c94740 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94741 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94742 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94743 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94744 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94745 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_16
  simp only [← c94740, ← c94741, ← c94742] at e_1545_16
  have e_1545_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_17
  simp only [← c94743, ← c94744, k2, ← c94745] at e_1545_17
  have hr := e_1545_17
  simp only [e_1545_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1545 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15024)) := by
  have c94746 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94747 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94748 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94749 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94750 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94751 := (publicBatchWrapper89_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1545_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_18
  simp only [← c94746, ← c94747, ← c94748] at e_1545_18
  have e_1545_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_19
  simp only [← c94749, ← c94750, k2, ← c94751] at e_1545_19
  have hr := e_1545_19
  simp only [e_1545_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15025)) := by
  have c94752 := (publicBatchWrapper89_copies2961 a h).1
  have c94753 := (publicBatchWrapper89_copies2961 a h).2.1
  have c94754 := (publicBatchWrapper89_copies2961 a h).2.2.1
  have c94755 := (publicBatchWrapper89_copies2961 a h).2.2.2.1
  have c94756 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.1
  have c94757 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_0
  simp only [← c94752, ← c94753, ← c94754] at e_1546_0
  have e_1546_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_1
  simp only [← c94755, ← c94756, k2, ← c94757] at e_1546_1
  have hr := e_1546_1
  simp only [e_1546_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15026)) := by
  have c94758 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.1
  have c94759 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.1
  have c94760 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.1
  have c94761 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.1
  have c94762 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94763 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_2
  simp only [← c94758, ← c94759, ← c94760] at e_1546_2
  have e_1546_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_3
  simp only [← c94761, ← c94762, k2, ← c94763] at e_1546_3
  have hr := e_1546_3
  simp only [e_1546_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15027)) := by
  have c94764 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94765 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94766 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94767 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94768 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94769 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_4
  simp only [← c94764, ← c94765, ← c94766] at e_1546_4
  have e_1546_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_5
  simp only [← c94767, ← c94768, k2, ← c94769] at e_1546_5
  have hr := e_1546_5
  simp only [e_1546_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15028)) := by
  have c94770 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94771 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94772 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94773 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94774 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94775 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_6
  simp only [← c94770, ← c94771, ← c94772] at e_1546_6
  have e_1546_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_7
  simp only [← c94773, ← c94774, k2, ← c94775] at e_1546_7
  have hr := e_1546_7
  simp only [e_1546_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15029)) := by
  have c94776 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94777 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94778 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94779 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94780 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94781 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_8
  simp only [← c94776, ← c94777, ← c94778] at e_1546_8
  have e_1546_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_9
  simp only [← c94779, ← c94780, k2, ← c94781] at e_1546_9
  have hr := e_1546_9
  simp only [e_1546_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15030)) := by
  have c94782 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94783 := (publicBatchWrapper89_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94784 := (publicBatchWrapper89_copies2962 a h).1
  have c94785 := (publicBatchWrapper89_copies2962 a h).2.1
  have c94786 := (publicBatchWrapper89_copies2962 a h).2.2.1
  have c94787 := (publicBatchWrapper89_copies2962 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_10
  simp only [← c94782, ← c94783, ← c94784] at e_1546_10
  have e_1546_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_11
  simp only [← c94785, ← c94786, k2, ← c94787] at e_1546_11
  have hr := e_1546_11
  simp only [e_1546_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15031)) := by
  have c94788 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.1
  have c94789 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.1
  have c94790 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.1
  have c94791 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.1
  have c94792 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.1
  have c94793 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_12
  simp only [← c94788, ← c94789, ← c94790] at e_1546_12
  have e_1546_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_13
  simp only [← c94791, ← c94792, k2, ← c94793] at e_1546_13
  have hr := e_1546_13
  simp only [e_1546_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15032)) := by
  have c94794 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94795 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94796 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94797 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94798 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94799 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_14
  simp only [← c94794, ← c94795, ← c94796] at e_1546_14
  have e_1546_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_15
  simp only [← c94797, ← c94798, k2, ← c94799] at e_1546_15
  have hr := e_1546_15
  simp only [e_1546_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15033)) := by
  have c94800 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94801 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94802 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94803 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94804 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94805 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_16
  simp only [← c94800, ← c94801, ← c94802] at e_1546_16
  have e_1546_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_17
  simp only [← c94803, ← c94804, k2, ← c94805] at e_1546_17
  have hr := e_1546_17
  simp only [e_1546_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1546 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15034)) := by
  have c94806 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94807 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94808 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94809 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94810 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94811 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1546_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_18
  simp only [← c94806, ← c94807, ← c94808] at e_1546_18
  have e_1546_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_19
  simp only [← c94809, ← c94810, k2, ← c94811] at e_1546_19
  have hr := e_1546_19
  simp only [e_1546_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15035)) := by
  have c94812 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94813 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94814 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94815 := (publicBatchWrapper89_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94816 := (publicBatchWrapper89_copies2963 a h).1
  have c94817 := (publicBatchWrapper89_copies2963 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_0
  simp only [← c94812, ← c94813, ← c94814] at e_1547_0
  have e_1547_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_1
  simp only [← c94815, ← c94816, k2, ← c94817] at e_1547_1
  have hr := e_1547_1
  simp only [e_1547_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15036)) := by
  have c94818 := (publicBatchWrapper89_copies2963 a h).2.2.1
  have c94819 := (publicBatchWrapper89_copies2963 a h).2.2.2.1
  have c94820 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.1
  have c94821 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.1
  have c94822 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.1
  have c94823 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_2
  simp only [← c94818, ← c94819, ← c94820] at e_1547_2
  have e_1547_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_3
  simp only [← c94821, ← c94822, k2, ← c94823] at e_1547_3
  have hr := e_1547_3
  simp only [e_1547_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15037)) := by
  have c94824 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.1
  have c94825 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.1
  have c94826 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94827 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94828 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94829 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_4
  simp only [← c94824, ← c94825, ← c94826] at e_1547_4
  have e_1547_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_5
  simp only [← c94827, ← c94828, k2, ← c94829] at e_1547_5
  have hr := e_1547_5
  simp only [e_1547_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15038)) := by
  have c94830 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94831 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94832 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94833 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94834 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94835 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_6
  simp only [← c94830, ← c94831, ← c94832] at e_1547_6
  have e_1547_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_7
  simp only [← c94833, ← c94834, k2, ← c94835] at e_1547_7
  have hr := e_1547_7
  simp only [e_1547_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15039)) := by
  have c94836 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94837 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94838 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94839 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94840 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94841 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_8
  simp only [← c94836, ← c94837, ← c94838] at e_1547_8
  have e_1547_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_9
  simp only [← c94839, ← c94840, k2, ← c94841] at e_1547_9
  have hr := e_1547_9
  simp only [e_1547_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15040)) := by
  have c94842 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94843 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94844 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94845 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94846 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94847 := (publicBatchWrapper89_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_10
  simp only [← c94842, ← c94843, ← c94844] at e_1547_10
  have e_1547_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_11
  simp only [← c94845, ← c94846, k2, ← c94847] at e_1547_11
  have hr := e_1547_11
  simp only [e_1547_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15041)) := by
  have c94848 := (publicBatchWrapper89_copies2964 a h).1
  have c94849 := (publicBatchWrapper89_copies2964 a h).2.1
  have c94850 := (publicBatchWrapper89_copies2964 a h).2.2.1
  have c94851 := (publicBatchWrapper89_copies2964 a h).2.2.2.1
  have c94852 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.1
  have c94853 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_12
  simp only [← c94848, ← c94849, ← c94850] at e_1547_12
  have e_1547_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_13
  simp only [← c94851, ← c94852, k2, ← c94853] at e_1547_13
  have hr := e_1547_13
  simp only [e_1547_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15042)) := by
  have c94854 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.1
  have c94855 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.1
  have c94856 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.1
  have c94857 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.1
  have c94858 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94859 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_14
  simp only [← c94854, ← c94855, ← c94856] at e_1547_14
  have e_1547_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_15
  simp only [← c94857, ← c94858, k2, ← c94859] at e_1547_15
  have hr := e_1547_15
  simp only [e_1547_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15043)) := by
  have c94860 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94861 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94862 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94863 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94864 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94865 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_16
  simp only [← c94860, ← c94861, ← c94862] at e_1547_16
  have e_1547_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_17
  simp only [← c94863, ← c94864, k2, ← c94865] at e_1547_17
  have hr := e_1547_17
  simp only [e_1547_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1547 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15044)) := by
  have c94866 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94867 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94868 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94869 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94870 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94871 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1547_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_18
  simp only [← c94866, ← c94867, ← c94868] at e_1547_18
  have e_1547_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_19
  simp only [← c94869, ← c94870, k2, ← c94871] at e_1547_19
  have hr := e_1547_19
  simp only [e_1547_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15045)) := by
  have c94872 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94873 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94874 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94875 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94876 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94877 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_0
  simp only [← c94872, ← c94873, ← c94874] at e_1548_0
  have e_1548_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_1
  simp only [← c94875, ← c94876, k2, ← c94877] at e_1548_1
  have hr := e_1548_1
  simp only [e_1548_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15046)) := by
  have c94878 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94879 := (publicBatchWrapper89_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94880 := (publicBatchWrapper89_copies2965 a h).1
  have c94881 := (publicBatchWrapper89_copies2965 a h).2.1
  have c94882 := (publicBatchWrapper89_copies2965 a h).2.2.1
  have c94883 := (publicBatchWrapper89_copies2965 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_2
  simp only [← c94878, ← c94879, ← c94880] at e_1548_2
  have e_1548_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_3
  simp only [← c94881, ← c94882, k2, ← c94883] at e_1548_3
  have hr := e_1548_3
  simp only [e_1548_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15047)) := by
  have c94884 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.1
  have c94885 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.1
  have c94886 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.1
  have c94887 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.1
  have c94888 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.1
  have c94889 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_4
  simp only [← c94884, ← c94885, ← c94886] at e_1548_4
  have e_1548_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_5
  simp only [← c94887, ← c94888, k2, ← c94889] at e_1548_5
  have hr := e_1548_5
  simp only [e_1548_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15048)) := by
  have c94890 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94891 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94892 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94893 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94894 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94895 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_6
  simp only [← c94890, ← c94891, ← c94892] at e_1548_6
  have e_1548_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_7
  simp only [← c94893, ← c94894, k2, ← c94895] at e_1548_7
  have hr := e_1548_7
  simp only [e_1548_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15049)) := by
  have c94896 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94897 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94898 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94899 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94900 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94901 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_8
  simp only [← c94896, ← c94897, ← c94898] at e_1548_8
  have e_1548_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_9
  simp only [← c94899, ← c94900, k2, ← c94901] at e_1548_9
  have hr := e_1548_9
  simp only [e_1548_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15050)) := by
  have c94902 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94903 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94904 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94905 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94906 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94907 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_10
  simp only [← c94902, ← c94903, ← c94904] at e_1548_10
  have e_1548_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_11
  simp only [← c94905, ← c94906, k2, ← c94907] at e_1548_11
  have hr := e_1548_11
  simp only [e_1548_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15051)) := by
  have c94908 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94909 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94910 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94911 := (publicBatchWrapper89_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94912 := (publicBatchWrapper89_copies2966 a h).1
  have c94913 := (publicBatchWrapper89_copies2966 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_12
  simp only [← c94908, ← c94909, ← c94910] at e_1548_12
  have e_1548_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_13
  simp only [← c94911, ← c94912, k2, ← c94913] at e_1548_13
  have hr := e_1548_13
  simp only [e_1548_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15052)) := by
  have c94914 := (publicBatchWrapper89_copies2966 a h).2.2.1
  have c94915 := (publicBatchWrapper89_copies2966 a h).2.2.2.1
  have c94916 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.1
  have c94917 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.1
  have c94918 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.1
  have c94919 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_14
  simp only [← c94914, ← c94915, ← c94916] at e_1548_14
  have e_1548_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_15
  simp only [← c94917, ← c94918, k2, ← c94919] at e_1548_15
  have hr := e_1548_15
  simp only [e_1548_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15053)) := by
  have c94920 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.1
  have c94921 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.1
  have c94922 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94923 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94924 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94925 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_16
  simp only [← c94920, ← c94921, ← c94922] at e_1548_16
  have e_1548_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_17
  simp only [← c94923, ← c94924, k2, ← c94925] at e_1548_17
  have hr := e_1548_17
  simp only [e_1548_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1548 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15054)) := by
  have c94926 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94927 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94928 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94929 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94930 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94931 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1548_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_18
  simp only [← c94926, ← c94927, ← c94928] at e_1548_18
  have e_1548_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_19
  simp only [← c94929, ← c94930, k2, ← c94931] at e_1548_19
  have hr := e_1548_19
  simp only [e_1548_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15055)) := by
  have c94932 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94933 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94934 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94935 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94936 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94937 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_0
  simp only [← c94932, ← c94933, ← c94934] at e_1549_0
  have e_1549_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_1
  simp only [← c94935, ← c94936, k2, ← c94937] at e_1549_1
  have hr := e_1549_1
  simp only [e_1549_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15056)) := by
  have c94938 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94939 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94940 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94941 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94942 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94943 := (publicBatchWrapper89_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_2
  simp only [← c94938, ← c94939, ← c94940] at e_1549_2
  have e_1549_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_3
  simp only [← c94941, ← c94942, k2, ← c94943] at e_1549_3
  have hr := e_1549_3
  simp only [e_1549_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15057)) := by
  have c94944 := (publicBatchWrapper89_copies2967 a h).1
  have c94945 := (publicBatchWrapper89_copies2967 a h).2.1
  have c94946 := (publicBatchWrapper89_copies2967 a h).2.2.1
  have c94947 := (publicBatchWrapper89_copies2967 a h).2.2.2.1
  have c94948 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.1
  have c94949 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_4
  simp only [← c94944, ← c94945, ← c94946] at e_1549_4
  have e_1549_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_5
  simp only [← c94947, ← c94948, k2, ← c94949] at e_1549_5
  have hr := e_1549_5
  simp only [e_1549_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15058)) := by
  have c94950 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.1
  have c94951 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.1
  have c94952 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.1
  have c94953 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.1
  have c94954 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94955 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_6
  simp only [← c94950, ← c94951, ← c94952] at e_1549_6
  have e_1549_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_7
  simp only [← c94953, ← c94954, k2, ← c94955] at e_1549_7
  have hr := e_1549_7
  simp only [e_1549_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15059)) := by
  have c94956 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94957 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94958 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94959 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94960 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94961 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_8
  simp only [← c94956, ← c94957, ← c94958] at e_1549_8
  have e_1549_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_9
  simp only [← c94959, ← c94960, k2, ← c94961] at e_1549_9
  have hr := e_1549_9
  simp only [e_1549_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15060)) := by
  have c94962 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94963 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94964 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94965 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94966 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94967 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_10
  simp only [← c94962, ← c94963, ← c94964] at e_1549_10
  have e_1549_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_11
  simp only [← c94965, ← c94966, k2, ← c94967] at e_1549_11
  have hr := e_1549_11
  simp only [e_1549_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15061)) := by
  have c94968 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94969 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94970 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94971 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94972 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94973 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_12
  simp only [← c94968, ← c94969, ← c94970] at e_1549_12
  have e_1549_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_13
  simp only [← c94971, ← c94972, k2, ← c94973] at e_1549_13
  have hr := e_1549_13
  simp only [e_1549_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15062)) := by
  have c94974 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94975 := (publicBatchWrapper89_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94976 := (publicBatchWrapper89_copies2968 a h).1
  have c94977 := (publicBatchWrapper89_copies2968 a h).2.1
  have c94978 := (publicBatchWrapper89_copies2968 a h).2.2.1
  have c94979 := (publicBatchWrapper89_copies2968 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_14
  simp only [← c94974, ← c94975, ← c94976] at e_1549_14
  have e_1549_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_15
  simp only [← c94977, ← c94978, k2, ← c94979] at e_1549_15
  have hr := e_1549_15
  simp only [e_1549_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15063)) := by
  have c94980 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.1
  have c94981 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.1
  have c94982 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.1
  have c94983 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.1
  have c94984 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.1
  have c94985 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_16
  simp only [← c94980, ← c94981, ← c94982] at e_1549_16
  have e_1549_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_17
  simp only [← c94983, ← c94984, k2, ← c94985] at e_1549_17
  have hr := e_1549_17
  simp only [e_1549_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1549 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15064)) := by
  have c94986 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94987 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94988 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94989 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94990 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94991 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1549_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_18
  simp only [← c94986, ← c94987, ← c94988] at e_1549_18
  have e_1549_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_19
  simp only [← c94989, ← c94990, k2, ← c94991] at e_1549_19
  have hr := e_1549_19
  simp only [e_1549_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15065)) := by
  have c94992 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94993 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94994 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94995 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94996 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94997 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_0
  simp only [← c94992, ← c94993, ← c94994] at e_1550_0
  have e_1550_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_1
  simp only [← c94995, ← c94996, k2, ← c94997] at e_1550_1
  have hr := e_1550_1
  simp only [e_1550_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15066)) := by
  have c94998 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94999 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95000 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95001 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95002 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95003 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_2
  simp only [← c94998, ← c94999, ← c95000] at e_1550_2
  have e_1550_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_3
  simp only [← c95001, ← c95002, k2, ← c95003] at e_1550_3
  have hr := e_1550_3
  simp only [e_1550_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15067)) := by
  have c95004 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95005 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95006 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95007 := (publicBatchWrapper89_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95008 := (publicBatchWrapper89_copies2969 a h).1
  have c95009 := (publicBatchWrapper89_copies2969 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_4
  simp only [← c95004, ← c95005, ← c95006] at e_1550_4
  have e_1550_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_5
  simp only [← c95007, ← c95008, k2, ← c95009] at e_1550_5
  have hr := e_1550_5
  simp only [e_1550_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15068)) := by
  have c95010 := (publicBatchWrapper89_copies2969 a h).2.2.1
  have c95011 := (publicBatchWrapper89_copies2969 a h).2.2.2.1
  have c95012 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.1
  have c95013 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.1
  have c95014 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.1
  have c95015 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_6
  simp only [← c95010, ← c95011, ← c95012] at e_1550_6
  have e_1550_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_7
  simp only [← c95013, ← c95014, k2, ← c95015] at e_1550_7
  have hr := e_1550_7
  simp only [e_1550_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15069)) := by
  have c95016 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.1
  have c95017 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.1
  have c95018 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95019 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95020 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95021 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_8
  simp only [← c95016, ← c95017, ← c95018] at e_1550_8
  have e_1550_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_9
  simp only [← c95019, ← c95020, k2, ← c95021] at e_1550_9
  have hr := e_1550_9
  simp only [e_1550_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15070)) := by
  have c95022 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95023 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95024 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95025 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95026 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95027 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_10
  simp only [← c95022, ← c95023, ← c95024] at e_1550_10
  have e_1550_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_11
  simp only [← c95025, ← c95026, k2, ← c95027] at e_1550_11
  have hr := e_1550_11
  simp only [e_1550_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15071)) := by
  have c95028 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95029 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95030 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95031 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95032 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95033 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_12
  simp only [← c95028, ← c95029, ← c95030] at e_1550_12
  have e_1550_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_13
  simp only [← c95031, ← c95032, k2, ← c95033] at e_1550_13
  have hr := e_1550_13
  simp only [e_1550_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15072)) := by
  have c95034 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95035 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95036 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95037 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95038 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95039 := (publicBatchWrapper89_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_14
  simp only [← c95034, ← c95035, ← c95036] at e_1550_14
  have e_1550_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_15
  simp only [← c95037, ← c95038, k2, ← c95039] at e_1550_15
  have hr := e_1550_15
  simp only [e_1550_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15073)) := by
  have c95040 := (publicBatchWrapper89_copies2970 a h).1
  have c95041 := (publicBatchWrapper89_copies2970 a h).2.1
  have c95042 := (publicBatchWrapper89_copies2970 a h).2.2.1
  have c95043 := (publicBatchWrapper89_copies2970 a h).2.2.2.1
  have c95044 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.1
  have c95045 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_16
  simp only [← c95040, ← c95041, ← c95042] at e_1550_16
  have e_1550_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_17
  simp only [← c95043, ← c95044, k2, ← c95045] at e_1550_17
  have hr := e_1550_17
  simp only [e_1550_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1550 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15074)) := by
  have c95046 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.1
  have c95047 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.1
  have c95048 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.1
  have c95049 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.1
  have c95050 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95051 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1550_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_18
  simp only [← c95046, ← c95047, ← c95048] at e_1550_18
  have e_1550_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_19
  simp only [← c95049, ← c95050, k2, ← c95051] at e_1550_19
  have hr := e_1550_19
  simp only [e_1550_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15075)) := by
  have c95052 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95053 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95054 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95055 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95056 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95057 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_0
  simp only [← c95052, ← c95053, ← c95054] at e_1551_0
  have e_1551_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_1
  simp only [← c95055, ← c95056, k2, ← c95057] at e_1551_1
  have hr := e_1551_1
  simp only [e_1551_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15076)) := by
  have c95058 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95059 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95060 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95061 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95062 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95063 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_2
  simp only [← c95058, ← c95059, ← c95060] at e_1551_2
  have e_1551_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_3
  simp only [← c95061, ← c95062, k2, ← c95063] at e_1551_3
  have hr := e_1551_3
  simp only [e_1551_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15077)) := by
  have c95064 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95065 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95066 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95067 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95068 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95069 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_4
  simp only [← c95064, ← c95065, ← c95066] at e_1551_4
  have e_1551_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_5
  simp only [← c95067, ← c95068, k2, ← c95069] at e_1551_5
  have hr := e_1551_5
  simp only [e_1551_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15078)) := by
  have c95070 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95071 := (publicBatchWrapper89_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95072 := (publicBatchWrapper89_copies2971 a h).1
  have c95073 := (publicBatchWrapper89_copies2971 a h).2.1
  have c95074 := (publicBatchWrapper89_copies2971 a h).2.2.1
  have c95075 := (publicBatchWrapper89_copies2971 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_6
  simp only [← c95070, ← c95071, ← c95072] at e_1551_6
  have e_1551_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_7
  simp only [← c95073, ← c95074, k2, ← c95075] at e_1551_7
  have hr := e_1551_7
  simp only [e_1551_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15079)) := by
  have c95076 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.1
  have c95077 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.1
  have c95078 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.1
  have c95079 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.1
  have c95080 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.1
  have c95081 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_8
  simp only [← c95076, ← c95077, ← c95078] at e_1551_8
  have e_1551_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_9
  simp only [← c95079, ← c95080, k2, ← c95081] at e_1551_9
  have hr := e_1551_9
  simp only [e_1551_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15080)) := by
  have c95082 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95083 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95084 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95085 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95086 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95087 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_10
  simp only [← c95082, ← c95083, ← c95084] at e_1551_10
  have e_1551_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_11
  simp only [← c95085, ← c95086, k2, ← c95087] at e_1551_11
  have hr := e_1551_11
  simp only [e_1551_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15081)) := by
  have c95088 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95089 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95090 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95091 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95092 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95093 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_12
  simp only [← c95088, ← c95089, ← c95090] at e_1551_12
  have e_1551_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_13
  simp only [← c95091, ← c95092, k2, ← c95093] at e_1551_13
  have hr := e_1551_13
  simp only [e_1551_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15082)) := by
  have c95094 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95095 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95096 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95097 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95098 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95099 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_14
  simp only [← c95094, ← c95095, ← c95096] at e_1551_14
  have e_1551_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_15
  simp only [← c95097, ← c95098, k2, ← c95099] at e_1551_15
  have hr := e_1551_15
  simp only [e_1551_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15083)) := by
  have c95100 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95101 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95102 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95103 := (publicBatchWrapper89_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95104 := (publicBatchWrapper89_copies2972 a h).1
  have c95105 := (publicBatchWrapper89_copies2972 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_16
  simp only [← c95100, ← c95101, ← c95102] at e_1551_16
  have e_1551_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_17
  simp only [← c95103, ← c95104, k2, ← c95105] at e_1551_17
  have hr := e_1551_17
  simp only [e_1551_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1551 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15084)) := by
  have c95106 := (publicBatchWrapper89_copies2972 a h).2.2.1
  have c95107 := (publicBatchWrapper89_copies2972 a h).2.2.2.1
  have c95108 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.1
  have c95109 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.1
  have c95110 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.1
  have c95111 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1551_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_18
  simp only [← c95106, ← c95107, ← c95108] at e_1551_18
  have e_1551_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_19
  simp only [← c95109, ← c95110, k2, ← c95111] at e_1551_19
  have hr := e_1551_19
  simp only [e_1551_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15085)) := by
  have c95112 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.1
  have c95113 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.1
  have c95114 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95115 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95116 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95117 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_0
  simp only [← c95112, ← c95113, ← c95114] at e_1552_0
  have e_1552_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_1
  simp only [← c95115, ← c95116, k2, ← c95117] at e_1552_1
  have hr := e_1552_1
  simp only [e_1552_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15086)) := by
  have c95118 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95119 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95120 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95121 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95122 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95123 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_2
  simp only [← c95118, ← c95119, ← c95120] at e_1552_2
  have e_1552_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_3
  simp only [← c95121, ← c95122, k2, ← c95123] at e_1552_3
  have hr := e_1552_3
  simp only [e_1552_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15087)) := by
  have c95124 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95125 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95126 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95127 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95128 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95129 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_4
  simp only [← c95124, ← c95125, ← c95126] at e_1552_4
  have e_1552_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_5
  simp only [← c95127, ← c95128, k2, ← c95129] at e_1552_5
  have hr := e_1552_5
  simp only [e_1552_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15088)) := by
  have c95130 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95131 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95132 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95133 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95134 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95135 := (publicBatchWrapper89_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_6
  simp only [← c95130, ← c95131, ← c95132] at e_1552_6
  have e_1552_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_7
  simp only [← c95133, ← c95134, k2, ← c95135] at e_1552_7
  have hr := e_1552_7
  simp only [e_1552_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15089)) := by
  have c95136 := (publicBatchWrapper89_copies2973 a h).1
  have c95137 := (publicBatchWrapper89_copies2973 a h).2.1
  have c95138 := (publicBatchWrapper89_copies2973 a h).2.2.1
  have c95139 := (publicBatchWrapper89_copies2973 a h).2.2.2.1
  have c95140 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.1
  have c95141 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_8
  simp only [← c95136, ← c95137, ← c95138] at e_1552_8
  have e_1552_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_9
  simp only [← c95139, ← c95140, k2, ← c95141] at e_1552_9
  have hr := e_1552_9
  simp only [e_1552_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15090)) := by
  have c95142 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.1
  have c95143 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.1
  have c95144 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.1
  have c95145 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.1
  have c95146 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95147 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_10
  simp only [← c95142, ← c95143, ← c95144] at e_1552_10
  have e_1552_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_11
  simp only [← c95145, ← c95146, k2, ← c95147] at e_1552_11
  have hr := e_1552_11
  simp only [e_1552_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15091)) := by
  have c95148 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95149 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95150 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95151 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95152 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95153 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_12
  simp only [← c95148, ← c95149, ← c95150] at e_1552_12
  have e_1552_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_13
  simp only [← c95151, ← c95152, k2, ← c95153] at e_1552_13
  have hr := e_1552_13
  simp only [e_1552_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15092)) := by
  have c95154 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95155 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95156 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95157 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95158 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95159 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_14
  simp only [← c95154, ← c95155, ← c95156] at e_1552_14
  have e_1552_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_15
  simp only [← c95157, ← c95158, k2, ← c95159] at e_1552_15
  have hr := e_1552_15
  simp only [e_1552_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15093)) := by
  have c95160 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95161 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95162 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95163 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95164 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95165 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_16
  simp only [← c95160, ← c95161, ← c95162] at e_1552_16
  have e_1552_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_17
  simp only [← c95163, ← c95164, k2, ← c95165] at e_1552_17
  have hr := e_1552_17
  simp only [e_1552_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1552 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15094)) := by
  have c95166 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95167 := (publicBatchWrapper89_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95168 := (publicBatchWrapper89_copies2974 a h).1
  have c95169 := (publicBatchWrapper89_copies2974 a h).2.1
  have c95170 := (publicBatchWrapper89_copies2974 a h).2.2.1
  have c95171 := (publicBatchWrapper89_copies2974 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1552_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_18
  simp only [← c95166, ← c95167, ← c95168] at e_1552_18
  have e_1552_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_19
  simp only [← c95169, ← c95170, k2, ← c95171] at e_1552_19
  have hr := e_1552_19
  simp only [e_1552_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15095)) := by
  have c95172 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.1
  have c95173 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.1
  have c95174 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.1
  have c95175 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.1
  have c95176 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.1
  have c95177 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_0
  simp only [← c95172, ← c95173, ← c95174] at e_1553_0
  have e_1553_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_1
  simp only [← c95175, ← c95176, k2, ← c95177] at e_1553_1
  have hr := e_1553_1
  simp only [e_1553_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15096)) := by
  have c95178 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95179 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95180 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95181 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95182 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95183 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_2
  simp only [← c95178, ← c95179, ← c95180] at e_1553_2
  have e_1553_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_3
  simp only [← c95181, ← c95182, k2, ← c95183] at e_1553_3
  have hr := e_1553_3
  simp only [e_1553_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15097)) := by
  have c95184 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95185 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95186 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95187 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95188 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95189 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_4
  simp only [← c95184, ← c95185, ← c95186] at e_1553_4
  have e_1553_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_5
  simp only [← c95187, ← c95188, k2, ← c95189] at e_1553_5
  have hr := e_1553_5
  simp only [e_1553_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15098)) := by
  have c95190 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95191 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95192 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95193 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95194 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95195 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_6
  simp only [← c95190, ← c95191, ← c95192] at e_1553_6
  have e_1553_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_7
  simp only [← c95193, ← c95194, k2, ← c95195] at e_1553_7
  have hr := e_1553_7
  simp only [e_1553_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15099)) := by
  have c95196 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95197 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95198 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95199 := (publicBatchWrapper89_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95200 := (publicBatchWrapper89_copies2975 a h).1
  have c95201 := (publicBatchWrapper89_copies2975 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_8
  simp only [← c95196, ← c95197, ← c95198] at e_1553_8
  have e_1553_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_9
  simp only [← c95199, ← c95200, k2, ← c95201] at e_1553_9
  have hr := e_1553_9
  simp only [e_1553_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15100)) := by
  have c95202 := (publicBatchWrapper89_copies2975 a h).2.2.1
  have c95203 := (publicBatchWrapper89_copies2975 a h).2.2.2.1
  have c95204 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.1
  have c95205 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.1
  have c95206 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.1
  have c95207 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_10
  simp only [← c95202, ← c95203, ← c95204] at e_1553_10
  have e_1553_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_11
  simp only [← c95205, ← c95206, k2, ← c95207] at e_1553_11
  have hr := e_1553_11
  simp only [e_1553_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15101)) := by
  have c95208 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.1
  have c95209 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.1
  have c95210 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95211 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95212 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95213 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_12
  simp only [← c95208, ← c95209, ← c95210] at e_1553_12
  have e_1553_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_13
  simp only [← c95211, ← c95212, k2, ← c95213] at e_1553_13
  have hr := e_1553_13
  simp only [e_1553_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15102)) := by
  have c95214 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95215 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95216 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95217 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95218 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95219 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_14
  simp only [← c95214, ← c95215, ← c95216] at e_1553_14
  have e_1553_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_15
  simp only [← c95217, ← c95218, k2, ← c95219] at e_1553_15
  have hr := e_1553_15
  simp only [e_1553_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15103)) := by
  have c95220 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95221 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95222 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95223 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95224 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95225 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_16
  simp only [← c95220, ← c95221, ← c95222] at e_1553_16
  have e_1553_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_17
  simp only [← c95223, ← c95224, k2, ← c95225] at e_1553_17
  have hr := e_1553_17
  simp only [e_1553_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1553 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15104)) := by
  have c95226 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95227 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95228 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95229 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95230 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95231 := (publicBatchWrapper89_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1553_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_18
  simp only [← c95226, ← c95227, ← c95228] at e_1553_18
  have e_1553_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_19
  simp only [← c95229, ← c95230, k2, ← c95231] at e_1553_19
  have hr := e_1553_19
  simp only [e_1553_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15105)) := by
  have c95232 := (publicBatchWrapper89_copies2976 a h).1
  have c95233 := (publicBatchWrapper89_copies2976 a h).2.1
  have c95234 := (publicBatchWrapper89_copies2976 a h).2.2.1
  have c95235 := (publicBatchWrapper89_copies2976 a h).2.2.2.1
  have c95236 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.1
  have c95237 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_0
  simp only [← c95232, ← c95233, ← c95234] at e_1554_0
  have e_1554_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_1
  simp only [← c95235, ← c95236, k2, ← c95237] at e_1554_1
  have hr := e_1554_1
  simp only [e_1554_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15106)) := by
  have c95238 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.1
  have c95239 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.1
  have c95240 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.1
  have c95241 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.1
  have c95242 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95243 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_2
  simp only [← c95238, ← c95239, ← c95240] at e_1554_2
  have e_1554_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_3
  simp only [← c95241, ← c95242, k2, ← c95243] at e_1554_3
  have hr := e_1554_3
  simp only [e_1554_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15107)) := by
  have c95244 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95245 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95246 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95247 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95248 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95249 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_4
  simp only [← c95244, ← c95245, ← c95246] at e_1554_4
  have e_1554_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_5
  simp only [← c95247, ← c95248, k2, ← c95249] at e_1554_5
  have hr := e_1554_5
  simp only [e_1554_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15108)) := by
  have c95250 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95251 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95252 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95253 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95254 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95255 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_6
  simp only [← c95250, ← c95251, ← c95252] at e_1554_6
  have e_1554_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_7
  simp only [← c95253, ← c95254, k2, ← c95255] at e_1554_7
  have hr := e_1554_7
  simp only [e_1554_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15109)) := by
  have c95256 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95257 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95258 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95259 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95260 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95261 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_8
  simp only [← c95256, ← c95257, ← c95258] at e_1554_8
  have e_1554_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_9
  simp only [← c95259, ← c95260, k2, ← c95261] at e_1554_9
  have hr := e_1554_9
  simp only [e_1554_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15110)) := by
  have c95262 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95263 := (publicBatchWrapper89_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95264 := (publicBatchWrapper89_copies2977 a h).1
  have c95265 := (publicBatchWrapper89_copies2977 a h).2.1
  have c95266 := (publicBatchWrapper89_copies2977 a h).2.2.1
  have c95267 := (publicBatchWrapper89_copies2977 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_10
  simp only [← c95262, ← c95263, ← c95264] at e_1554_10
  have e_1554_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_11
  simp only [← c95265, ← c95266, k2, ← c95267] at e_1554_11
  have hr := e_1554_11
  simp only [e_1554_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15111)) := by
  have c95268 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.1
  have c95269 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.1
  have c95270 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.1
  have c95271 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.1
  have c95272 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.1
  have c95273 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_12
  simp only [← c95268, ← c95269, ← c95270] at e_1554_12
  have e_1554_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_13
  simp only [← c95271, ← c95272, k2, ← c95273] at e_1554_13
  have hr := e_1554_13
  simp only [e_1554_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15112)) := by
  have c95274 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95275 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95276 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95277 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95278 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95279 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_14
  simp only [← c95274, ← c95275, ← c95276] at e_1554_14
  have e_1554_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_15
  simp only [← c95277, ← c95278, k2, ← c95279] at e_1554_15
  have hr := e_1554_15
  simp only [e_1554_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15113)) := by
  have c95280 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95281 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95282 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95283 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95284 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95285 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_16
  simp only [← c95280, ← c95281, ← c95282] at e_1554_16
  have e_1554_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_17
  simp only [← c95283, ← c95284, k2, ← c95285] at e_1554_17
  have hr := e_1554_17
  simp only [e_1554_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1554 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15114)) := by
  have c95286 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95287 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95288 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95289 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95290 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95291 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1554_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_18
  simp only [← c95286, ← c95287, ← c95288] at e_1554_18
  have e_1554_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_19
  simp only [← c95289, ← c95290, k2, ← c95291] at e_1554_19
  have hr := e_1554_19
  simp only [e_1554_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15115)) := by
  have c95292 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95293 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95294 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95295 := (publicBatchWrapper89_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95296 := (publicBatchWrapper89_copies2978 a h).1
  have c95297 := (publicBatchWrapper89_copies2978 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_0
  simp only [← c95292, ← c95293, ← c95294] at e_1555_0
  have e_1555_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_1
  simp only [← c95295, ← c95296, k2, ← c95297] at e_1555_1
  have hr := e_1555_1
  simp only [e_1555_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15116)) := by
  have c95298 := (publicBatchWrapper89_copies2978 a h).2.2.1
  have c95299 := (publicBatchWrapper89_copies2978 a h).2.2.2.1
  have c95300 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.1
  have c95301 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.1
  have c95302 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.1
  have c95303 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_2
  simp only [← c95298, ← c95299, ← c95300] at e_1555_2
  have e_1555_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_3
  simp only [← c95301, ← c95302, k2, ← c95303] at e_1555_3
  have hr := e_1555_3
  simp only [e_1555_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15117)) := by
  have c95304 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.1
  have c95305 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.1
  have c95306 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95307 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95308 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95309 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_4
  simp only [← c95304, ← c95305, ← c95306] at e_1555_4
  have e_1555_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_5
  simp only [← c95307, ← c95308, k2, ← c95309] at e_1555_5
  have hr := e_1555_5
  simp only [e_1555_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15118)) := by
  have c95310 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95311 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95312 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95313 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95314 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95315 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_6
  simp only [← c95310, ← c95311, ← c95312] at e_1555_6
  have e_1555_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_7
  simp only [← c95313, ← c95314, k2, ← c95315] at e_1555_7
  have hr := e_1555_7
  simp only [e_1555_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15119)) := by
  have c95316 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95317 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95318 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95319 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95320 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95321 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_8
  simp only [← c95316, ← c95317, ← c95318] at e_1555_8
  have e_1555_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_9
  simp only [← c95319, ← c95320, k2, ← c95321] at e_1555_9
  have hr := e_1555_9
  simp only [e_1555_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15120)) := by
  have c95322 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95323 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95324 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95325 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95326 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95327 := (publicBatchWrapper89_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_10
  simp only [← c95322, ← c95323, ← c95324] at e_1555_10
  have e_1555_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_11
  simp only [← c95325, ← c95326, k2, ← c95327] at e_1555_11
  have hr := e_1555_11
  simp only [e_1555_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15121)) := by
  have c95328 := (publicBatchWrapper89_copies2979 a h).1
  have c95329 := (publicBatchWrapper89_copies2979 a h).2.1
  have c95330 := (publicBatchWrapper89_copies2979 a h).2.2.1
  have c95331 := (publicBatchWrapper89_copies2979 a h).2.2.2.1
  have c95332 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.1
  have c95333 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_12
  simp only [← c95328, ← c95329, ← c95330] at e_1555_12
  have e_1555_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_13
  simp only [← c95331, ← c95332, k2, ← c95333] at e_1555_13
  have hr := e_1555_13
  simp only [e_1555_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15122)) := by
  have c95334 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.1
  have c95335 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.1
  have c95336 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.1
  have c95337 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.1
  have c95338 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95339 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_14
  simp only [← c95334, ← c95335, ← c95336] at e_1555_14
  have e_1555_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_15
  simp only [← c95337, ← c95338, k2, ← c95339] at e_1555_15
  have hr := e_1555_15
  simp only [e_1555_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15123)) := by
  have c95340 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95341 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95342 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95343 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95344 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95345 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_16
  simp only [← c95340, ← c95341, ← c95342] at e_1555_16
  have e_1555_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_17
  simp only [← c95343, ← c95344, k2, ← c95345] at e_1555_17
  have hr := e_1555_17
  simp only [e_1555_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1555 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15124)) := by
  have c95346 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95347 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95348 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95349 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95350 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95351 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1555_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_18
  simp only [← c95346, ← c95347, ← c95348] at e_1555_18
  have e_1555_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_19
  simp only [← c95349, ← c95350, k2, ← c95351] at e_1555_19
  have hr := e_1555_19
  simp only [e_1555_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15125)) := by
  have c95352 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95353 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95354 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95355 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95356 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95357 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_0
  simp only [← c95352, ← c95353, ← c95354] at e_1556_0
  have e_1556_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_1
  simp only [← c95355, ← c95356, k2, ← c95357] at e_1556_1
  have hr := e_1556_1
  simp only [e_1556_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15126)) := by
  have c95358 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95359 := (publicBatchWrapper89_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95360 := (publicBatchWrapper89_copies2980 a h).1
  have c95361 := (publicBatchWrapper89_copies2980 a h).2.1
  have c95362 := (publicBatchWrapper89_copies2980 a h).2.2.1
  have c95363 := (publicBatchWrapper89_copies2980 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_2
  simp only [← c95358, ← c95359, ← c95360] at e_1556_2
  have e_1556_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_3
  simp only [← c95361, ← c95362, k2, ← c95363] at e_1556_3
  have hr := e_1556_3
  simp only [e_1556_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15127)) := by
  have c95364 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.1
  have c95365 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.1
  have c95366 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.1
  have c95367 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.1
  have c95368 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.1
  have c95369 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_4
  simp only [← c95364, ← c95365, ← c95366] at e_1556_4
  have e_1556_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_5
  simp only [← c95367, ← c95368, k2, ← c95369] at e_1556_5
  have hr := e_1556_5
  simp only [e_1556_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15128)) := by
  have c95370 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95371 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95372 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95373 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95374 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95375 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_6
  simp only [← c95370, ← c95371, ← c95372] at e_1556_6
  have e_1556_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_7
  simp only [← c95373, ← c95374, k2, ← c95375] at e_1556_7
  have hr := e_1556_7
  simp only [e_1556_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15129)) := by
  have c95376 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95377 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95378 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95379 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95380 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95381 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_8
  simp only [← c95376, ← c95377, ← c95378] at e_1556_8
  have e_1556_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_9
  simp only [← c95379, ← c95380, k2, ← c95381] at e_1556_9
  have hr := e_1556_9
  simp only [e_1556_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15130)) := by
  have c95382 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95383 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95384 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95385 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95386 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95387 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_10
  simp only [← c95382, ← c95383, ← c95384] at e_1556_10
  have e_1556_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_11
  simp only [← c95385, ← c95386, k2, ← c95387] at e_1556_11
  have hr := e_1556_11
  simp only [e_1556_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15131)) := by
  have c95388 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95389 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95390 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95391 := (publicBatchWrapper89_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95392 := (publicBatchWrapper89_copies2981 a h).1
  have c95393 := (publicBatchWrapper89_copies2981 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_12
  simp only [← c95388, ← c95389, ← c95390] at e_1556_12
  have e_1556_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_13
  simp only [← c95391, ← c95392, k2, ← c95393] at e_1556_13
  have hr := e_1556_13
  simp only [e_1556_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15132)) := by
  have c95394 := (publicBatchWrapper89_copies2981 a h).2.2.1
  have c95395 := (publicBatchWrapper89_copies2981 a h).2.2.2.1
  have c95396 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.1
  have c95397 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.1
  have c95398 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.1
  have c95399 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_14
  simp only [← c95394, ← c95395, ← c95396] at e_1556_14
  have e_1556_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_15
  simp only [← c95397, ← c95398, k2, ← c95399] at e_1556_15
  have hr := e_1556_15
  simp only [e_1556_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15133)) := by
  have c95400 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.1
  have c95401 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.1
  have c95402 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95403 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95404 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95405 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_16
  simp only [← c95400, ← c95401, ← c95402] at e_1556_16
  have e_1556_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_17
  simp only [← c95403, ← c95404, k2, ← c95405] at e_1556_17
  have hr := e_1556_17
  simp only [e_1556_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1556 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15134)) := by
  have c95406 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95407 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95408 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95409 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95410 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95411 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1556_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_18
  simp only [← c95406, ← c95407, ← c95408] at e_1556_18
  have e_1556_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_19
  simp only [← c95409, ← c95410, k2, ← c95411] at e_1556_19
  have hr := e_1556_19
  simp only [e_1556_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15135)) := by
  have c95412 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95413 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95414 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95415 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95416 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95417 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_0
  simp only [← c95412, ← c95413, ← c95414] at e_1557_0
  have e_1557_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_1
  simp only [← c95415, ← c95416, k2, ← c95417] at e_1557_1
  have hr := e_1557_1
  simp only [e_1557_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15136)) := by
  have c95418 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95419 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95420 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95421 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95422 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95423 := (publicBatchWrapper89_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_2
  simp only [← c95418, ← c95419, ← c95420] at e_1557_2
  have e_1557_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_3
  simp only [← c95421, ← c95422, k2, ← c95423] at e_1557_3
  have hr := e_1557_3
  simp only [e_1557_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15137)) := by
  have c95424 := (publicBatchWrapper89_copies2982 a h).1
  have c95425 := (publicBatchWrapper89_copies2982 a h).2.1
  have c95426 := (publicBatchWrapper89_copies2982 a h).2.2.1
  have c95427 := (publicBatchWrapper89_copies2982 a h).2.2.2.1
  have c95428 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.1
  have c95429 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_4
  simp only [← c95424, ← c95425, ← c95426] at e_1557_4
  have e_1557_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_5
  simp only [← c95427, ← c95428, k2, ← c95429] at e_1557_5
  have hr := e_1557_5
  simp only [e_1557_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15138)) := by
  have c95430 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.1
  have c95431 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.1
  have c95432 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.1
  have c95433 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.1
  have c95434 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95435 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_6
  simp only [← c95430, ← c95431, ← c95432] at e_1557_6
  have e_1557_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_7
  simp only [← c95433, ← c95434, k2, ← c95435] at e_1557_7
  have hr := e_1557_7
  simp only [e_1557_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15139)) := by
  have c95436 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95437 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95438 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95439 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95440 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95441 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_8
  simp only [← c95436, ← c95437, ← c95438] at e_1557_8
  have e_1557_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_9
  simp only [← c95439, ← c95440, k2, ← c95441] at e_1557_9
  have hr := e_1557_9
  simp only [e_1557_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15140)) := by
  have c95442 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95443 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95444 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95445 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95446 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95447 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_10
  simp only [← c95442, ← c95443, ← c95444] at e_1557_10
  have e_1557_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_11
  simp only [← c95445, ← c95446, k2, ← c95447] at e_1557_11
  have hr := e_1557_11
  simp only [e_1557_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15141)) := by
  have c95448 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95449 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95450 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95451 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95452 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95453 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_12
  simp only [← c95448, ← c95449, ← c95450] at e_1557_12
  have e_1557_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_13
  simp only [← c95451, ← c95452, k2, ← c95453] at e_1557_13
  have hr := e_1557_13
  simp only [e_1557_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15142)) := by
  have c95454 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95455 := (publicBatchWrapper89_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95456 := (publicBatchWrapper89_copies2983 a h).1
  have c95457 := (publicBatchWrapper89_copies2983 a h).2.1
  have c95458 := (publicBatchWrapper89_copies2983 a h).2.2.1
  have c95459 := (publicBatchWrapper89_copies2983 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_14
  simp only [← c95454, ← c95455, ← c95456] at e_1557_14
  have e_1557_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_15
  simp only [← c95457, ← c95458, k2, ← c95459] at e_1557_15
  have hr := e_1557_15
  simp only [e_1557_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15143)) := by
  have c95460 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.1
  have c95461 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.1
  have c95462 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.1
  have c95463 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.1
  have c95464 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.1
  have c95465 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_16
  simp only [← c95460, ← c95461, ← c95462] at e_1557_16
  have e_1557_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_17
  simp only [← c95463, ← c95464, k2, ← c95465] at e_1557_17
  have hr := e_1557_17
  simp only [e_1557_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1557 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15144)) := by
  have c95466 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95467 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95468 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95469 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95470 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95471 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1557_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_18
  simp only [← c95466, ← c95467, ← c95468] at e_1557_18
  have e_1557_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_19
  simp only [← c95469, ← c95470, k2, ← c95471] at e_1557_19
  have hr := e_1557_19
  simp only [e_1557_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15145)) := by
  have c95472 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95473 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95474 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95475 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95476 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95477 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_0
  simp only [← c95472, ← c95473, ← c95474] at e_1558_0
  have e_1558_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_1
  simp only [← c95475, ← c95476, k2, ← c95477] at e_1558_1
  have hr := e_1558_1
  simp only [e_1558_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15146)) := by
  have c95478 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95479 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95480 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95481 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95482 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95483 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_2
  simp only [← c95478, ← c95479, ← c95480] at e_1558_2
  have e_1558_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_3
  simp only [← c95481, ← c95482, k2, ← c95483] at e_1558_3
  have hr := e_1558_3
  simp only [e_1558_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15147)) := by
  have c95484 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95485 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95486 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95487 := (publicBatchWrapper89_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95488 := (publicBatchWrapper89_copies2984 a h).1
  have c95489 := (publicBatchWrapper89_copies2984 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_4
  simp only [← c95484, ← c95485, ← c95486] at e_1558_4
  have e_1558_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_5
  simp only [← c95487, ← c95488, k2, ← c95489] at e_1558_5
  have hr := e_1558_5
  simp only [e_1558_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15148)) := by
  have c95490 := (publicBatchWrapper89_copies2984 a h).2.2.1
  have c95491 := (publicBatchWrapper89_copies2984 a h).2.2.2.1
  have c95492 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.1
  have c95493 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.1
  have c95494 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.1
  have c95495 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_6
  simp only [← c95490, ← c95491, ← c95492] at e_1558_6
  have e_1558_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_7
  simp only [← c95493, ← c95494, k2, ← c95495] at e_1558_7
  have hr := e_1558_7
  simp only [e_1558_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15149)) := by
  have c95496 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.1
  have c95497 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.1
  have c95498 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95499 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95500 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95501 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_8
  simp only [← c95496, ← c95497, ← c95498] at e_1558_8
  have e_1558_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_9
  simp only [← c95499, ← c95500, k2, ← c95501] at e_1558_9
  have hr := e_1558_9
  simp only [e_1558_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15150)) := by
  have c95502 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95503 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95504 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95505 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95506 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95507 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_10
  simp only [← c95502, ← c95503, ← c95504] at e_1558_10
  have e_1558_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_11
  simp only [← c95505, ← c95506, k2, ← c95507] at e_1558_11
  have hr := e_1558_11
  simp only [e_1558_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15151)) := by
  have c95508 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95509 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95510 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95511 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95512 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95513 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_12
  simp only [← c95508, ← c95509, ← c95510] at e_1558_12
  have e_1558_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_13
  simp only [← c95511, ← c95512, k2, ← c95513] at e_1558_13
  have hr := e_1558_13
  simp only [e_1558_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15152)) := by
  have c95514 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95515 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95516 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95517 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95518 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95519 := (publicBatchWrapper89_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_14
  simp only [← c95514, ← c95515, ← c95516] at e_1558_14
  have e_1558_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_15
  simp only [← c95517, ← c95518, k2, ← c95519] at e_1558_15
  have hr := e_1558_15
  simp only [e_1558_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15153)) := by
  have c95520 := (publicBatchWrapper89_copies2985 a h).1
  have c95521 := (publicBatchWrapper89_copies2985 a h).2.1
  have c95522 := (publicBatchWrapper89_copies2985 a h).2.2.1
  have c95523 := (publicBatchWrapper89_copies2985 a h).2.2.2.1
  have c95524 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.1
  have c95525 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_16
  simp only [← c95520, ← c95521, ← c95522] at e_1558_16
  have e_1558_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_17
  simp only [← c95523, ← c95524, k2, ← c95525] at e_1558_17
  have hr := e_1558_17
  simp only [e_1558_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1558 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15154)) := by
  have c95526 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.1
  have c95527 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.1
  have c95528 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.1
  have c95529 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.1
  have c95530 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95531 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1558_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_18
  simp only [← c95526, ← c95527, ← c95528] at e_1558_18
  have e_1558_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_19
  simp only [← c95529, ← c95530, k2, ← c95531] at e_1558_19
  have hr := e_1558_19
  simp only [e_1558_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15155)) := by
  have c95532 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95533 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95534 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95535 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95536 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95537 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_0
  simp only [← c95532, ← c95533, ← c95534] at e_1559_0
  have e_1559_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_1
  simp only [← c95535, ← c95536, k2, ← c95537] at e_1559_1
  have hr := e_1559_1
  simp only [e_1559_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15156)) := by
  have c95538 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95539 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95540 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95541 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95542 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95543 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_2
  simp only [← c95538, ← c95539, ← c95540] at e_1559_2
  have e_1559_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_3
  simp only [← c95541, ← c95542, k2, ← c95543] at e_1559_3
  have hr := e_1559_3
  simp only [e_1559_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
