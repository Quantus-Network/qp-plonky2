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

theorem publicBatchWrapper89_f10743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1116 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9647)) := by
  have c68964 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.1
  have c68965 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.1
  have c68966 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.1
  have c68967 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.1
  have c68968 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.1
  have c68969 := (publicBatchWrapper89_copies2155 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1116_4 := arithEq_of_rows h (row := 1116) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_4
  simp only [← c68964, ← c68965, ← c68966] at e_1116_4
  have e_1116_5 := arithEq_of_rows h (row := 1116) (i := 5) rfl (by decide)
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
  have e_1116_6 := arithEq_of_rows h (row := 1116) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_6
  simp only [← c68970, ← c68971, ← c68972] at e_1116_6
  have e_1116_7 := arithEq_of_rows h (row := 1116) (i := 7) rfl (by decide)
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
  have e_1116_8 := arithEq_of_rows h (row := 1116) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_8
  simp only [← c68976, ← c68977, ← c68978] at e_1116_8
  have e_1116_9 := arithEq_of_rows h (row := 1116) (i := 9) rfl (by decide)
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
  have e_1116_10 := arithEq_of_rows h (row := 1116) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_10
  simp only [← c68982, ← c68983, ← c68984] at e_1116_10
  have e_1116_11 := arithEq_of_rows h (row := 1116) (i := 11) rfl (by decide)
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
  have e_1116_12 := arithEq_of_rows h (row := 1116) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_12
  simp only [← c68988, ← c68989, ← c68990] at e_1116_12
  have e_1116_13 := arithEq_of_rows h (row := 1116) (i := 13) rfl (by decide)
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
  have e_1116_14 := arithEq_of_rows h (row := 1116) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_14
  simp only [← c68994, ← c68995, ← c68996] at e_1116_14
  have e_1116_15 := arithEq_of_rows h (row := 1116) (i := 15) rfl (by decide)
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
  have e_1116_16 := arithEq_of_rows h (row := 1116) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_16
  simp only [← c69000, ← c69001, ← c69002] at e_1116_16
  have e_1116_17 := arithEq_of_rows h (row := 1116) (i := 17) rfl (by decide)
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
  have e_1116_18 := arithEq_of_rows h (row := 1116) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1116_18
  simp only [← c69006, ← c69007, ← c69008] at e_1116_18
  have e_1116_19 := arithEq_of_rows h (row := 1116) (i := 19) rfl (by decide)
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
  have e_1117_0 := arithEq_of_rows h (row := 1117) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_0
  simp only [← c69012, ← c69013, ← c69014] at e_1117_0
  have e_1117_1 := arithEq_of_rows h (row := 1117) (i := 1) rfl (by decide)
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
  have e_1117_2 := arithEq_of_rows h (row := 1117) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_2
  simp only [← c69018, ← c69019, ← c69020] at e_1117_2
  have e_1117_3 := arithEq_of_rows h (row := 1117) (i := 3) rfl (by decide)
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
  have e_1117_4 := arithEq_of_rows h (row := 1117) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_4
  simp only [← c69024, ← c69025, ← c69026] at e_1117_4
  have e_1117_5 := arithEq_of_rows h (row := 1117) (i := 5) rfl (by decide)
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
  have e_1117_6 := arithEq_of_rows h (row := 1117) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_6
  simp only [← c69030, ← c69031, ← c69032] at e_1117_6
  have e_1117_7 := arithEq_of_rows h (row := 1117) (i := 7) rfl (by decide)
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
  have e_1117_8 := arithEq_of_rows h (row := 1117) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_8
  simp only [← c69036, ← c69037, ← c69038] at e_1117_8
  have e_1117_9 := arithEq_of_rows h (row := 1117) (i := 9) rfl (by decide)
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
  have e_1117_10 := arithEq_of_rows h (row := 1117) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_10
  simp only [← c69042, ← c69043, ← c69044] at e_1117_10
  have e_1117_11 := arithEq_of_rows h (row := 1117) (i := 11) rfl (by decide)
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
  have e_1117_12 := arithEq_of_rows h (row := 1117) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_12
  simp only [← c69048, ← c69049, ← c69050] at e_1117_12
  have e_1117_13 := arithEq_of_rows h (row := 1117) (i := 13) rfl (by decide)
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
  have e_1117_14 := arithEq_of_rows h (row := 1117) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_14
  simp only [← c69054, ← c69055, ← c69056] at e_1117_14
  have e_1117_15 := arithEq_of_rows h (row := 1117) (i := 15) rfl (by decide)
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
  have e_1117_16 := arithEq_of_rows h (row := 1117) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_16
  simp only [← c69060, ← c69061, ← c69062] at e_1117_16
  have e_1117_17 := arithEq_of_rows h (row := 1117) (i := 17) rfl (by decide)
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
  have e_1117_18 := arithEq_of_rows h (row := 1117) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1117_18
  simp only [← c69066, ← c69067, ← c69068] at e_1117_18
  have e_1117_19 := arithEq_of_rows h (row := 1117) (i := 19) rfl (by decide)
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
  have e_1118_0 := arithEq_of_rows h (row := 1118) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_0
  simp only [← c69072, ← c69073, ← c69074] at e_1118_0
  have e_1118_1 := arithEq_of_rows h (row := 1118) (i := 1) rfl (by decide)
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
  have e_1118_2 := arithEq_of_rows h (row := 1118) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_2
  simp only [← c69078, ← c69079, ← c69080] at e_1118_2
  have e_1118_3 := arithEq_of_rows h (row := 1118) (i := 3) rfl (by decide)
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
  have e_1118_4 := arithEq_of_rows h (row := 1118) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_4
  simp only [← c69084, ← c69085, ← c69086] at e_1118_4
  have e_1118_5 := arithEq_of_rows h (row := 1118) (i := 5) rfl (by decide)
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
  have e_1118_6 := arithEq_of_rows h (row := 1118) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_6
  simp only [← c69090, ← c69091, ← c69092] at e_1118_6
  have e_1118_7 := arithEq_of_rows h (row := 1118) (i := 7) rfl (by decide)
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
  have e_1118_8 := arithEq_of_rows h (row := 1118) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_8
  simp only [← c69096, ← c69097, ← c69098] at e_1118_8
  have e_1118_9 := arithEq_of_rows h (row := 1118) (i := 9) rfl (by decide)
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
  have e_1118_10 := arithEq_of_rows h (row := 1118) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_10
  simp only [← c69102, ← c69103, ← c69104] at e_1118_10
  have e_1118_11 := arithEq_of_rows h (row := 1118) (i := 11) rfl (by decide)
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
  have e_1118_12 := arithEq_of_rows h (row := 1118) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_12
  simp only [← c69108, ← c69109, ← c69110] at e_1118_12
  have e_1118_13 := arithEq_of_rows h (row := 1118) (i := 13) rfl (by decide)
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
  have e_1118_14 := arithEq_of_rows h (row := 1118) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_14
  simp only [← c69114, ← c69115, ← c69116] at e_1118_14
  have e_1118_15 := arithEq_of_rows h (row := 1118) (i := 15) rfl (by decide)
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
  have e_1118_16 := arithEq_of_rows h (row := 1118) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_16
  simp only [← c69120, ← c69121, ← c69122] at e_1118_16
  have e_1118_17 := arithEq_of_rows h (row := 1118) (i := 17) rfl (by decide)
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
  have e_1118_18 := arithEq_of_rows h (row := 1118) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1118_18
  simp only [← c69126, ← c69127, ← c69128] at e_1118_18
  have e_1118_19 := arithEq_of_rows h (row := 1118) (i := 19) rfl (by decide)
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
  have e_1119_0 := arithEq_of_rows h (row := 1119) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_0
  simp only [← c69132, ← c69133, ← c69134] at e_1119_0
  have e_1119_1 := arithEq_of_rows h (row := 1119) (i := 1) rfl (by decide)
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
  have e_1119_2 := arithEq_of_rows h (row := 1119) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_2
  simp only [← c69138, ← c69139, ← c69140] at e_1119_2
  have e_1119_3 := arithEq_of_rows h (row := 1119) (i := 3) rfl (by decide)
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
  have e_1119_4 := arithEq_of_rows h (row := 1119) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_4
  simp only [← c69144, ← c69145, ← c69146] at e_1119_4
  have e_1119_5 := arithEq_of_rows h (row := 1119) (i := 5) rfl (by decide)
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
  have e_1119_6 := arithEq_of_rows h (row := 1119) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_6
  simp only [← c69150, ← c69151, ← c69152] at e_1119_6
  have e_1119_7 := arithEq_of_rows h (row := 1119) (i := 7) rfl (by decide)
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
  have e_1119_8 := arithEq_of_rows h (row := 1119) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_8
  simp only [← c69156, ← c69157, ← c69158] at e_1119_8
  have e_1119_9 := arithEq_of_rows h (row := 1119) (i := 9) rfl (by decide)
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
  have e_1119_10 := arithEq_of_rows h (row := 1119) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_10
  simp only [← c69162, ← c69163, ← c69164] at e_1119_10
  have e_1119_11 := arithEq_of_rows h (row := 1119) (i := 11) rfl (by decide)
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
  have e_1119_12 := arithEq_of_rows h (row := 1119) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_12
  simp only [← c69168, ← c69169, ← c69170] at e_1119_12
  have e_1119_13 := arithEq_of_rows h (row := 1119) (i := 13) rfl (by decide)
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
  have e_1119_14 := arithEq_of_rows h (row := 1119) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_14
  simp only [← c69174, ← c69175, ← c69176] at e_1119_14
  have e_1119_15 := arithEq_of_rows h (row := 1119) (i := 15) rfl (by decide)
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
  have e_1119_16 := arithEq_of_rows h (row := 1119) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_16
  simp only [← c69180, ← c69181, ← c69182] at e_1119_16
  have e_1119_17 := arithEq_of_rows h (row := 1119) (i := 17) rfl (by decide)
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
  have e_1119_18 := arithEq_of_rows h (row := 1119) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1119_18
  simp only [← c69186, ← c69187, ← c69188] at e_1119_18
  have e_1119_19 := arithEq_of_rows h (row := 1119) (i := 19) rfl (by decide)
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
  have e_1120_0 := arithEq_of_rows h (row := 1120) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_0
  simp only [← c69192, ← c69193, ← c69194] at e_1120_0
  have e_1120_1 := arithEq_of_rows h (row := 1120) (i := 1) rfl (by decide)
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
  have e_1120_2 := arithEq_of_rows h (row := 1120) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_2
  simp only [← c69198, ← c69199, ← c69200] at e_1120_2
  have e_1120_3 := arithEq_of_rows h (row := 1120) (i := 3) rfl (by decide)
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
  have e_1120_4 := arithEq_of_rows h (row := 1120) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_4
  simp only [← c69204, ← c69205, ← c69206] at e_1120_4
  have e_1120_5 := arithEq_of_rows h (row := 1120) (i := 5) rfl (by decide)
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
  have e_1120_6 := arithEq_of_rows h (row := 1120) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_6
  simp only [← c69210, ← c69211, ← c69212] at e_1120_6
  have e_1120_7 := arithEq_of_rows h (row := 1120) (i := 7) rfl (by decide)
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
  have e_1120_8 := arithEq_of_rows h (row := 1120) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_8
  simp only [← c69216, ← c69217, ← c69218] at e_1120_8
  have e_1120_9 := arithEq_of_rows h (row := 1120) (i := 9) rfl (by decide)
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
  have e_1120_10 := arithEq_of_rows h (row := 1120) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_10
  simp only [← c69222, ← c69223, ← c69224] at e_1120_10
  have e_1120_11 := arithEq_of_rows h (row := 1120) (i := 11) rfl (by decide)
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
  have e_1120_12 := arithEq_of_rows h (row := 1120) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_12
  simp only [← c69228, ← c69229, ← c69230] at e_1120_12
  have e_1120_13 := arithEq_of_rows h (row := 1120) (i := 13) rfl (by decide)
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
  have e_1120_14 := arithEq_of_rows h (row := 1120) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_14
  simp only [← c69234, ← c69235, ← c69236] at e_1120_14
  have e_1120_15 := arithEq_of_rows h (row := 1120) (i := 15) rfl (by decide)
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
  have e_1120_16 := arithEq_of_rows h (row := 1120) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_16
  simp only [← c69240, ← c69241, ← c69242] at e_1120_16
  have e_1120_17 := arithEq_of_rows h (row := 1120) (i := 17) rfl (by decide)
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
  have e_1120_18 := arithEq_of_rows h (row := 1120) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1120_18
  simp only [← c69246, ← c69247, ← c69248] at e_1120_18
  have e_1120_19 := arithEq_of_rows h (row := 1120) (i := 19) rfl (by decide)
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
  have e_1121_0 := arithEq_of_rows h (row := 1121) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_0
  simp only [← c69252, ← c69253, ← c69254] at e_1121_0
  have e_1121_1 := arithEq_of_rows h (row := 1121) (i := 1) rfl (by decide)
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
  have e_1121_2 := arithEq_of_rows h (row := 1121) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_2
  simp only [← c69258, ← c69259, ← c69260] at e_1121_2
  have e_1121_3 := arithEq_of_rows h (row := 1121) (i := 3) rfl (by decide)
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
  have e_1121_4 := arithEq_of_rows h (row := 1121) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_4
  simp only [← c69264, ← c69265, ← c69266] at e_1121_4
  have e_1121_5 := arithEq_of_rows h (row := 1121) (i := 5) rfl (by decide)
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
  have e_1121_6 := arithEq_of_rows h (row := 1121) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_6
  simp only [← c69270, ← c69271, ← c69272] at e_1121_6
  have e_1121_7 := arithEq_of_rows h (row := 1121) (i := 7) rfl (by decide)
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
  have e_1121_8 := arithEq_of_rows h (row := 1121) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_8
  simp only [← c69276, ← c69277, ← c69278] at e_1121_8
  have e_1121_9 := arithEq_of_rows h (row := 1121) (i := 9) rfl (by decide)
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
  have e_1121_10 := arithEq_of_rows h (row := 1121) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_10
  simp only [← c69282, ← c69283, ← c69284] at e_1121_10
  have e_1121_11 := arithEq_of_rows h (row := 1121) (i := 11) rfl (by decide)
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
  have e_1121_12 := arithEq_of_rows h (row := 1121) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_12
  simp only [← c69288, ← c69289, ← c69290] at e_1121_12
  have e_1121_13 := arithEq_of_rows h (row := 1121) (i := 13) rfl (by decide)
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
  have e_1121_14 := arithEq_of_rows h (row := 1121) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_14
  simp only [← c69294, ← c69295, ← c69296] at e_1121_14
  have e_1121_15 := arithEq_of_rows h (row := 1121) (i := 15) rfl (by decide)
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
  have e_1121_16 := arithEq_of_rows h (row := 1121) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_16
  simp only [← c69300, ← c69301, ← c69302] at e_1121_16
  have e_1121_17 := arithEq_of_rows h (row := 1121) (i := 17) rfl (by decide)
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
  have e_1121_18 := arithEq_of_rows h (row := 1121) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1121_18
  simp only [← c69306, ← c69307, ← c69308] at e_1121_18
  have e_1121_19 := arithEq_of_rows h (row := 1121) (i := 19) rfl (by decide)
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
  have e_1122_0 := arithEq_of_rows h (row := 1122) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_0
  simp only [← c69312, ← c69313, ← c69314] at e_1122_0
  have e_1122_1 := arithEq_of_rows h (row := 1122) (i := 1) rfl (by decide)
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
  have e_1122_2 := arithEq_of_rows h (row := 1122) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_2
  simp only [← c69318, ← c69319, ← c69320] at e_1122_2
  have e_1122_3 := arithEq_of_rows h (row := 1122) (i := 3) rfl (by decide)
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
  have e_1122_4 := arithEq_of_rows h (row := 1122) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_4
  simp only [← c69324, ← c69325, ← c69326] at e_1122_4
  have e_1122_5 := arithEq_of_rows h (row := 1122) (i := 5) rfl (by decide)
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
  have e_1122_6 := arithEq_of_rows h (row := 1122) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_6
  simp only [← c69330, ← c69331, ← c69332] at e_1122_6
  have e_1122_7 := arithEq_of_rows h (row := 1122) (i := 7) rfl (by decide)
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
  have e_1122_8 := arithEq_of_rows h (row := 1122) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_8
  simp only [← c69336, ← c69337, ← c69338] at e_1122_8
  have e_1122_9 := arithEq_of_rows h (row := 1122) (i := 9) rfl (by decide)
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
  have e_1122_10 := arithEq_of_rows h (row := 1122) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_10
  simp only [← c69342, ← c69343, ← c69344] at e_1122_10
  have e_1122_11 := arithEq_of_rows h (row := 1122) (i := 11) rfl (by decide)
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
  have e_1122_12 := arithEq_of_rows h (row := 1122) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_12
  simp only [← c69348, ← c69349, ← c69350] at e_1122_12
  have e_1122_13 := arithEq_of_rows h (row := 1122) (i := 13) rfl (by decide)
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
  have e_1122_14 := arithEq_of_rows h (row := 1122) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_14
  simp only [← c69354, ← c69355, ← c69356] at e_1122_14
  have e_1122_15 := arithEq_of_rows h (row := 1122) (i := 15) rfl (by decide)
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
  have e_1122_16 := arithEq_of_rows h (row := 1122) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_16
  simp only [← c69360, ← c69361, ← c69362] at e_1122_16
  have e_1122_17 := arithEq_of_rows h (row := 1122) (i := 17) rfl (by decide)
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
  have e_1122_18 := arithEq_of_rows h (row := 1122) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1122_18
  simp only [← c69366, ← c69367, ← c69368] at e_1122_18
  have e_1122_19 := arithEq_of_rows h (row := 1122) (i := 19) rfl (by decide)
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
  have e_1123_0 := arithEq_of_rows h (row := 1123) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_0
  simp only [← c69372, ← c69373, ← c69374] at e_1123_0
  have e_1123_1 := arithEq_of_rows h (row := 1123) (i := 1) rfl (by decide)
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
  have e_1123_2 := arithEq_of_rows h (row := 1123) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_2
  simp only [← c69378, ← c69379, ← c69380] at e_1123_2
  have e_1123_3 := arithEq_of_rows h (row := 1123) (i := 3) rfl (by decide)
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
  have e_1123_4 := arithEq_of_rows h (row := 1123) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_4
  simp only [← c69384, ← c69385, ← c69386] at e_1123_4
  have e_1123_5 := arithEq_of_rows h (row := 1123) (i := 5) rfl (by decide)
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
  have e_1123_6 := arithEq_of_rows h (row := 1123) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_6
  simp only [← c69390, ← c69391, ← c69392] at e_1123_6
  have e_1123_7 := arithEq_of_rows h (row := 1123) (i := 7) rfl (by decide)
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
  have e_1123_8 := arithEq_of_rows h (row := 1123) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_8
  simp only [← c69396, ← c69397, ← c69398] at e_1123_8
  have e_1123_9 := arithEq_of_rows h (row := 1123) (i := 9) rfl (by decide)
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
  have e_1123_10 := arithEq_of_rows h (row := 1123) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_10
  simp only [← c69402, ← c69403, ← c69404] at e_1123_10
  have e_1123_11 := arithEq_of_rows h (row := 1123) (i := 11) rfl (by decide)
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
  have e_1123_12 := arithEq_of_rows h (row := 1123) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_12
  simp only [← c69408, ← c69409, ← c69410] at e_1123_12
  have e_1123_13 := arithEq_of_rows h (row := 1123) (i := 13) rfl (by decide)
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
  have e_1123_14 := arithEq_of_rows h (row := 1123) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_14
  simp only [← c69414, ← c69415, ← c69416] at e_1123_14
  have e_1123_15 := arithEq_of_rows h (row := 1123) (i := 15) rfl (by decide)
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
  have e_1123_16 := arithEq_of_rows h (row := 1123) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_16
  simp only [← c69420, ← c69421, ← c69422] at e_1123_16
  have e_1123_17 := arithEq_of_rows h (row := 1123) (i := 17) rfl (by decide)
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
  have e_1123_18 := arithEq_of_rows h (row := 1123) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1123_18
  simp only [← c69426, ← c69427, ← c69428] at e_1123_18
  have e_1123_19 := arithEq_of_rows h (row := 1123) (i := 19) rfl (by decide)
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
  have e_1124_0 := arithEq_of_rows h (row := 1124) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_0
  simp only [← c69432, ← c69433, ← c69434] at e_1124_0
  have e_1124_1 := arithEq_of_rows h (row := 1124) (i := 1) rfl (by decide)
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
  have e_1124_2 := arithEq_of_rows h (row := 1124) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_2
  simp only [← c69438, ← c69439, ← c69440] at e_1124_2
  have e_1124_3 := arithEq_of_rows h (row := 1124) (i := 3) rfl (by decide)
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
  have e_1124_4 := arithEq_of_rows h (row := 1124) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_4
  simp only [← c69444, ← c69445, ← c69446] at e_1124_4
  have e_1124_5 := arithEq_of_rows h (row := 1124) (i := 5) rfl (by decide)
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
  have e_1124_6 := arithEq_of_rows h (row := 1124) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_6
  simp only [← c69450, ← c69451, ← c69452] at e_1124_6
  have e_1124_7 := arithEq_of_rows h (row := 1124) (i := 7) rfl (by decide)
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
  have e_1124_8 := arithEq_of_rows h (row := 1124) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_8
  simp only [← c69456, ← c69457, ← c69458] at e_1124_8
  have e_1124_9 := arithEq_of_rows h (row := 1124) (i := 9) rfl (by decide)
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
  have e_1124_10 := arithEq_of_rows h (row := 1124) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_10
  simp only [← c69462, ← c69463, ← c69464] at e_1124_10
  have e_1124_11 := arithEq_of_rows h (row := 1124) (i := 11) rfl (by decide)
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
  have e_1124_12 := arithEq_of_rows h (row := 1124) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_12
  simp only [← c69468, ← c69469, ← c69470] at e_1124_12
  have e_1124_13 := arithEq_of_rows h (row := 1124) (i := 13) rfl (by decide)
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
  have e_1124_14 := arithEq_of_rows h (row := 1124) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_14
  simp only [← c69474, ← c69475, ← c69476] at e_1124_14
  have e_1124_15 := arithEq_of_rows h (row := 1124) (i := 15) rfl (by decide)
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
  have e_1124_16 := arithEq_of_rows h (row := 1124) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_16
  simp only [← c69480, ← c69481, ← c69482] at e_1124_16
  have e_1124_17 := arithEq_of_rows h (row := 1124) (i := 17) rfl (by decide)
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
  have e_1124_18 := arithEq_of_rows h (row := 1124) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1124_18
  simp only [← c69486, ← c69487, ← c69488] at e_1124_18
  have e_1124_19 := arithEq_of_rows h (row := 1124) (i := 19) rfl (by decide)
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
  have e_1125_0 := arithEq_of_rows h (row := 1125) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_0
  simp only [← c69492, ← c69493, ← c69494] at e_1125_0
  have e_1125_1 := arithEq_of_rows h (row := 1125) (i := 1) rfl (by decide)
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
  have e_1125_2 := arithEq_of_rows h (row := 1125) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_2
  simp only [← c69498, ← c69499, ← c69500] at e_1125_2
  have e_1125_3 := arithEq_of_rows h (row := 1125) (i := 3) rfl (by decide)
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
  have e_1125_4 := arithEq_of_rows h (row := 1125) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_4
  simp only [← c69504, ← c69505, ← c69506] at e_1125_4
  have e_1125_5 := arithEq_of_rows h (row := 1125) (i := 5) rfl (by decide)
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
  have e_1125_6 := arithEq_of_rows h (row := 1125) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_6
  simp only [← c69510, ← c69511, ← c69512] at e_1125_6
  have e_1125_7 := arithEq_of_rows h (row := 1125) (i := 7) rfl (by decide)
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
  have e_1125_8 := arithEq_of_rows h (row := 1125) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_8
  simp only [← c69516, ← c69517, ← c69518] at e_1125_8
  have e_1125_9 := arithEq_of_rows h (row := 1125) (i := 9) rfl (by decide)
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
  have e_1125_10 := arithEq_of_rows h (row := 1125) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_10
  simp only [← c69522, ← c69523, ← c69524] at e_1125_10
  have e_1125_11 := arithEq_of_rows h (row := 1125) (i := 11) rfl (by decide)
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
  have e_1125_12 := arithEq_of_rows h (row := 1125) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_12
  simp only [← c69528, ← c69529, ← c69530] at e_1125_12
  have e_1125_13 := arithEq_of_rows h (row := 1125) (i := 13) rfl (by decide)
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
  have e_1125_14 := arithEq_of_rows h (row := 1125) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_14
  simp only [← c69534, ← c69535, ← c69536] at e_1125_14
  have e_1125_15 := arithEq_of_rows h (row := 1125) (i := 15) rfl (by decide)
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
  have e_1125_16 := arithEq_of_rows h (row := 1125) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_16
  simp only [← c69540, ← c69541, ← c69542] at e_1125_16
  have e_1125_17 := arithEq_of_rows h (row := 1125) (i := 17) rfl (by decide)
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
  have e_1125_18 := arithEq_of_rows h (row := 1125) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1125_18
  simp only [← c69546, ← c69547, ← c69548] at e_1125_18
  have e_1125_19 := arithEq_of_rows h (row := 1125) (i := 19) rfl (by decide)
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
  have e_1126_0 := arithEq_of_rows h (row := 1126) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_0
  simp only [← c69552, ← c69553, ← c69554] at e_1126_0
  have e_1126_1 := arithEq_of_rows h (row := 1126) (i := 1) rfl (by decide)
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
  have e_1126_2 := arithEq_of_rows h (row := 1126) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_2
  simp only [← c69558, ← c69559, ← c69560] at e_1126_2
  have e_1126_3 := arithEq_of_rows h (row := 1126) (i := 3) rfl (by decide)
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
  have e_1126_4 := arithEq_of_rows h (row := 1126) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_4
  simp only [← c69564, ← c69565, ← c69566] at e_1126_4
  have e_1126_5 := arithEq_of_rows h (row := 1126) (i := 5) rfl (by decide)
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
  have e_1126_6 := arithEq_of_rows h (row := 1126) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_6
  simp only [← c69570, ← c69571, ← c69572] at e_1126_6
  have e_1126_7 := arithEq_of_rows h (row := 1126) (i := 7) rfl (by decide)
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
  have e_1126_8 := arithEq_of_rows h (row := 1126) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_8
  simp only [← c69576, ← c69577, ← c69578] at e_1126_8
  have e_1126_9 := arithEq_of_rows h (row := 1126) (i := 9) rfl (by decide)
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
  have e_1126_10 := arithEq_of_rows h (row := 1126) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_10
  simp only [← c69582, ← c69583, ← c69584] at e_1126_10
  have e_1126_11 := arithEq_of_rows h (row := 1126) (i := 11) rfl (by decide)
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
  have e_1126_12 := arithEq_of_rows h (row := 1126) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_12
  simp only [← c69588, ← c69589, ← c69590] at e_1126_12
  have e_1126_13 := arithEq_of_rows h (row := 1126) (i := 13) rfl (by decide)
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
  have e_1126_14 := arithEq_of_rows h (row := 1126) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_14
  simp only [← c69594, ← c69595, ← c69596] at e_1126_14
  have e_1126_15 := arithEq_of_rows h (row := 1126) (i := 15) rfl (by decide)
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
  have e_1126_16 := arithEq_of_rows h (row := 1126) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_16
  simp only [← c69600, ← c69601, ← c69602] at e_1126_16
  have e_1126_17 := arithEq_of_rows h (row := 1126) (i := 17) rfl (by decide)
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
  have e_1126_18 := arithEq_of_rows h (row := 1126) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1126_18
  simp only [← c69606, ← c69607, ← c69608] at e_1126_18
  have e_1126_19 := arithEq_of_rows h (row := 1126) (i := 19) rfl (by decide)
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
  have e_1127_0 := arithEq_of_rows h (row := 1127) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_0
  simp only [← c69612, ← c69613, ← c69614] at e_1127_0
  have e_1127_1 := arithEq_of_rows h (row := 1127) (i := 1) rfl (by decide)
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
  have e_1127_2 := arithEq_of_rows h (row := 1127) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_2
  simp only [← c69618, ← c69619, ← c69620] at e_1127_2
  have e_1127_3 := arithEq_of_rows h (row := 1127) (i := 3) rfl (by decide)
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
  have e_1127_4 := arithEq_of_rows h (row := 1127) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_4
  simp only [← c69624, ← c69625, ← c69626] at e_1127_4
  have e_1127_5 := arithEq_of_rows h (row := 1127) (i := 5) rfl (by decide)
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
  have e_1127_6 := arithEq_of_rows h (row := 1127) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_6
  simp only [← c69630, ← c69631, ← c69632] at e_1127_6
  have e_1127_7 := arithEq_of_rows h (row := 1127) (i := 7) rfl (by decide)
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
  have e_1127_8 := arithEq_of_rows h (row := 1127) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_8
  simp only [← c69636, ← c69637, ← c69638] at e_1127_8
  have e_1127_9 := arithEq_of_rows h (row := 1127) (i := 9) rfl (by decide)
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
  have e_1127_10 := arithEq_of_rows h (row := 1127) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_10
  simp only [← c69642, ← c69643, ← c69644] at e_1127_10
  have e_1127_11 := arithEq_of_rows h (row := 1127) (i := 11) rfl (by decide)
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
  have e_1127_12 := arithEq_of_rows h (row := 1127) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_12
  simp only [← c69648, ← c69649, ← c69650] at e_1127_12
  have e_1127_13 := arithEq_of_rows h (row := 1127) (i := 13) rfl (by decide)
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
  have e_1127_14 := arithEq_of_rows h (row := 1127) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_14
  simp only [← c69654, ← c69655, ← c69656] at e_1127_14
  have e_1127_15 := arithEq_of_rows h (row := 1127) (i := 15) rfl (by decide)
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
  have e_1127_16 := arithEq_of_rows h (row := 1127) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_16
  simp only [← c69660, ← c69661, ← c69662] at e_1127_16
  have e_1127_17 := arithEq_of_rows h (row := 1127) (i := 17) rfl (by decide)
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
  have e_1127_18 := arithEq_of_rows h (row := 1127) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1127_18
  simp only [← c69666, ← c69667, ← c69668] at e_1127_18
  have e_1127_19 := arithEq_of_rows h (row := 1127) (i := 19) rfl (by decide)
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
  have e_1128_0 := arithEq_of_rows h (row := 1128) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_0
  simp only [← c69672, ← c69673, ← c69674] at e_1128_0
  have e_1128_1 := arithEq_of_rows h (row := 1128) (i := 1) rfl (by decide)
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
  have e_1128_2 := arithEq_of_rows h (row := 1128) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_2
  simp only [← c69678, ← c69679, ← c69680] at e_1128_2
  have e_1128_3 := arithEq_of_rows h (row := 1128) (i := 3) rfl (by decide)
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
  have e_1128_4 := arithEq_of_rows h (row := 1128) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_4
  simp only [← c69684, ← c69685, ← c69686] at e_1128_4
  have e_1128_5 := arithEq_of_rows h (row := 1128) (i := 5) rfl (by decide)
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
  have e_1128_6 := arithEq_of_rows h (row := 1128) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_6
  simp only [← c69690, ← c69691, ← c69692] at e_1128_6
  have e_1128_7 := arithEq_of_rows h (row := 1128) (i := 7) rfl (by decide)
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
  have e_1128_8 := arithEq_of_rows h (row := 1128) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_8
  simp only [← c69696, ← c69697, ← c69698] at e_1128_8
  have e_1128_9 := arithEq_of_rows h (row := 1128) (i := 9) rfl (by decide)
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
  have e_1128_10 := arithEq_of_rows h (row := 1128) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_10
  simp only [← c69702, ← c69703, ← c69704] at e_1128_10
  have e_1128_11 := arithEq_of_rows h (row := 1128) (i := 11) rfl (by decide)
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
  have e_1128_12 := arithEq_of_rows h (row := 1128) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_12
  simp only [← c69708, ← c69709, ← c69710] at e_1128_12
  have e_1128_13 := arithEq_of_rows h (row := 1128) (i := 13) rfl (by decide)
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
  have e_1128_14 := arithEq_of_rows h (row := 1128) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_14
  simp only [← c69714, ← c69715, ← c69716] at e_1128_14
  have e_1128_15 := arithEq_of_rows h (row := 1128) (i := 15) rfl (by decide)
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
  have e_1128_16 := arithEq_of_rows h (row := 1128) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_16
  simp only [← c69720, ← c69721, ← c69722] at e_1128_16
  have e_1128_17 := arithEq_of_rows h (row := 1128) (i := 17) rfl (by decide)
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
  have e_1128_18 := arithEq_of_rows h (row := 1128) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1128_18
  simp only [← c69726, ← c69727, ← c69728] at e_1128_18
  have e_1128_19 := arithEq_of_rows h (row := 1128) (i := 19) rfl (by decide)
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
  have e_1129_0 := arithEq_of_rows h (row := 1129) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_0
  simp only [← c69732, ← c69733, ← c69734] at e_1129_0
  have e_1129_1 := arithEq_of_rows h (row := 1129) (i := 1) rfl (by decide)
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
  have e_1129_2 := arithEq_of_rows h (row := 1129) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_2
  simp only [← c69738, ← c69739, ← c69740] at e_1129_2
  have e_1129_3 := arithEq_of_rows h (row := 1129) (i := 3) rfl (by decide)
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
  have e_1129_4 := arithEq_of_rows h (row := 1129) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_4
  simp only [← c69744, ← c69745, ← c69746] at e_1129_4
  have e_1129_5 := arithEq_of_rows h (row := 1129) (i := 5) rfl (by decide)
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
  have e_1129_6 := arithEq_of_rows h (row := 1129) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_6
  simp only [← c69750, ← c69751, ← c69752] at e_1129_6
  have e_1129_7 := arithEq_of_rows h (row := 1129) (i := 7) rfl (by decide)
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
  have e_1129_8 := arithEq_of_rows h (row := 1129) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_8
  simp only [← c69756, ← c69757, ← c69758] at e_1129_8
  have e_1129_9 := arithEq_of_rows h (row := 1129) (i := 9) rfl (by decide)
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
  have e_1129_10 := arithEq_of_rows h (row := 1129) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_10
  simp only [← c69762, ← c69763, ← c69764] at e_1129_10
  have e_1129_11 := arithEq_of_rows h (row := 1129) (i := 11) rfl (by decide)
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
  have e_1129_12 := arithEq_of_rows h (row := 1129) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_12
  simp only [← c69768, ← c69769, ← c69770] at e_1129_12
  have e_1129_13 := arithEq_of_rows h (row := 1129) (i := 13) rfl (by decide)
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
  have e_1129_14 := arithEq_of_rows h (row := 1129) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_14
  simp only [← c69774, ← c69775, ← c69776] at e_1129_14
  have e_1129_15 := arithEq_of_rows h (row := 1129) (i := 15) rfl (by decide)
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
  have e_1129_16 := arithEq_of_rows h (row := 1129) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_16
  simp only [← c69780, ← c69781, ← c69782] at e_1129_16
  have e_1129_17 := arithEq_of_rows h (row := 1129) (i := 17) rfl (by decide)
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
  have e_1129_18 := arithEq_of_rows h (row := 1129) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1129_18
  simp only [← c69786, ← c69787, ← c69788] at e_1129_18
  have e_1129_19 := arithEq_of_rows h (row := 1129) (i := 19) rfl (by decide)
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
  have e_1130_0 := arithEq_of_rows h (row := 1130) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_0
  simp only [← c69792, ← c69793, ← c69794] at e_1130_0
  have e_1130_1 := arithEq_of_rows h (row := 1130) (i := 1) rfl (by decide)
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
  have e_1130_2 := arithEq_of_rows h (row := 1130) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_2
  simp only [← c69798, ← c69799, ← c69800] at e_1130_2
  have e_1130_3 := arithEq_of_rows h (row := 1130) (i := 3) rfl (by decide)
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
  have e_1130_4 := arithEq_of_rows h (row := 1130) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_4
  simp only [← c69804, ← c69805, ← c69806] at e_1130_4
  have e_1130_5 := arithEq_of_rows h (row := 1130) (i := 5) rfl (by decide)
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
  have e_1130_6 := arithEq_of_rows h (row := 1130) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_6
  simp only [← c69810, ← c69811, ← c69812] at e_1130_6
  have e_1130_7 := arithEq_of_rows h (row := 1130) (i := 7) rfl (by decide)
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
  have e_1130_8 := arithEq_of_rows h (row := 1130) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_8
  simp only [← c69816, ← c69817, ← c69818] at e_1130_8
  have e_1130_9 := arithEq_of_rows h (row := 1130) (i := 9) rfl (by decide)
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
  have e_1130_10 := arithEq_of_rows h (row := 1130) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_10
  simp only [← c69822, ← c69823, ← c69824] at e_1130_10
  have e_1130_11 := arithEq_of_rows h (row := 1130) (i := 11) rfl (by decide)
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
  have e_1130_12 := arithEq_of_rows h (row := 1130) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_12
  simp only [← c69828, ← c69829, ← c69830] at e_1130_12
  have e_1130_13 := arithEq_of_rows h (row := 1130) (i := 13) rfl (by decide)
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
  have e_1130_14 := arithEq_of_rows h (row := 1130) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_14
  simp only [← c69834, ← c69835, ← c69836] at e_1130_14
  have e_1130_15 := arithEq_of_rows h (row := 1130) (i := 15) rfl (by decide)
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
  have e_1130_16 := arithEq_of_rows h (row := 1130) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_16
  simp only [← c69840, ← c69841, ← c69842] at e_1130_16
  have e_1130_17 := arithEq_of_rows h (row := 1130) (i := 17) rfl (by decide)
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
  have e_1130_18 := arithEq_of_rows h (row := 1130) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1130_18
  simp only [← c69846, ← c69847, ← c69848] at e_1130_18
  have e_1130_19 := arithEq_of_rows h (row := 1130) (i := 19) rfl (by decide)
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
  have e_1131_0 := arithEq_of_rows h (row := 1131) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_0
  simp only [← c69852, ← c69853, ← c69854] at e_1131_0
  have e_1131_1 := arithEq_of_rows h (row := 1131) (i := 1) rfl (by decide)
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
  have e_1131_2 := arithEq_of_rows h (row := 1131) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_2
  simp only [← c69858, ← c69859, ← c69860] at e_1131_2
  have e_1131_3 := arithEq_of_rows h (row := 1131) (i := 3) rfl (by decide)
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
  have e_1131_4 := arithEq_of_rows h (row := 1131) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_4
  simp only [← c69864, ← c69865, ← c69866] at e_1131_4
  have e_1131_5 := arithEq_of_rows h (row := 1131) (i := 5) rfl (by decide)
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
  have e_1131_6 := arithEq_of_rows h (row := 1131) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_6
  simp only [← c69870, ← c69871, ← c69872] at e_1131_6
  have e_1131_7 := arithEq_of_rows h (row := 1131) (i := 7) rfl (by decide)
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
  have e_1131_8 := arithEq_of_rows h (row := 1131) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_8
  simp only [← c69876, ← c69877, ← c69878] at e_1131_8
  have e_1131_9 := arithEq_of_rows h (row := 1131) (i := 9) rfl (by decide)
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
  have e_1131_10 := arithEq_of_rows h (row := 1131) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_10
  simp only [← c69882, ← c69883, ← c69884] at e_1131_10
  have e_1131_11 := arithEq_of_rows h (row := 1131) (i := 11) rfl (by decide)
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
  have e_1131_12 := arithEq_of_rows h (row := 1131) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_12
  simp only [← c69888, ← c69889, ← c69890] at e_1131_12
  have e_1131_13 := arithEq_of_rows h (row := 1131) (i := 13) rfl (by decide)
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
  have e_1131_14 := arithEq_of_rows h (row := 1131) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_14
  simp only [← c69894, ← c69895, ← c69896] at e_1131_14
  have e_1131_15 := arithEq_of_rows h (row := 1131) (i := 15) rfl (by decide)
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
  have e_1131_16 := arithEq_of_rows h (row := 1131) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_16
  simp only [← c69900, ← c69901, ← c69902] at e_1131_16
  have e_1131_17 := arithEq_of_rows h (row := 1131) (i := 17) rfl (by decide)
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
  have e_1131_18 := arithEq_of_rows h (row := 1131) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1131_18
  simp only [← c69906, ← c69907, ← c69908] at e_1131_18
  have e_1131_19 := arithEq_of_rows h (row := 1131) (i := 19) rfl (by decide)
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
  have e_1132_0 := arithEq_of_rows h (row := 1132) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_0
  simp only [← c69912, ← c69913, ← c69914] at e_1132_0
  have e_1132_1 := arithEq_of_rows h (row := 1132) (i := 1) rfl (by decide)
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
  have e_1132_2 := arithEq_of_rows h (row := 1132) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_2
  simp only [← c69918, ← c69919, ← c69920] at e_1132_2
  have e_1132_3 := arithEq_of_rows h (row := 1132) (i := 3) rfl (by decide)
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
  have e_1132_4 := arithEq_of_rows h (row := 1132) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_4
  simp only [← c69924, ← c69925, ← c69926] at e_1132_4
  have e_1132_5 := arithEq_of_rows h (row := 1132) (i := 5) rfl (by decide)
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
  have e_1132_6 := arithEq_of_rows h (row := 1132) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_6
  simp only [← c69930, ← c69931, ← c69932] at e_1132_6
  have e_1132_7 := arithEq_of_rows h (row := 1132) (i := 7) rfl (by decide)
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
  have e_1132_8 := arithEq_of_rows h (row := 1132) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_8
  simp only [← c69936, ← c69937, ← c69938] at e_1132_8
  have e_1132_9 := arithEq_of_rows h (row := 1132) (i := 9) rfl (by decide)
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
  have e_1132_10 := arithEq_of_rows h (row := 1132) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_10
  simp only [← c69942, ← c69943, ← c69944] at e_1132_10
  have e_1132_11 := arithEq_of_rows h (row := 1132) (i := 11) rfl (by decide)
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
  have e_1132_12 := arithEq_of_rows h (row := 1132) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_12
  simp only [← c69948, ← c69949, ← c69950] at e_1132_12
  have e_1132_13 := arithEq_of_rows h (row := 1132) (i := 13) rfl (by decide)
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
  have e_1132_14 := arithEq_of_rows h (row := 1132) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_14
  simp only [← c69954, ← c69955, ← c69956] at e_1132_14
  have e_1132_15 := arithEq_of_rows h (row := 1132) (i := 15) rfl (by decide)
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
  have e_1132_16 := arithEq_of_rows h (row := 1132) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_16
  simp only [← c69960, ← c69961, ← c69962] at e_1132_16
  have e_1132_17 := arithEq_of_rows h (row := 1132) (i := 17) rfl (by decide)
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
  have e_1132_18 := arithEq_of_rows h (row := 1132) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1132_18
  simp only [← c69966, ← c69967, ← c69968] at e_1132_18
  have e_1132_19 := arithEq_of_rows h (row := 1132) (i := 19) rfl (by decide)
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
  have e_1133_0 := arithEq_of_rows h (row := 1133) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_0
  simp only [← c69972, ← c69973, ← c69974] at e_1133_0
  have e_1133_1 := arithEq_of_rows h (row := 1133) (i := 1) rfl (by decide)
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
  have e_1133_2 := arithEq_of_rows h (row := 1133) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_2
  simp only [← c69978, ← c69979, ← c69980] at e_1133_2
  have e_1133_3 := arithEq_of_rows h (row := 1133) (i := 3) rfl (by decide)
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
  have e_1133_4 := arithEq_of_rows h (row := 1133) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_4
  simp only [← c69984, ← c69985, ← c69986] at e_1133_4
  have e_1133_5 := arithEq_of_rows h (row := 1133) (i := 5) rfl (by decide)
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
  have e_1133_6 := arithEq_of_rows h (row := 1133) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_6
  simp only [← c69990, ← c69991, ← c69992] at e_1133_6
  have e_1133_7 := arithEq_of_rows h (row := 1133) (i := 7) rfl (by decide)
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
  have e_1133_8 := arithEq_of_rows h (row := 1133) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_8
  simp only [← c69996, ← c69997, ← c69998] at e_1133_8
  have e_1133_9 := arithEq_of_rows h (row := 1133) (i := 9) rfl (by decide)
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
  have e_1133_10 := arithEq_of_rows h (row := 1133) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_10
  simp only [← c70002, ← c70003, ← c70004] at e_1133_10
  have e_1133_11 := arithEq_of_rows h (row := 1133) (i := 11) rfl (by decide)
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
  have e_1133_12 := arithEq_of_rows h (row := 1133) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_12
  simp only [← c70008, ← c70009, ← c70010] at e_1133_12
  have e_1133_13 := arithEq_of_rows h (row := 1133) (i := 13) rfl (by decide)
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
  have e_1133_14 := arithEq_of_rows h (row := 1133) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_14
  simp only [← c70014, ← c70015, ← c70016] at e_1133_14
  have e_1133_15 := arithEq_of_rows h (row := 1133) (i := 15) rfl (by decide)
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
  have e_1133_16 := arithEq_of_rows h (row := 1133) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_16
  simp only [← c70020, ← c70021, ← c70022] at e_1133_16
  have e_1133_17 := arithEq_of_rows h (row := 1133) (i := 17) rfl (by decide)
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
  have e_1133_18 := arithEq_of_rows h (row := 1133) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1133_18
  simp only [← c70026, ← c70027, ← c70028] at e_1133_18
  have e_1133_19 := arithEq_of_rows h (row := 1133) (i := 19) rfl (by decide)
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
  have e_1134_0 := arithEq_of_rows h (row := 1134) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_0
  simp only [← c70032, ← c70033, ← c70034] at e_1134_0
  have e_1134_1 := arithEq_of_rows h (row := 1134) (i := 1) rfl (by decide)
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
  have e_1134_2 := arithEq_of_rows h (row := 1134) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_2
  simp only [← c70038, ← c70039, ← c70040] at e_1134_2
  have e_1134_3 := arithEq_of_rows h (row := 1134) (i := 3) rfl (by decide)
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
  have e_1134_4 := arithEq_of_rows h (row := 1134) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_4
  simp only [← c70044, ← c70045, ← c70046] at e_1134_4
  have e_1134_5 := arithEq_of_rows h (row := 1134) (i := 5) rfl (by decide)
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
  have e_1134_6 := arithEq_of_rows h (row := 1134) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_6
  simp only [← c70050, ← c70051, ← c70052] at e_1134_6
  have e_1134_7 := arithEq_of_rows h (row := 1134) (i := 7) rfl (by decide)
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
  have e_1134_8 := arithEq_of_rows h (row := 1134) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_8
  simp only [← c70056, ← c70057, ← c70058] at e_1134_8
  have e_1134_9 := arithEq_of_rows h (row := 1134) (i := 9) rfl (by decide)
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
  have e_1134_10 := arithEq_of_rows h (row := 1134) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_10
  simp only [← c70062, ← c70063, ← c70064] at e_1134_10
  have e_1134_11 := arithEq_of_rows h (row := 1134) (i := 11) rfl (by decide)
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
  have e_1134_12 := arithEq_of_rows h (row := 1134) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_12
  simp only [← c70068, ← c70069, ← c70070] at e_1134_12
  have e_1134_13 := arithEq_of_rows h (row := 1134) (i := 13) rfl (by decide)
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
  have e_1134_14 := arithEq_of_rows h (row := 1134) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_14
  simp only [← c70074, ← c70075, ← c70076] at e_1134_14
  have e_1134_15 := arithEq_of_rows h (row := 1134) (i := 15) rfl (by decide)
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
  have e_1134_16 := arithEq_of_rows h (row := 1134) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_16
  simp only [← c70080, ← c70081, ← c70082] at e_1134_16
  have e_1134_17 := arithEq_of_rows h (row := 1134) (i := 17) rfl (by decide)
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
  have e_1134_18 := arithEq_of_rows h (row := 1134) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1134_18
  simp only [← c70086, ← c70087, ← c70088] at e_1134_18
  have e_1134_19 := arithEq_of_rows h (row := 1134) (i := 19) rfl (by decide)
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
  have e_1135_0 := arithEq_of_rows h (row := 1135) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_0
  simp only [← c70092, ← c70093, ← c70094] at e_1135_0
  have e_1135_1 := arithEq_of_rows h (row := 1135) (i := 1) rfl (by decide)
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
  have e_1135_2 := arithEq_of_rows h (row := 1135) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_2
  simp only [← c70098, ← c70099, ← c70100] at e_1135_2
  have e_1135_3 := arithEq_of_rows h (row := 1135) (i := 3) rfl (by decide)
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
  have e_1135_4 := arithEq_of_rows h (row := 1135) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_4
  simp only [← c70104, ← c70105, ← c70106] at e_1135_4
  have e_1135_5 := arithEq_of_rows h (row := 1135) (i := 5) rfl (by decide)
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
  have e_1135_6 := arithEq_of_rows h (row := 1135) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_6
  simp only [← c70110, ← c70111, ← c70112] at e_1135_6
  have e_1135_7 := arithEq_of_rows h (row := 1135) (i := 7) rfl (by decide)
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
  have e_1135_8 := arithEq_of_rows h (row := 1135) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_8
  simp only [← c70116, ← c70117, ← c70118] at e_1135_8
  have e_1135_9 := arithEq_of_rows h (row := 1135) (i := 9) rfl (by decide)
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
  have e_1135_10 := arithEq_of_rows h (row := 1135) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_10
  simp only [← c70122, ← c70123, ← c70124] at e_1135_10
  have e_1135_11 := arithEq_of_rows h (row := 1135) (i := 11) rfl (by decide)
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
  have e_1135_12 := arithEq_of_rows h (row := 1135) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_12
  simp only [← c70128, ← c70129, ← c70130] at e_1135_12
  have e_1135_13 := arithEq_of_rows h (row := 1135) (i := 13) rfl (by decide)
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
  have e_1135_14 := arithEq_of_rows h (row := 1135) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_14
  simp only [← c70134, ← c70135, ← c70136] at e_1135_14
  have e_1135_15 := arithEq_of_rows h (row := 1135) (i := 15) rfl (by decide)
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
  have e_1135_16 := arithEq_of_rows h (row := 1135) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_16
  simp only [← c70140, ← c70141, ← c70142] at e_1135_16
  have e_1135_17 := arithEq_of_rows h (row := 1135) (i := 17) rfl (by decide)
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
  have e_1135_18 := arithEq_of_rows h (row := 1135) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1135_18
  simp only [← c70146, ← c70147, ← c70148] at e_1135_18
  have e_1135_19 := arithEq_of_rows h (row := 1135) (i := 19) rfl (by decide)
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
  have e_1136_0 := arithEq_of_rows h (row := 1136) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_0
  simp only [← c70152, ← c70153, ← c70154] at e_1136_0
  have e_1136_1 := arithEq_of_rows h (row := 1136) (i := 1) rfl (by decide)
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
  have e_1136_2 := arithEq_of_rows h (row := 1136) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_2
  simp only [← c70158, ← c70159, ← c70160] at e_1136_2
  have e_1136_3 := arithEq_of_rows h (row := 1136) (i := 3) rfl (by decide)
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
  have e_1136_4 := arithEq_of_rows h (row := 1136) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_4
  simp only [← c70164, ← c70165, ← c70166] at e_1136_4
  have e_1136_5 := arithEq_of_rows h (row := 1136) (i := 5) rfl (by decide)
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
  have e_1136_6 := arithEq_of_rows h (row := 1136) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_6
  simp only [← c70170, ← c70171, ← c70172] at e_1136_6
  have e_1136_7 := arithEq_of_rows h (row := 1136) (i := 7) rfl (by decide)
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
  have e_1136_8 := arithEq_of_rows h (row := 1136) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_8
  simp only [← c70176, ← c70177, ← c70178] at e_1136_8
  have e_1136_9 := arithEq_of_rows h (row := 1136) (i := 9) rfl (by decide)
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
  have e_1136_10 := arithEq_of_rows h (row := 1136) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_10
  simp only [← c70182, ← c70183, ← c70184] at e_1136_10
  have e_1136_11 := arithEq_of_rows h (row := 1136) (i := 11) rfl (by decide)
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
  have e_1136_12 := arithEq_of_rows h (row := 1136) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_12
  simp only [← c70188, ← c70189, ← c70190] at e_1136_12
  have e_1136_13 := arithEq_of_rows h (row := 1136) (i := 13) rfl (by decide)
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
  have e_1136_14 := arithEq_of_rows h (row := 1136) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_14
  simp only [← c70194, ← c70195, ← c70196] at e_1136_14
  have e_1136_15 := arithEq_of_rows h (row := 1136) (i := 15) rfl (by decide)
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
  have e_1136_16 := arithEq_of_rows h (row := 1136) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_16
  simp only [← c70200, ← c70201, ← c70202] at e_1136_16
  have e_1136_17 := arithEq_of_rows h (row := 1136) (i := 17) rfl (by decide)
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
  have e_1136_18 := arithEq_of_rows h (row := 1136) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1136_18
  simp only [← c70206, ← c70207, ← c70208] at e_1136_18
  have e_1136_19 := arithEq_of_rows h (row := 1136) (i := 19) rfl (by decide)
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
  have e_1137_0 := arithEq_of_rows h (row := 1137) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_0
  simp only [← c70212, ← c70213, ← c70214] at e_1137_0
  have e_1137_1 := arithEq_of_rows h (row := 1137) (i := 1) rfl (by decide)
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
  have e_1137_2 := arithEq_of_rows h (row := 1137) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_2
  simp only [← c70218, ← c70219, ← c70220] at e_1137_2
  have e_1137_3 := arithEq_of_rows h (row := 1137) (i := 3) rfl (by decide)
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
  have e_1137_4 := arithEq_of_rows h (row := 1137) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_4
  simp only [← c70224, ← c70225, ← c70226] at e_1137_4
  have e_1137_5 := arithEq_of_rows h (row := 1137) (i := 5) rfl (by decide)
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
  have e_1137_6 := arithEq_of_rows h (row := 1137) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_6
  simp only [← c70230, ← c70231, ← c70232] at e_1137_6
  have e_1137_7 := arithEq_of_rows h (row := 1137) (i := 7) rfl (by decide)
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
  have e_1137_8 := arithEq_of_rows h (row := 1137) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_8
  simp only [← c70236, ← c70237, ← c70238] at e_1137_8
  have e_1137_9 := arithEq_of_rows h (row := 1137) (i := 9) rfl (by decide)
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
  have e_1137_10 := arithEq_of_rows h (row := 1137) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_10
  simp only [← c70242, ← c70243, ← c70244] at e_1137_10
  have e_1137_11 := arithEq_of_rows h (row := 1137) (i := 11) rfl (by decide)
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
  have e_1137_12 := arithEq_of_rows h (row := 1137) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_12
  simp only [← c70248, ← c70249, ← c70250] at e_1137_12
  have e_1137_13 := arithEq_of_rows h (row := 1137) (i := 13) rfl (by decide)
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
  have e_1137_14 := arithEq_of_rows h (row := 1137) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_14
  simp only [← c70254, ← c70255, ← c70256] at e_1137_14
  have e_1137_15 := arithEq_of_rows h (row := 1137) (i := 15) rfl (by decide)
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
  have e_1137_16 := arithEq_of_rows h (row := 1137) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_16
  simp only [← c70260, ← c70261, ← c70262] at e_1137_16
  have e_1137_17 := arithEq_of_rows h (row := 1137) (i := 17) rfl (by decide)
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
  have e_1137_18 := arithEq_of_rows h (row := 1137) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1137_18
  simp only [← c70266, ← c70267, ← c70268] at e_1137_18
  have e_1137_19 := arithEq_of_rows h (row := 1137) (i := 19) rfl (by decide)
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
  have e_1138_0 := arithEq_of_rows h (row := 1138) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_0
  simp only [← c70272, ← c70273, ← c70274] at e_1138_0
  have e_1138_1 := arithEq_of_rows h (row := 1138) (i := 1) rfl (by decide)
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
  have e_1138_2 := arithEq_of_rows h (row := 1138) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_2
  simp only [← c70278, ← c70279, ← c70280] at e_1138_2
  have e_1138_3 := arithEq_of_rows h (row := 1138) (i := 3) rfl (by decide)
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
  have e_1138_4 := arithEq_of_rows h (row := 1138) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_4
  simp only [← c70284, ← c70285, ← c70286] at e_1138_4
  have e_1138_5 := arithEq_of_rows h (row := 1138) (i := 5) rfl (by decide)
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
  have e_1138_6 := arithEq_of_rows h (row := 1138) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_6
  simp only [← c70290, ← c70291, ← c70292] at e_1138_6
  have e_1138_7 := arithEq_of_rows h (row := 1138) (i := 7) rfl (by decide)
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
  have e_1138_8 := arithEq_of_rows h (row := 1138) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_8
  simp only [← c70296, ← c70297, ← c70298] at e_1138_8
  have e_1138_9 := arithEq_of_rows h (row := 1138) (i := 9) rfl (by decide)
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
  have e_1138_10 := arithEq_of_rows h (row := 1138) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_10
  simp only [← c70302, ← c70303, ← c70304] at e_1138_10
  have e_1138_11 := arithEq_of_rows h (row := 1138) (i := 11) rfl (by decide)
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
  have e_1138_12 := arithEq_of_rows h (row := 1138) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_12
  simp only [← c70308, ← c70309, ← c70310] at e_1138_12
  have e_1138_13 := arithEq_of_rows h (row := 1138) (i := 13) rfl (by decide)
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
  have e_1138_14 := arithEq_of_rows h (row := 1138) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_14
  simp only [← c70314, ← c70315, ← c70316] at e_1138_14
  have e_1138_15 := arithEq_of_rows h (row := 1138) (i := 15) rfl (by decide)
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
  have e_1138_16 := arithEq_of_rows h (row := 1138) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_16
  simp only [← c70320, ← c70321, ← c70322] at e_1138_16
  have e_1138_17 := arithEq_of_rows h (row := 1138) (i := 17) rfl (by decide)
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
  have e_1138_18 := arithEq_of_rows h (row := 1138) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1138_18
  simp only [← c70326, ← c70327, ← c70328] at e_1138_18
  have e_1138_19 := arithEq_of_rows h (row := 1138) (i := 19) rfl (by decide)
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
  have e_1139_0 := arithEq_of_rows h (row := 1139) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_0
  simp only [← c70332, ← c70333, ← c70334] at e_1139_0
  have e_1139_1 := arithEq_of_rows h (row := 1139) (i := 1) rfl (by decide)
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
  have e_1139_2 := arithEq_of_rows h (row := 1139) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_2
  simp only [← c70338, ← c70339, ← c70340] at e_1139_2
  have e_1139_3 := arithEq_of_rows h (row := 1139) (i := 3) rfl (by decide)
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
  have e_1139_4 := arithEq_of_rows h (row := 1139) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_4
  simp only [← c70344, ← c70345, ← c70346] at e_1139_4
  have e_1139_5 := arithEq_of_rows h (row := 1139) (i := 5) rfl (by decide)
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
  have e_1139_6 := arithEq_of_rows h (row := 1139) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_6
  simp only [← c70350, ← c70351, ← c70352] at e_1139_6
  have e_1139_7 := arithEq_of_rows h (row := 1139) (i := 7) rfl (by decide)
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
  have e_1139_8 := arithEq_of_rows h (row := 1139) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_8
  simp only [← c70356, ← c70357, ← c70358] at e_1139_8
  have e_1139_9 := arithEq_of_rows h (row := 1139) (i := 9) rfl (by decide)
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
  have e_1139_10 := arithEq_of_rows h (row := 1139) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_10
  simp only [← c70362, ← c70363, ← c70364] at e_1139_10
  have e_1139_11 := arithEq_of_rows h (row := 1139) (i := 11) rfl (by decide)
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
  have e_1139_12 := arithEq_of_rows h (row := 1139) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_12
  simp only [← c70368, ← c70369, ← c70370] at e_1139_12
  have e_1139_13 := arithEq_of_rows h (row := 1139) (i := 13) rfl (by decide)
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
  have e_1139_14 := arithEq_of_rows h (row := 1139) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_14
  simp only [← c70374, ← c70375, ← c70376] at e_1139_14
  have e_1139_15 := arithEq_of_rows h (row := 1139) (i := 15) rfl (by decide)
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
  have e_1139_16 := arithEq_of_rows h (row := 1139) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_16
  simp only [← c70380, ← c70381, ← c70382] at e_1139_16
  have e_1139_17 := arithEq_of_rows h (row := 1139) (i := 17) rfl (by decide)
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
  have e_1139_18 := arithEq_of_rows h (row := 1139) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1139_18
  simp only [← c70386, ← c70387, ← c70388] at e_1139_18
  have e_1139_19 := arithEq_of_rows h (row := 1139) (i := 19) rfl (by decide)
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
  have e_1140_0 := arithEq_of_rows h (row := 1140) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_0
  simp only [← c70392, ← c70393, ← c70394] at e_1140_0
  have e_1140_1 := arithEq_of_rows h (row := 1140) (i := 1) rfl (by decide)
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
  have e_1140_2 := arithEq_of_rows h (row := 1140) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_2
  simp only [← c70398, ← c70399, ← c70400] at e_1140_2
  have e_1140_3 := arithEq_of_rows h (row := 1140) (i := 3) rfl (by decide)
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
  have e_1140_4 := arithEq_of_rows h (row := 1140) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_4
  simp only [← c70404, ← c70405, ← c70406] at e_1140_4
  have e_1140_5 := arithEq_of_rows h (row := 1140) (i := 5) rfl (by decide)
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
  have e_1140_6 := arithEq_of_rows h (row := 1140) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_6
  simp only [← c70410, ← c70411, ← c70412] at e_1140_6
  have e_1140_7 := arithEq_of_rows h (row := 1140) (i := 7) rfl (by decide)
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
  have e_1140_8 := arithEq_of_rows h (row := 1140) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_8
  simp only [← c70416, ← c70417, ← c70418] at e_1140_8
  have e_1140_9 := arithEq_of_rows h (row := 1140) (i := 9) rfl (by decide)
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
  have e_1140_10 := arithEq_of_rows h (row := 1140) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_10
  simp only [← c70422, ← c70423, ← c70424] at e_1140_10
  have e_1140_11 := arithEq_of_rows h (row := 1140) (i := 11) rfl (by decide)
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
  have e_1140_12 := arithEq_of_rows h (row := 1140) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_12
  simp only [← c70428, ← c70429, ← c70430] at e_1140_12
  have e_1140_13 := arithEq_of_rows h (row := 1140) (i := 13) rfl (by decide)
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
  have e_1140_14 := arithEq_of_rows h (row := 1140) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_14
  simp only [← c70434, ← c70435, ← c70436] at e_1140_14
  have e_1140_15 := arithEq_of_rows h (row := 1140) (i := 15) rfl (by decide)
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
  have e_1140_16 := arithEq_of_rows h (row := 1140) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_16
  simp only [← c70440, ← c70441, ← c70442] at e_1140_16
  have e_1140_17 := arithEq_of_rows h (row := 1140) (i := 17) rfl (by decide)
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
  have e_1140_18 := arithEq_of_rows h (row := 1140) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1140_18
  simp only [← c70446, ← c70447, ← c70448] at e_1140_18
  have e_1140_19 := arithEq_of_rows h (row := 1140) (i := 19) rfl (by decide)
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
  have e_1141_0 := arithEq_of_rows h (row := 1141) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_0
  simp only [← c70452, ← c70453, ← c70454] at e_1141_0
  have e_1141_1 := arithEq_of_rows h (row := 1141) (i := 1) rfl (by decide)
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
  have e_1141_2 := arithEq_of_rows h (row := 1141) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_2
  simp only [← c70458, ← c70459, ← c70460] at e_1141_2
  have e_1141_3 := arithEq_of_rows h (row := 1141) (i := 3) rfl (by decide)
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
  have e_1141_4 := arithEq_of_rows h (row := 1141) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_4
  simp only [← c70464, ← c70465, ← c70466] at e_1141_4
  have e_1141_5 := arithEq_of_rows h (row := 1141) (i := 5) rfl (by decide)
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
  have e_1141_6 := arithEq_of_rows h (row := 1141) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_6
  simp only [← c70470, ← c70471, ← c70472] at e_1141_6
  have e_1141_7 := arithEq_of_rows h (row := 1141) (i := 7) rfl (by decide)
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
  have e_1141_8 := arithEq_of_rows h (row := 1141) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_8
  simp only [← c70476, ← c70477, ← c70478] at e_1141_8
  have e_1141_9 := arithEq_of_rows h (row := 1141) (i := 9) rfl (by decide)
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
  have e_1141_10 := arithEq_of_rows h (row := 1141) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_10
  simp only [← c70482, ← c70483, ← c70484] at e_1141_10
  have e_1141_11 := arithEq_of_rows h (row := 1141) (i := 11) rfl (by decide)
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
  have e_1141_12 := arithEq_of_rows h (row := 1141) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_12
  simp only [← c70488, ← c70489, ← c70490] at e_1141_12
  have e_1141_13 := arithEq_of_rows h (row := 1141) (i := 13) rfl (by decide)
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
  have e_1141_14 := arithEq_of_rows h (row := 1141) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_14
  simp only [← c70494, ← c70495, ← c70496] at e_1141_14
  have e_1141_15 := arithEq_of_rows h (row := 1141) (i := 15) rfl (by decide)
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
  have e_1141_16 := arithEq_of_rows h (row := 1141) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_16
  simp only [← c70500, ← c70501, ← c70502] at e_1141_16
  have e_1141_17 := arithEq_of_rows h (row := 1141) (i := 17) rfl (by decide)
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
  have e_1141_18 := arithEq_of_rows h (row := 1141) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1141_18
  simp only [← c70506, ← c70507, ← c70508] at e_1141_18
  have e_1141_19 := arithEq_of_rows h (row := 1141) (i := 19) rfl (by decide)
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
  have e_1142_0 := arithEq_of_rows h (row := 1142) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_0
  simp only [← c70512, ← c70513, ← c70514] at e_1142_0
  have e_1142_1 := arithEq_of_rows h (row := 1142) (i := 1) rfl (by decide)
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
  have e_1142_2 := arithEq_of_rows h (row := 1142) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_2
  simp only [← c70518, ← c70519, ← c70520] at e_1142_2
  have e_1142_3 := arithEq_of_rows h (row := 1142) (i := 3) rfl (by decide)
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
  have e_1142_4 := arithEq_of_rows h (row := 1142) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_4
  simp only [← c70524, ← c70525, ← c70526] at e_1142_4
  have e_1142_5 := arithEq_of_rows h (row := 1142) (i := 5) rfl (by decide)
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
  have e_1142_6 := arithEq_of_rows h (row := 1142) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_6
  simp only [← c70530, ← c70531, ← c70532] at e_1142_6
  have e_1142_7 := arithEq_of_rows h (row := 1142) (i := 7) rfl (by decide)
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
  have e_1142_8 := arithEq_of_rows h (row := 1142) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_8
  simp only [← c70536, ← c70537, ← c70538] at e_1142_8
  have e_1142_9 := arithEq_of_rows h (row := 1142) (i := 9) rfl (by decide)
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
  have e_1142_10 := arithEq_of_rows h (row := 1142) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_10
  simp only [← c70542, ← c70543, ← c70544] at e_1142_10
  have e_1142_11 := arithEq_of_rows h (row := 1142) (i := 11) rfl (by decide)
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
  have e_1142_12 := arithEq_of_rows h (row := 1142) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_12
  simp only [← c70548, ← c70549, ← c70550] at e_1142_12
  have e_1142_13 := arithEq_of_rows h (row := 1142) (i := 13) rfl (by decide)
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
  have e_1142_14 := arithEq_of_rows h (row := 1142) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_14
  simp only [← c70554, ← c70555, ← c70556] at e_1142_14
  have e_1142_15 := arithEq_of_rows h (row := 1142) (i := 15) rfl (by decide)
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
  have e_1142_16 := arithEq_of_rows h (row := 1142) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_16
  simp only [← c70560, ← c70561, ← c70562] at e_1142_16
  have e_1142_17 := arithEq_of_rows h (row := 1142) (i := 17) rfl (by decide)
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
  have e_1142_18 := arithEq_of_rows h (row := 1142) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1142_18
  simp only [← c70566, ← c70567, ← c70568] at e_1142_18
  have e_1142_19 := arithEq_of_rows h (row := 1142) (i := 19) rfl (by decide)
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
  have e_1143_0 := arithEq_of_rows h (row := 1143) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_0
  simp only [← c70572, ← c70573, ← c70574] at e_1143_0
  have e_1143_1 := arithEq_of_rows h (row := 1143) (i := 1) rfl (by decide)
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
  have e_1143_2 := arithEq_of_rows h (row := 1143) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_2
  simp only [← c70578, ← c70579, ← c70580] at e_1143_2
  have e_1143_3 := arithEq_of_rows h (row := 1143) (i := 3) rfl (by decide)
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
  have e_1143_4 := arithEq_of_rows h (row := 1143) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_4
  simp only [← c70584, ← c70585, ← c70586] at e_1143_4
  have e_1143_5 := arithEq_of_rows h (row := 1143) (i := 5) rfl (by decide)
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
  have e_1143_6 := arithEq_of_rows h (row := 1143) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_6
  simp only [← c70590, ← c70591, ← c70592] at e_1143_6
  have e_1143_7 := arithEq_of_rows h (row := 1143) (i := 7) rfl (by decide)
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
  have e_1143_8 := arithEq_of_rows h (row := 1143) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_8
  simp only [← c70596, ← c70597, ← c70598] at e_1143_8
  have e_1143_9 := arithEq_of_rows h (row := 1143) (i := 9) rfl (by decide)
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
  have e_1143_10 := arithEq_of_rows h (row := 1143) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_10
  simp only [← c70602, ← c70603, ← c70604] at e_1143_10
  have e_1143_11 := arithEq_of_rows h (row := 1143) (i := 11) rfl (by decide)
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
  have e_1143_12 := arithEq_of_rows h (row := 1143) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_12
  simp only [← c70608, ← c70609, ← c70610] at e_1143_12
  have e_1143_13 := arithEq_of_rows h (row := 1143) (i := 13) rfl (by decide)
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
  have e_1143_14 := arithEq_of_rows h (row := 1143) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_14
  simp only [← c70614, ← c70615, ← c70616] at e_1143_14
  have e_1143_15 := arithEq_of_rows h (row := 1143) (i := 15) rfl (by decide)
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
  have e_1143_16 := arithEq_of_rows h (row := 1143) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_16
  simp only [← c70620, ← c70621, ← c70622] at e_1143_16
  have e_1143_17 := arithEq_of_rows h (row := 1143) (i := 17) rfl (by decide)
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
  have e_1143_18 := arithEq_of_rows h (row := 1143) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1143_18
  simp only [← c70626, ← c70627, ← c70628] at e_1143_18
  have e_1143_19 := arithEq_of_rows h (row := 1143) (i := 19) rfl (by decide)
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
  have e_1144_0 := arithEq_of_rows h (row := 1144) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_0
  simp only [← c70632, ← c70633, ← c70634] at e_1144_0
  have e_1144_1 := arithEq_of_rows h (row := 1144) (i := 1) rfl (by decide)
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
  have e_1144_2 := arithEq_of_rows h (row := 1144) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_2
  simp only [← c70638, ← c70639, ← c70640] at e_1144_2
  have e_1144_3 := arithEq_of_rows h (row := 1144) (i := 3) rfl (by decide)
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
  have e_1144_4 := arithEq_of_rows h (row := 1144) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_4
  simp only [← c70644, ← c70645, ← c70646] at e_1144_4
  have e_1144_5 := arithEq_of_rows h (row := 1144) (i := 5) rfl (by decide)
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
  have e_1144_6 := arithEq_of_rows h (row := 1144) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_6
  simp only [← c70650, ← c70651, ← c70652] at e_1144_6
  have e_1144_7 := arithEq_of_rows h (row := 1144) (i := 7) rfl (by decide)
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
  have e_1144_8 := arithEq_of_rows h (row := 1144) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_8
  simp only [← c70656, ← c70657, ← c70658] at e_1144_8
  have e_1144_9 := arithEq_of_rows h (row := 1144) (i := 9) rfl (by decide)
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
  have e_1144_10 := arithEq_of_rows h (row := 1144) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_10
  simp only [← c70662, ← c70663, ← c70664] at e_1144_10
  have e_1144_11 := arithEq_of_rows h (row := 1144) (i := 11) rfl (by decide)
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
  have e_1144_12 := arithEq_of_rows h (row := 1144) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_12
  simp only [← c70668, ← c70669, ← c70670] at e_1144_12
  have e_1144_13 := arithEq_of_rows h (row := 1144) (i := 13) rfl (by decide)
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
  have e_1144_14 := arithEq_of_rows h (row := 1144) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_14
  simp only [← c70674, ← c70675, ← c70676] at e_1144_14
  have e_1144_15 := arithEq_of_rows h (row := 1144) (i := 15) rfl (by decide)
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
  have e_1144_16 := arithEq_of_rows h (row := 1144) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_16
  simp only [← c70680, ← c70681, ← c70682] at e_1144_16
  have e_1144_17 := arithEq_of_rows h (row := 1144) (i := 17) rfl (by decide)
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
  have e_1144_18 := arithEq_of_rows h (row := 1144) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1144_18
  simp only [← c70686, ← c70687, ← c70688] at e_1144_18
  have e_1144_19 := arithEq_of_rows h (row := 1144) (i := 19) rfl (by decide)
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
  have e_1145_0 := arithEq_of_rows h (row := 1145) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_0
  simp only [← c70692, ← c70693, ← c70694] at e_1145_0
  have e_1145_1 := arithEq_of_rows h (row := 1145) (i := 1) rfl (by decide)
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
  have e_1145_2 := arithEq_of_rows h (row := 1145) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_2
  simp only [← c70698, ← c70699, ← c70700] at e_1145_2
  have e_1145_3 := arithEq_of_rows h (row := 1145) (i := 3) rfl (by decide)
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
  have e_1145_4 := arithEq_of_rows h (row := 1145) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_4
  simp only [← c70704, ← c70705, ← c70706] at e_1145_4
  have e_1145_5 := arithEq_of_rows h (row := 1145) (i := 5) rfl (by decide)
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
  have e_1145_6 := arithEq_of_rows h (row := 1145) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_6
  simp only [← c70710, ← c70711, ← c70712] at e_1145_6
  have e_1145_7 := arithEq_of_rows h (row := 1145) (i := 7) rfl (by decide)
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
  have e_1145_8 := arithEq_of_rows h (row := 1145) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_8
  simp only [← c70716, ← c70717, ← c70718] at e_1145_8
  have e_1145_9 := arithEq_of_rows h (row := 1145) (i := 9) rfl (by decide)
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
  have e_1145_10 := arithEq_of_rows h (row := 1145) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_10
  simp only [← c70722, ← c70723, ← c70724] at e_1145_10
  have e_1145_11 := arithEq_of_rows h (row := 1145) (i := 11) rfl (by decide)
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
  have e_1145_12 := arithEq_of_rows h (row := 1145) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_12
  simp only [← c70728, ← c70729, ← c70730] at e_1145_12
  have e_1145_13 := arithEq_of_rows h (row := 1145) (i := 13) rfl (by decide)
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
  have e_1145_14 := arithEq_of_rows h (row := 1145) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_14
  simp only [← c70734, ← c70735, ← c70736] at e_1145_14
  have e_1145_15 := arithEq_of_rows h (row := 1145) (i := 15) rfl (by decide)
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
  have e_1145_16 := arithEq_of_rows h (row := 1145) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_16
  simp only [← c70740, ← c70741, ← c70742] at e_1145_16
  have e_1145_17 := arithEq_of_rows h (row := 1145) (i := 17) rfl (by decide)
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
  have e_1145_18 := arithEq_of_rows h (row := 1145) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1145_18
  simp only [← c70746, ← c70747, ← c70748] at e_1145_18
  have e_1145_19 := arithEq_of_rows h (row := 1145) (i := 19) rfl (by decide)
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
  have e_1146_0 := arithEq_of_rows h (row := 1146) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_0
  simp only [← c70752, ← c70753, ← c70754] at e_1146_0
  have e_1146_1 := arithEq_of_rows h (row := 1146) (i := 1) rfl (by decide)
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
  have e_1146_2 := arithEq_of_rows h (row := 1146) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_2
  simp only [← c70758, ← c70759, ← c70760] at e_1146_2
  have e_1146_3 := arithEq_of_rows h (row := 1146) (i := 3) rfl (by decide)
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
  have e_1146_4 := arithEq_of_rows h (row := 1146) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_4
  simp only [← c70764, ← c70765, ← c70766] at e_1146_4
  have e_1146_5 := arithEq_of_rows h (row := 1146) (i := 5) rfl (by decide)
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
  have e_1146_6 := arithEq_of_rows h (row := 1146) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_6
  simp only [← c70770, ← c70771, ← c70772] at e_1146_6
  have e_1146_7 := arithEq_of_rows h (row := 1146) (i := 7) rfl (by decide)
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
  have e_1146_8 := arithEq_of_rows h (row := 1146) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_8
  simp only [← c70776, ← c70777, ← c70778] at e_1146_8
  have e_1146_9 := arithEq_of_rows h (row := 1146) (i := 9) rfl (by decide)
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
  have e_1146_10 := arithEq_of_rows h (row := 1146) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_10
  simp only [← c70782, ← c70783, ← c70784] at e_1146_10
  have e_1146_11 := arithEq_of_rows h (row := 1146) (i := 11) rfl (by decide)
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
  have e_1146_12 := arithEq_of_rows h (row := 1146) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_12
  simp only [← c70788, ← c70789, ← c70790] at e_1146_12
  have e_1146_13 := arithEq_of_rows h (row := 1146) (i := 13) rfl (by decide)
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
  have e_1146_14 := arithEq_of_rows h (row := 1146) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_14
  simp only [← c70794, ← c70795, ← c70796] at e_1146_14
  have e_1146_15 := arithEq_of_rows h (row := 1146) (i := 15) rfl (by decide)
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
  have e_1146_16 := arithEq_of_rows h (row := 1146) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_16
  simp only [← c70800, ← c70801, ← c70802] at e_1146_16
  have e_1146_17 := arithEq_of_rows h (row := 1146) (i := 17) rfl (by decide)
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
  have e_1146_18 := arithEq_of_rows h (row := 1146) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1146_18
  simp only [← c70806, ← c70807, ← c70808] at e_1146_18
  have e_1146_19 := arithEq_of_rows h (row := 1146) (i := 19) rfl (by decide)
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
  have e_1147_0 := arithEq_of_rows h (row := 1147) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_0
  simp only [← c70812, ← c70813, ← c70814] at e_1147_0
  have e_1147_1 := arithEq_of_rows h (row := 1147) (i := 1) rfl (by decide)
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
  have e_1147_2 := arithEq_of_rows h (row := 1147) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_2
  simp only [← c70818, ← c70819, ← c70820] at e_1147_2
  have e_1147_3 := arithEq_of_rows h (row := 1147) (i := 3) rfl (by decide)
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
  have e_1147_4 := arithEq_of_rows h (row := 1147) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_4
  simp only [← c70824, ← c70825, ← c70826] at e_1147_4
  have e_1147_5 := arithEq_of_rows h (row := 1147) (i := 5) rfl (by decide)
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
  have e_1147_6 := arithEq_of_rows h (row := 1147) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_6
  simp only [← c70830, ← c70831, ← c70832] at e_1147_6
  have e_1147_7 := arithEq_of_rows h (row := 1147) (i := 7) rfl (by decide)
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
  have e_1147_8 := arithEq_of_rows h (row := 1147) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_8
  simp only [← c70836, ← c70837, ← c70838] at e_1147_8
  have e_1147_9 := arithEq_of_rows h (row := 1147) (i := 9) rfl (by decide)
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
  have e_1147_10 := arithEq_of_rows h (row := 1147) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_10
  simp only [← c70842, ← c70843, ← c70844] at e_1147_10
  have e_1147_11 := arithEq_of_rows h (row := 1147) (i := 11) rfl (by decide)
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
  have e_1147_12 := arithEq_of_rows h (row := 1147) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_12
  simp only [← c70848, ← c70849, ← c70850] at e_1147_12
  have e_1147_13 := arithEq_of_rows h (row := 1147) (i := 13) rfl (by decide)
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
  have e_1147_14 := arithEq_of_rows h (row := 1147) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_14
  simp only [← c70854, ← c70855, ← c70856] at e_1147_14
  have e_1147_15 := arithEq_of_rows h (row := 1147) (i := 15) rfl (by decide)
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
  have e_1147_16 := arithEq_of_rows h (row := 1147) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_16
  simp only [← c70860, ← c70861, ← c70862] at e_1147_16
  have e_1147_17 := arithEq_of_rows h (row := 1147) (i := 17) rfl (by decide)
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
  have e_1147_18 := arithEq_of_rows h (row := 1147) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1147_18
  simp only [← c70866, ← c70867, ← c70868] at e_1147_18
  have e_1147_19 := arithEq_of_rows h (row := 1147) (i := 19) rfl (by decide)
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
  have e_1148_0 := arithEq_of_rows h (row := 1148) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_0
  simp only [← c70872, ← c70873, ← c70874] at e_1148_0
  have e_1148_1 := arithEq_of_rows h (row := 1148) (i := 1) rfl (by decide)
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
  have e_1148_2 := arithEq_of_rows h (row := 1148) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_2
  simp only [← c70878, ← c70879, ← c70880] at e_1148_2
  have e_1148_3 := arithEq_of_rows h (row := 1148) (i := 3) rfl (by decide)
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
  have e_1148_4 := arithEq_of_rows h (row := 1148) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_4
  simp only [← c70884, ← c70885, ← c70886] at e_1148_4
  have e_1148_5 := arithEq_of_rows h (row := 1148) (i := 5) rfl (by decide)
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
  have e_1148_6 := arithEq_of_rows h (row := 1148) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_6
  simp only [← c70890, ← c70891, ← c70892] at e_1148_6
  have e_1148_7 := arithEq_of_rows h (row := 1148) (i := 7) rfl (by decide)
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
  have e_1148_8 := arithEq_of_rows h (row := 1148) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_8
  simp only [← c70896, ← c70897, ← c70898] at e_1148_8
  have e_1148_9 := arithEq_of_rows h (row := 1148) (i := 9) rfl (by decide)
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
  have e_1148_10 := arithEq_of_rows h (row := 1148) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_10
  simp only [← c70902, ← c70903, ← c70904] at e_1148_10
  have e_1148_11 := arithEq_of_rows h (row := 1148) (i := 11) rfl (by decide)
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
  have e_1148_12 := arithEq_of_rows h (row := 1148) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_12
  simp only [← c70908, ← c70909, ← c70910] at e_1148_12
  have e_1148_13 := arithEq_of_rows h (row := 1148) (i := 13) rfl (by decide)
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
  have e_1148_14 := arithEq_of_rows h (row := 1148) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_14
  simp only [← c70914, ← c70915, ← c70916] at e_1148_14
  have e_1148_15 := arithEq_of_rows h (row := 1148) (i := 15) rfl (by decide)
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
  have e_1148_16 := arithEq_of_rows h (row := 1148) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_16
  simp only [← c70920, ← c70921, ← c70922] at e_1148_16
  have e_1148_17 := arithEq_of_rows h (row := 1148) (i := 17) rfl (by decide)
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
  have e_1148_18 := arithEq_of_rows h (row := 1148) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1148_18
  simp only [← c70926, ← c70927, ← c70928] at e_1148_18
  have e_1148_19 := arithEq_of_rows h (row := 1148) (i := 19) rfl (by decide)
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
  have e_1149_0 := arithEq_of_rows h (row := 1149) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_0
  simp only [← c70932, ← c70933, ← c70934] at e_1149_0
  have e_1149_1 := arithEq_of_rows h (row := 1149) (i := 1) rfl (by decide)
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
  have e_1149_2 := arithEq_of_rows h (row := 1149) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_2
  simp only [← c70938, ← c70939, ← c70940] at e_1149_2
  have e_1149_3 := arithEq_of_rows h (row := 1149) (i := 3) rfl (by decide)
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
  have e_1149_4 := arithEq_of_rows h (row := 1149) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_4
  simp only [← c70944, ← c70945, ← c70946] at e_1149_4
  have e_1149_5 := arithEq_of_rows h (row := 1149) (i := 5) rfl (by decide)
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
  have e_1149_6 := arithEq_of_rows h (row := 1149) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_6
  simp only [← c70950, ← c70951, ← c70952] at e_1149_6
  have e_1149_7 := arithEq_of_rows h (row := 1149) (i := 7) rfl (by decide)
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
  have e_1149_8 := arithEq_of_rows h (row := 1149) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_8
  simp only [← c70956, ← c70957, ← c70958] at e_1149_8
  have e_1149_9 := arithEq_of_rows h (row := 1149) (i := 9) rfl (by decide)
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
  have e_1149_10 := arithEq_of_rows h (row := 1149) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_10
  simp only [← c70962, ← c70963, ← c70964] at e_1149_10
  have e_1149_11 := arithEq_of_rows h (row := 1149) (i := 11) rfl (by decide)
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
  have e_1149_12 := arithEq_of_rows h (row := 1149) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_12
  simp only [← c70968, ← c70969, ← c70970] at e_1149_12
  have e_1149_13 := arithEq_of_rows h (row := 1149) (i := 13) rfl (by decide)
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
  have e_1149_14 := arithEq_of_rows h (row := 1149) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_14
  simp only [← c70974, ← c70975, ← c70976] at e_1149_14
  have e_1149_15 := arithEq_of_rows h (row := 1149) (i := 15) rfl (by decide)
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
  have e_1149_16 := arithEq_of_rows h (row := 1149) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_16
  simp only [← c70980, ← c70981, ← c70982] at e_1149_16
  have e_1149_17 := arithEq_of_rows h (row := 1149) (i := 17) rfl (by decide)
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
  have e_1149_18 := arithEq_of_rows h (row := 1149) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1149_18
  simp only [← c70986, ← c70987, ← c70988] at e_1149_18
  have e_1149_19 := arithEq_of_rows h (row := 1149) (i := 19) rfl (by decide)
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
  have e_1150_0 := arithEq_of_rows h (row := 1150) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_0
  simp only [← c70992, ← c70993, ← c70994] at e_1150_0
  have e_1150_1 := arithEq_of_rows h (row := 1150) (i := 1) rfl (by decide)
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
  have e_1150_2 := arithEq_of_rows h (row := 1150) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_2
  simp only [← c70998, ← c70999, ← c71000] at e_1150_2
  have e_1150_3 := arithEq_of_rows h (row := 1150) (i := 3) rfl (by decide)
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
  have e_1150_4 := arithEq_of_rows h (row := 1150) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_4
  simp only [← c71004, ← c71005, ← c71006] at e_1150_4
  have e_1150_5 := arithEq_of_rows h (row := 1150) (i := 5) rfl (by decide)
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
  have e_1150_6 := arithEq_of_rows h (row := 1150) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_6
  simp only [← c71010, ← c71011, ← c71012] at e_1150_6
  have e_1150_7 := arithEq_of_rows h (row := 1150) (i := 7) rfl (by decide)
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
  have e_1150_8 := arithEq_of_rows h (row := 1150) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_8
  simp only [← c71016, ← c71017, ← c71018] at e_1150_8
  have e_1150_9 := arithEq_of_rows h (row := 1150) (i := 9) rfl (by decide)
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
  have e_1150_10 := arithEq_of_rows h (row := 1150) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_10
  simp only [← c71022, ← c71023, ← c71024] at e_1150_10
  have e_1150_11 := arithEq_of_rows h (row := 1150) (i := 11) rfl (by decide)
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
  have e_1150_12 := arithEq_of_rows h (row := 1150) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_12
  simp only [← c71028, ← c71029, ← c71030] at e_1150_12
  have e_1150_13 := arithEq_of_rows h (row := 1150) (i := 13) rfl (by decide)
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
  have e_1150_14 := arithEq_of_rows h (row := 1150) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_14
  simp only [← c71034, ← c71035, ← c71036] at e_1150_14
  have e_1150_15 := arithEq_of_rows h (row := 1150) (i := 15) rfl (by decide)
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
  have e_1150_16 := arithEq_of_rows h (row := 1150) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_16
  simp only [← c71040, ← c71041, ← c71042] at e_1150_16
  have e_1150_17 := arithEq_of_rows h (row := 1150) (i := 17) rfl (by decide)
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
  have e_1150_18 := arithEq_of_rows h (row := 1150) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1150_18
  simp only [← c71046, ← c71047, ← c71048] at e_1150_18
  have e_1150_19 := arithEq_of_rows h (row := 1150) (i := 19) rfl (by decide)
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
  have e_1151_0 := arithEq_of_rows h (row := 1151) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_0
  simp only [← c71052, ← c71053, ← c71054] at e_1151_0
  have e_1151_1 := arithEq_of_rows h (row := 1151) (i := 1) rfl (by decide)
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
  have e_1151_2 := arithEq_of_rows h (row := 1151) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_2
  simp only [← c71058, ← c71059, ← c71060] at e_1151_2
  have e_1151_3 := arithEq_of_rows h (row := 1151) (i := 3) rfl (by decide)
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
  have e_1151_4 := arithEq_of_rows h (row := 1151) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_4
  simp only [← c71064, ← c71065, ← c71066] at e_1151_4
  have e_1151_5 := arithEq_of_rows h (row := 1151) (i := 5) rfl (by decide)
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
  have e_1151_6 := arithEq_of_rows h (row := 1151) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_6
  simp only [← c71070, ← c71071, ← c71072] at e_1151_6
  have e_1151_7 := arithEq_of_rows h (row := 1151) (i := 7) rfl (by decide)
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
  have e_1151_8 := arithEq_of_rows h (row := 1151) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_8
  simp only [← c71076, ← c71077, ← c71078] at e_1151_8
  have e_1151_9 := arithEq_of_rows h (row := 1151) (i := 9) rfl (by decide)
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
  have e_1151_10 := arithEq_of_rows h (row := 1151) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_10
  simp only [← c71082, ← c71083, ← c71084] at e_1151_10
  have e_1151_11 := arithEq_of_rows h (row := 1151) (i := 11) rfl (by decide)
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
  have e_1151_12 := arithEq_of_rows h (row := 1151) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_12
  simp only [← c71088, ← c71089, ← c71090] at e_1151_12
  have e_1151_13 := arithEq_of_rows h (row := 1151) (i := 13) rfl (by decide)
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
  have e_1151_14 := arithEq_of_rows h (row := 1151) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_14
  simp only [← c71094, ← c71095, ← c71096] at e_1151_14
  have e_1151_15 := arithEq_of_rows h (row := 1151) (i := 15) rfl (by decide)
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
  have e_1151_16 := arithEq_of_rows h (row := 1151) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_16
  simp only [← c71100, ← c71101, ← c71102] at e_1151_16
  have e_1151_17 := arithEq_of_rows h (row := 1151) (i := 17) rfl (by decide)
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
  have e_1151_18 := arithEq_of_rows h (row := 1151) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1151_18
  simp only [← c71106, ← c71107, ← c71108] at e_1151_18
  have e_1151_19 := arithEq_of_rows h (row := 1151) (i := 19) rfl (by decide)
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
  have e_1152_0 := arithEq_of_rows h (row := 1152) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_0
  simp only [← c71112, ← c71113, ← c71114] at e_1152_0
  have e_1152_1 := arithEq_of_rows h (row := 1152) (i := 1) rfl (by decide)
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
  have e_1152_2 := arithEq_of_rows h (row := 1152) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_2
  simp only [← c71118, ← c71119, ← c71120] at e_1152_2
  have e_1152_3 := arithEq_of_rows h (row := 1152) (i := 3) rfl (by decide)
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
  have e_1152_4 := arithEq_of_rows h (row := 1152) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_4
  simp only [← c71124, ← c71125, ← c71126] at e_1152_4
  have e_1152_5 := arithEq_of_rows h (row := 1152) (i := 5) rfl (by decide)
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
  have e_1152_6 := arithEq_of_rows h (row := 1152) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_6
  simp only [← c71130, ← c71131, ← c71132] at e_1152_6
  have e_1152_7 := arithEq_of_rows h (row := 1152) (i := 7) rfl (by decide)
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
  have e_1152_8 := arithEq_of_rows h (row := 1152) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_8
  simp only [← c71136, ← c71137, ← c71138] at e_1152_8
  have e_1152_9 := arithEq_of_rows h (row := 1152) (i := 9) rfl (by decide)
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
  have e_1152_10 := arithEq_of_rows h (row := 1152) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_10
  simp only [← c71142, ← c71143, ← c71144] at e_1152_10
  have e_1152_11 := arithEq_of_rows h (row := 1152) (i := 11) rfl (by decide)
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
  have e_1152_12 := arithEq_of_rows h (row := 1152) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_12
  simp only [← c71148, ← c71149, ← c71150] at e_1152_12
  have e_1152_13 := arithEq_of_rows h (row := 1152) (i := 13) rfl (by decide)
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
  have e_1152_14 := arithEq_of_rows h (row := 1152) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_14
  simp only [← c71154, ← c71155, ← c71156] at e_1152_14
  have e_1152_15 := arithEq_of_rows h (row := 1152) (i := 15) rfl (by decide)
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
  have e_1152_16 := arithEq_of_rows h (row := 1152) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_16
  simp only [← c71160, ← c71161, ← c71162] at e_1152_16
  have e_1152_17 := arithEq_of_rows h (row := 1152) (i := 17) rfl (by decide)
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
  have e_1152_18 := arithEq_of_rows h (row := 1152) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1152_18
  simp only [← c71166, ← c71167, ← c71168] at e_1152_18
  have e_1152_19 := arithEq_of_rows h (row := 1152) (i := 19) rfl (by decide)
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
  have e_1153_0 := arithEq_of_rows h (row := 1153) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_0
  simp only [← c71172, ← c71173, ← c71174] at e_1153_0
  have e_1153_1 := arithEq_of_rows h (row := 1153) (i := 1) rfl (by decide)
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
  have e_1153_2 := arithEq_of_rows h (row := 1153) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_2
  simp only [← c71178, ← c71179, ← c71180] at e_1153_2
  have e_1153_3 := arithEq_of_rows h (row := 1153) (i := 3) rfl (by decide)
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
  have e_1153_4 := arithEq_of_rows h (row := 1153) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_4
  simp only [← c71184, ← c71185, ← c71186] at e_1153_4
  have e_1153_5 := arithEq_of_rows h (row := 1153) (i := 5) rfl (by decide)
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
  have e_1153_6 := arithEq_of_rows h (row := 1153) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_6
  simp only [← c71190, ← c71191, ← c71192] at e_1153_6
  have e_1153_7 := arithEq_of_rows h (row := 1153) (i := 7) rfl (by decide)
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
  have e_1153_8 := arithEq_of_rows h (row := 1153) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_8
  simp only [← c71196, ← c71197, ← c71198] at e_1153_8
  have e_1153_9 := arithEq_of_rows h (row := 1153) (i := 9) rfl (by decide)
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
  have e_1153_10 := arithEq_of_rows h (row := 1153) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_10
  simp only [← c71202, ← c71203, ← c71204] at e_1153_10
  have e_1153_11 := arithEq_of_rows h (row := 1153) (i := 11) rfl (by decide)
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
  have e_1153_12 := arithEq_of_rows h (row := 1153) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_12
  simp only [← c71208, ← c71209, ← c71210] at e_1153_12
  have e_1153_13 := arithEq_of_rows h (row := 1153) (i := 13) rfl (by decide)
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
  have e_1153_14 := arithEq_of_rows h (row := 1153) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_14
  simp only [← c71214, ← c71215, ← c71216] at e_1153_14
  have e_1153_15 := arithEq_of_rows h (row := 1153) (i := 15) rfl (by decide)
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
  have e_1153_16 := arithEq_of_rows h (row := 1153) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_16
  simp only [← c71220, ← c71221, ← c71222] at e_1153_16
  have e_1153_17 := arithEq_of_rows h (row := 1153) (i := 17) rfl (by decide)
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
  have e_1153_18 := arithEq_of_rows h (row := 1153) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1153_18
  simp only [← c71226, ← c71227, ← c71228] at e_1153_18
  have e_1153_19 := arithEq_of_rows h (row := 1153) (i := 19) rfl (by decide)
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
  have e_1154_0 := arithEq_of_rows h (row := 1154) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_0
  simp only [← c71232, ← c71233, ← c71234] at e_1154_0
  have e_1154_1 := arithEq_of_rows h (row := 1154) (i := 1) rfl (by decide)
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
  have e_1154_2 := arithEq_of_rows h (row := 1154) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_2
  simp only [← c71238, ← c71239, ← c71240] at e_1154_2
  have e_1154_3 := arithEq_of_rows h (row := 1154) (i := 3) rfl (by decide)
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
  have e_1154_4 := arithEq_of_rows h (row := 1154) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_4
  simp only [← c71244, ← c71245, ← c71246] at e_1154_4
  have e_1154_5 := arithEq_of_rows h (row := 1154) (i := 5) rfl (by decide)
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
  have e_1154_6 := arithEq_of_rows h (row := 1154) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_6
  simp only [← c71250, ← c71251, ← c71252] at e_1154_6
  have e_1154_7 := arithEq_of_rows h (row := 1154) (i := 7) rfl (by decide)
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
  have e_1154_8 := arithEq_of_rows h (row := 1154) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_8
  simp only [← c71256, ← c71257, ← c71258] at e_1154_8
  have e_1154_9 := arithEq_of_rows h (row := 1154) (i := 9) rfl (by decide)
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
  have e_1154_10 := arithEq_of_rows h (row := 1154) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_10
  simp only [← c71262, ← c71263, ← c71264] at e_1154_10
  have e_1154_11 := arithEq_of_rows h (row := 1154) (i := 11) rfl (by decide)
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
  have e_1154_12 := arithEq_of_rows h (row := 1154) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_12
  simp only [← c71268, ← c71269, ← c71270] at e_1154_12
  have e_1154_13 := arithEq_of_rows h (row := 1154) (i := 13) rfl (by decide)
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
  have e_1154_14 := arithEq_of_rows h (row := 1154) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_14
  simp only [← c71274, ← c71275, ← c71276] at e_1154_14
  have e_1154_15 := arithEq_of_rows h (row := 1154) (i := 15) rfl (by decide)
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
  have e_1154_16 := arithEq_of_rows h (row := 1154) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_16
  simp only [← c71280, ← c71281, ← c71282] at e_1154_16
  have e_1154_17 := arithEq_of_rows h (row := 1154) (i := 17) rfl (by decide)
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
  have e_1154_18 := arithEq_of_rows h (row := 1154) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1154_18
  simp only [← c71286, ← c71287, ← c71288] at e_1154_18
  have e_1154_19 := arithEq_of_rows h (row := 1154) (i := 19) rfl (by decide)
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
  have e_1155_0 := arithEq_of_rows h (row := 1155) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_0
  simp only [← c71292, ← c71293, ← c71294] at e_1155_0
  have e_1155_1 := arithEq_of_rows h (row := 1155) (i := 1) rfl (by decide)
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
  have e_1155_2 := arithEq_of_rows h (row := 1155) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_2
  simp only [← c71298, ← c71299, ← c71300] at e_1155_2
  have e_1155_3 := arithEq_of_rows h (row := 1155) (i := 3) rfl (by decide)
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
  have e_1155_4 := arithEq_of_rows h (row := 1155) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_4
  simp only [← c71304, ← c71305, ← c71306] at e_1155_4
  have e_1155_5 := arithEq_of_rows h (row := 1155) (i := 5) rfl (by decide)
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
  have e_1155_6 := arithEq_of_rows h (row := 1155) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_6
  simp only [← c71310, ← c71311, ← c71312] at e_1155_6
  have e_1155_7 := arithEq_of_rows h (row := 1155) (i := 7) rfl (by decide)
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
  have e_1155_8 := arithEq_of_rows h (row := 1155) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_8
  simp only [← c71316, ← c71317, ← c71318] at e_1155_8
  have e_1155_9 := arithEq_of_rows h (row := 1155) (i := 9) rfl (by decide)
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
  have e_1155_10 := arithEq_of_rows h (row := 1155) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_10
  simp only [← c71322, ← c71323, ← c71324] at e_1155_10
  have e_1155_11 := arithEq_of_rows h (row := 1155) (i := 11) rfl (by decide)
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
  have e_1155_12 := arithEq_of_rows h (row := 1155) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_12
  simp only [← c71328, ← c71329, ← c71330] at e_1155_12
  have e_1155_13 := arithEq_of_rows h (row := 1155) (i := 13) rfl (by decide)
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
  have e_1155_14 := arithEq_of_rows h (row := 1155) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_14
  simp only [← c71334, ← c71335, ← c71336] at e_1155_14
  have e_1155_15 := arithEq_of_rows h (row := 1155) (i := 15) rfl (by decide)
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
  have e_1155_16 := arithEq_of_rows h (row := 1155) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_16
  simp only [← c71340, ← c71341, ← c71342] at e_1155_16
  have e_1155_17 := arithEq_of_rows h (row := 1155) (i := 17) rfl (by decide)
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
  have e_1155_18 := arithEq_of_rows h (row := 1155) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1155_18
  simp only [← c71346, ← c71347, ← c71348] at e_1155_18
  have e_1155_19 := arithEq_of_rows h (row := 1155) (i := 19) rfl (by decide)
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
  have e_1156_0 := arithEq_of_rows h (row := 1156) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_0
  simp only [← c71352, ← c71353, ← c71354] at e_1156_0
  have e_1156_1 := arithEq_of_rows h (row := 1156) (i := 1) rfl (by decide)
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
  have e_1156_2 := arithEq_of_rows h (row := 1156) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_2
  simp only [← c71358, ← c71359, ← c71360] at e_1156_2
  have e_1156_3 := arithEq_of_rows h (row := 1156) (i := 3) rfl (by decide)
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
  have e_1156_4 := arithEq_of_rows h (row := 1156) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_4
  simp only [← c71364, ← c71365, ← c71366] at e_1156_4
  have e_1156_5 := arithEq_of_rows h (row := 1156) (i := 5) rfl (by decide)
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
  have e_1156_6 := arithEq_of_rows h (row := 1156) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_6
  simp only [← c71370, ← c71371, ← c71372] at e_1156_6
  have e_1156_7 := arithEq_of_rows h (row := 1156) (i := 7) rfl (by decide)
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
  have e_1156_8 := arithEq_of_rows h (row := 1156) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_8
  simp only [← c71376, ← c71377, ← c71378] at e_1156_8
  have e_1156_9 := arithEq_of_rows h (row := 1156) (i := 9) rfl (by decide)
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
  have e_1156_10 := arithEq_of_rows h (row := 1156) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_10
  simp only [← c71382, ← c71383, ← c71384] at e_1156_10
  have e_1156_11 := arithEq_of_rows h (row := 1156) (i := 11) rfl (by decide)
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
  have e_1156_12 := arithEq_of_rows h (row := 1156) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_12
  simp only [← c71388, ← c71389, ← c71390] at e_1156_12
  have e_1156_13 := arithEq_of_rows h (row := 1156) (i := 13) rfl (by decide)
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
  have e_1156_14 := arithEq_of_rows h (row := 1156) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_14
  simp only [← c71394, ← c71395, ← c71396] at e_1156_14
  have e_1156_15 := arithEq_of_rows h (row := 1156) (i := 15) rfl (by decide)
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
  have e_1156_16 := arithEq_of_rows h (row := 1156) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_16
  simp only [← c71400, ← c71401, ← c71402] at e_1156_16
  have e_1156_17 := arithEq_of_rows h (row := 1156) (i := 17) rfl (by decide)
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
  have e_1156_18 := arithEq_of_rows h (row := 1156) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1156_18
  simp only [← c71406, ← c71407, ← c71408] at e_1156_18
  have e_1156_19 := arithEq_of_rows h (row := 1156) (i := 19) rfl (by decide)
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
  have e_1157_0 := arithEq_of_rows h (row := 1157) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_0
  simp only [← c71412, ← c71413, ← c71414] at e_1157_0
  have e_1157_1 := arithEq_of_rows h (row := 1157) (i := 1) rfl (by decide)
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
  have e_1157_2 := arithEq_of_rows h (row := 1157) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_2
  simp only [← c71418, ← c71419, ← c71420] at e_1157_2
  have e_1157_3 := arithEq_of_rows h (row := 1157) (i := 3) rfl (by decide)
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
  have e_1157_4 := arithEq_of_rows h (row := 1157) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_4
  simp only [← c71424, ← c71425, ← c71426] at e_1157_4
  have e_1157_5 := arithEq_of_rows h (row := 1157) (i := 5) rfl (by decide)
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
  have e_1157_6 := arithEq_of_rows h (row := 1157) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_6
  simp only [← c71430, ← c71431, ← c71432] at e_1157_6
  have e_1157_7 := arithEq_of_rows h (row := 1157) (i := 7) rfl (by decide)
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
  have e_1157_8 := arithEq_of_rows h (row := 1157) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_8
  simp only [← c71436, ← c71437, ← c71438] at e_1157_8
  have e_1157_9 := arithEq_of_rows h (row := 1157) (i := 9) rfl (by decide)
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
  have e_1157_10 := arithEq_of_rows h (row := 1157) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_10
  simp only [← c71442, ← c71443, ← c71444] at e_1157_10
  have e_1157_11 := arithEq_of_rows h (row := 1157) (i := 11) rfl (by decide)
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
  have e_1157_12 := arithEq_of_rows h (row := 1157) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_12
  simp only [← c71448, ← c71449, ← c71450] at e_1157_12
  have e_1157_13 := arithEq_of_rows h (row := 1157) (i := 13) rfl (by decide)
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
  have e_1157_14 := arithEq_of_rows h (row := 1157) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_14
  simp only [← c71454, ← c71455, ← c71456] at e_1157_14
  have e_1157_15 := arithEq_of_rows h (row := 1157) (i := 15) rfl (by decide)
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
  have e_1157_16 := arithEq_of_rows h (row := 1157) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_16
  simp only [← c71460, ← c71461, ← c71462] at e_1157_16
  have e_1157_17 := arithEq_of_rows h (row := 1157) (i := 17) rfl (by decide)
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
  have e_1157_18 := arithEq_of_rows h (row := 1157) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1157_18
  simp only [← c71466, ← c71467, ← c71468] at e_1157_18
  have e_1157_19 := arithEq_of_rows h (row := 1157) (i := 19) rfl (by decide)
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
  have e_1158_0 := arithEq_of_rows h (row := 1158) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_0
  simp only [← c71472, ← c71473, ← c71474] at e_1158_0
  have e_1158_1 := arithEq_of_rows h (row := 1158) (i := 1) rfl (by decide)
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
  have e_1158_2 := arithEq_of_rows h (row := 1158) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_2
  simp only [← c71478, ← c71479, ← c71480] at e_1158_2
  have e_1158_3 := arithEq_of_rows h (row := 1158) (i := 3) rfl (by decide)
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
  have e_1158_4 := arithEq_of_rows h (row := 1158) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_4
  simp only [← c71484, ← c71485, ← c71486] at e_1158_4
  have e_1158_5 := arithEq_of_rows h (row := 1158) (i := 5) rfl (by decide)
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
  have e_1158_6 := arithEq_of_rows h (row := 1158) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_6
  simp only [← c71490, ← c71491, ← c71492] at e_1158_6
  have e_1158_7 := arithEq_of_rows h (row := 1158) (i := 7) rfl (by decide)
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
  have e_1158_8 := arithEq_of_rows h (row := 1158) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_8
  simp only [← c71496, ← c71497, ← c71498] at e_1158_8
  have e_1158_9 := arithEq_of_rows h (row := 1158) (i := 9) rfl (by decide)
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
  have e_1158_10 := arithEq_of_rows h (row := 1158) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_10
  simp only [← c71502, ← c71503, ← c71504] at e_1158_10
  have e_1158_11 := arithEq_of_rows h (row := 1158) (i := 11) rfl (by decide)
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
  have e_1158_12 := arithEq_of_rows h (row := 1158) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_12
  simp only [← c71508, ← c71509, ← c71510] at e_1158_12
  have e_1158_13 := arithEq_of_rows h (row := 1158) (i := 13) rfl (by decide)
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
  have e_1158_14 := arithEq_of_rows h (row := 1158) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_14
  simp only [← c71514, ← c71515, ← c71516] at e_1158_14
  have e_1158_15 := arithEq_of_rows h (row := 1158) (i := 15) rfl (by decide)
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
  have e_1158_16 := arithEq_of_rows h (row := 1158) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_16
  simp only [← c71520, ← c71521, ← c71522] at e_1158_16
  have e_1158_17 := arithEq_of_rows h (row := 1158) (i := 17) rfl (by decide)
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
  have e_1158_18 := arithEq_of_rows h (row := 1158) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1158_18
  simp only [← c71526, ← c71527, ← c71528] at e_1158_18
  have e_1158_19 := arithEq_of_rows h (row := 1158) (i := 19) rfl (by decide)
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
  have e_1159_0 := arithEq_of_rows h (row := 1159) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_0
  simp only [← c71532, ← c71533, ← c71534] at e_1159_0
  have e_1159_1 := arithEq_of_rows h (row := 1159) (i := 1) rfl (by decide)
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
  have e_1159_2 := arithEq_of_rows h (row := 1159) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_2
  simp only [← c71538, ← c71539, ← c71540] at e_1159_2
  have e_1159_3 := arithEq_of_rows h (row := 1159) (i := 3) rfl (by decide)
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
  have e_1159_4 := arithEq_of_rows h (row := 1159) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_4
  simp only [← c71544, ← c71545, ← c71546] at e_1159_4
  have e_1159_5 := arithEq_of_rows h (row := 1159) (i := 5) rfl (by decide)
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
  have e_1159_6 := arithEq_of_rows h (row := 1159) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_6
  simp only [← c71550, ← c71551, ← c71552] at e_1159_6
  have e_1159_7 := arithEq_of_rows h (row := 1159) (i := 7) rfl (by decide)
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
  have e_1159_8 := arithEq_of_rows h (row := 1159) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_8
  simp only [← c71556, ← c71557, ← c71558] at e_1159_8
  have e_1159_9 := arithEq_of_rows h (row := 1159) (i := 9) rfl (by decide)
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
  have e_1159_10 := arithEq_of_rows h (row := 1159) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_10
  simp only [← c71562, ← c71563, ← c71564] at e_1159_10
  have e_1159_11 := arithEq_of_rows h (row := 1159) (i := 11) rfl (by decide)
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
  have e_1159_12 := arithEq_of_rows h (row := 1159) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_12
  simp only [← c71568, ← c71569, ← c71570] at e_1159_12
  have e_1159_13 := arithEq_of_rows h (row := 1159) (i := 13) rfl (by decide)
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
  have e_1159_14 := arithEq_of_rows h (row := 1159) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_14
  simp only [← c71574, ← c71575, ← c71576] at e_1159_14
  have e_1159_15 := arithEq_of_rows h (row := 1159) (i := 15) rfl (by decide)
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
  have e_1159_16 := arithEq_of_rows h (row := 1159) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_16
  simp only [← c71580, ← c71581, ← c71582] at e_1159_16
  have e_1159_17 := arithEq_of_rows h (row := 1159) (i := 17) rfl (by decide)
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
  have e_1159_18 := arithEq_of_rows h (row := 1159) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1159_18
  simp only [← c71586, ← c71587, ← c71588] at e_1159_18
  have e_1159_19 := arithEq_of_rows h (row := 1159) (i := 19) rfl (by decide)
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
  have e_1160_0 := arithEq_of_rows h (row := 1160) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_0
  simp only [← c71592, ← c71593, ← c71594] at e_1160_0
  have e_1160_1 := arithEq_of_rows h (row := 1160) (i := 1) rfl (by decide)
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
  have e_1160_2 := arithEq_of_rows h (row := 1160) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_2
  simp only [← c71598, ← c71599, ← c71600] at e_1160_2
  have e_1160_3 := arithEq_of_rows h (row := 1160) (i := 3) rfl (by decide)
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
  have e_1160_4 := arithEq_of_rows h (row := 1160) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_4
  simp only [← c71604, ← c71605, ← c71606] at e_1160_4
  have e_1160_5 := arithEq_of_rows h (row := 1160) (i := 5) rfl (by decide)
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
  have e_1160_6 := arithEq_of_rows h (row := 1160) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_6
  simp only [← c71610, ← c71611, ← c71612] at e_1160_6
  have e_1160_7 := arithEq_of_rows h (row := 1160) (i := 7) rfl (by decide)
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
  have e_1160_8 := arithEq_of_rows h (row := 1160) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_8
  simp only [← c71616, ← c71617, ← c71618] at e_1160_8
  have e_1160_9 := arithEq_of_rows h (row := 1160) (i := 9) rfl (by decide)
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
  have e_1160_10 := arithEq_of_rows h (row := 1160) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_10
  simp only [← c71622, ← c71623, ← c71624] at e_1160_10
  have e_1160_11 := arithEq_of_rows h (row := 1160) (i := 11) rfl (by decide)
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
  have e_1160_12 := arithEq_of_rows h (row := 1160) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_12
  simp only [← c71628, ← c71629, ← c71630] at e_1160_12
  have e_1160_13 := arithEq_of_rows h (row := 1160) (i := 13) rfl (by decide)
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
  have e_1160_14 := arithEq_of_rows h (row := 1160) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_14
  simp only [← c71634, ← c71635, ← c71636] at e_1160_14
  have e_1160_15 := arithEq_of_rows h (row := 1160) (i := 15) rfl (by decide)
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
  have e_1160_16 := arithEq_of_rows h (row := 1160) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_16
  simp only [← c71640, ← c71641, ← c71642] at e_1160_16
  have e_1160_17 := arithEq_of_rows h (row := 1160) (i := 17) rfl (by decide)
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
  have e_1160_18 := arithEq_of_rows h (row := 1160) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1160_18
  simp only [← c71646, ← c71647, ← c71648] at e_1160_18
  have e_1160_19 := arithEq_of_rows h (row := 1160) (i := 19) rfl (by decide)
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
  have e_1161_0 := arithEq_of_rows h (row := 1161) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_0
  simp only [← c71652, ← c71653, ← c71654] at e_1161_0
  have e_1161_1 := arithEq_of_rows h (row := 1161) (i := 1) rfl (by decide)
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
  have e_1161_2 := arithEq_of_rows h (row := 1161) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_2
  simp only [← c71658, ← c71659, ← c71660] at e_1161_2
  have e_1161_3 := arithEq_of_rows h (row := 1161) (i := 3) rfl (by decide)
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
  have e_1161_4 := arithEq_of_rows h (row := 1161) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_4
  simp only [← c71664, ← c71665, ← c71666] at e_1161_4
  have e_1161_5 := arithEq_of_rows h (row := 1161) (i := 5) rfl (by decide)
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
  have e_1161_6 := arithEq_of_rows h (row := 1161) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_6
  simp only [← c71670, ← c71671, ← c71672] at e_1161_6
  have e_1161_7 := arithEq_of_rows h (row := 1161) (i := 7) rfl (by decide)
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
  have e_1161_8 := arithEq_of_rows h (row := 1161) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_8
  simp only [← c71676, ← c71677, ← c71678] at e_1161_8
  have e_1161_9 := arithEq_of_rows h (row := 1161) (i := 9) rfl (by decide)
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
  have e_1161_10 := arithEq_of_rows h (row := 1161) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_10
  simp only [← c71682, ← c71683, ← c71684] at e_1161_10
  have e_1161_11 := arithEq_of_rows h (row := 1161) (i := 11) rfl (by decide)
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
  have e_1161_12 := arithEq_of_rows h (row := 1161) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_12
  simp only [← c71688, ← c71689, ← c71690] at e_1161_12
  have e_1161_13 := arithEq_of_rows h (row := 1161) (i := 13) rfl (by decide)
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
  have e_1161_14 := arithEq_of_rows h (row := 1161) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_14
  simp only [← c71694, ← c71695, ← c71696] at e_1161_14
  have e_1161_15 := arithEq_of_rows h (row := 1161) (i := 15) rfl (by decide)
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
  have e_1161_16 := arithEq_of_rows h (row := 1161) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_16
  simp only [← c71700, ← c71701, ← c71702] at e_1161_16
  have e_1161_17 := arithEq_of_rows h (row := 1161) (i := 17) rfl (by decide)
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
  have e_1161_18 := arithEq_of_rows h (row := 1161) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1161_18
  simp only [← c71706, ← c71707, ← c71708] at e_1161_18
  have e_1161_19 := arithEq_of_rows h (row := 1161) (i := 19) rfl (by decide)
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
  have e_1162_0 := arithEq_of_rows h (row := 1162) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_0
  simp only [← c71712, ← c71713, ← c71714] at e_1162_0
  have e_1162_1 := arithEq_of_rows h (row := 1162) (i := 1) rfl (by decide)
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
  have e_1162_2 := arithEq_of_rows h (row := 1162) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_2
  simp only [← c71718, ← c71719, ← c71720] at e_1162_2
  have e_1162_3 := arithEq_of_rows h (row := 1162) (i := 3) rfl (by decide)
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
  have e_1162_4 := arithEq_of_rows h (row := 1162) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_4
  simp only [← c71724, ← c71725, ← c71726] at e_1162_4
  have e_1162_5 := arithEq_of_rows h (row := 1162) (i := 5) rfl (by decide)
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
  have e_1162_6 := arithEq_of_rows h (row := 1162) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_6
  simp only [← c71730, ← c71731, ← c71732] at e_1162_6
  have e_1162_7 := arithEq_of_rows h (row := 1162) (i := 7) rfl (by decide)
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
  have e_1162_8 := arithEq_of_rows h (row := 1162) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_8
  simp only [← c71736, ← c71737, ← c71738] at e_1162_8
  have e_1162_9 := arithEq_of_rows h (row := 1162) (i := 9) rfl (by decide)
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
  have e_1162_10 := arithEq_of_rows h (row := 1162) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_10
  simp only [← c71742, ← c71743, ← c71744] at e_1162_10
  have e_1162_11 := arithEq_of_rows h (row := 1162) (i := 11) rfl (by decide)
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
  have e_1162_12 := arithEq_of_rows h (row := 1162) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_12
  simp only [← c71748, ← c71749, ← c71750] at e_1162_12
  have e_1162_13 := arithEq_of_rows h (row := 1162) (i := 13) rfl (by decide)
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
  have e_1162_14 := arithEq_of_rows h (row := 1162) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_14
  simp only [← c71754, ← c71755, ← c71756] at e_1162_14
  have e_1162_15 := arithEq_of_rows h (row := 1162) (i := 15) rfl (by decide)
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
  have e_1162_16 := arithEq_of_rows h (row := 1162) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_16
  simp only [← c71760, ← c71761, ← c71762] at e_1162_16
  have e_1162_17 := arithEq_of_rows h (row := 1162) (i := 17) rfl (by decide)
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
  have e_1162_18 := arithEq_of_rows h (row := 1162) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1162_18
  simp only [← c71766, ← c71767, ← c71768] at e_1162_18
  have e_1162_19 := arithEq_of_rows h (row := 1162) (i := 19) rfl (by decide)
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
  have e_1163_0 := arithEq_of_rows h (row := 1163) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_0
  simp only [← c71772, ← c71773, ← c71774] at e_1163_0
  have e_1163_1 := arithEq_of_rows h (row := 1163) (i := 1) rfl (by decide)
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
  have e_1163_2 := arithEq_of_rows h (row := 1163) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_2
  simp only [← c71778, ← c71779, ← c71780] at e_1163_2
  have e_1163_3 := arithEq_of_rows h (row := 1163) (i := 3) rfl (by decide)
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
  have e_1163_4 := arithEq_of_rows h (row := 1163) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_4
  simp only [← c71784, ← c71785, ← c71786] at e_1163_4
  have e_1163_5 := arithEq_of_rows h (row := 1163) (i := 5) rfl (by decide)
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
  have e_1163_6 := arithEq_of_rows h (row := 1163) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_6
  simp only [← c71790, ← c71791, ← c71792] at e_1163_6
  have e_1163_7 := arithEq_of_rows h (row := 1163) (i := 7) rfl (by decide)
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
  have e_1163_8 := arithEq_of_rows h (row := 1163) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_8
  simp only [← c71796, ← c71797, ← c71798] at e_1163_8
  have e_1163_9 := arithEq_of_rows h (row := 1163) (i := 9) rfl (by decide)
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
  have e_1163_10 := arithEq_of_rows h (row := 1163) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_10
  simp only [← c71802, ← c71803, ← c71804] at e_1163_10
  have e_1163_11 := arithEq_of_rows h (row := 1163) (i := 11) rfl (by decide)
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
  have e_1163_12 := arithEq_of_rows h (row := 1163) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_12
  simp only [← c71808, ← c71809, ← c71810] at e_1163_12
  have e_1163_13 := arithEq_of_rows h (row := 1163) (i := 13) rfl (by decide)
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
  have e_1163_14 := arithEq_of_rows h (row := 1163) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_14
  simp only [← c71814, ← c71815, ← c71816] at e_1163_14
  have e_1163_15 := arithEq_of_rows h (row := 1163) (i := 15) rfl (by decide)
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
  have e_1163_16 := arithEq_of_rows h (row := 1163) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_16
  simp only [← c71820, ← c71821, ← c71822] at e_1163_16
  have e_1163_17 := arithEq_of_rows h (row := 1163) (i := 17) rfl (by decide)
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
  have e_1163_18 := arithEq_of_rows h (row := 1163) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1163_18
  simp only [← c71826, ← c71827, ← c71828] at e_1163_18
  have e_1163_19 := arithEq_of_rows h (row := 1163) (i := 19) rfl (by decide)
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
  have e_1164_0 := arithEq_of_rows h (row := 1164) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_0
  simp only [← c71832, ← c71833, ← c71834] at e_1164_0
  have e_1164_1 := arithEq_of_rows h (row := 1164) (i := 1) rfl (by decide)
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
  have e_1164_2 := arithEq_of_rows h (row := 1164) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_2
  simp only [← c71838, ← c71839, ← c71840] at e_1164_2
  have e_1164_3 := arithEq_of_rows h (row := 1164) (i := 3) rfl (by decide)
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
  have e_1164_4 := arithEq_of_rows h (row := 1164) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_4
  simp only [← c71844, ← c71845, ← c71846] at e_1164_4
  have e_1164_5 := arithEq_of_rows h (row := 1164) (i := 5) rfl (by decide)
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
  have e_1164_6 := arithEq_of_rows h (row := 1164) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_6
  simp only [← c71850, ← c71851, ← c71852] at e_1164_6
  have e_1164_7 := arithEq_of_rows h (row := 1164) (i := 7) rfl (by decide)
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
  have e_1164_8 := arithEq_of_rows h (row := 1164) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_8
  simp only [← c71856, ← c71857, ← c71858] at e_1164_8
  have e_1164_9 := arithEq_of_rows h (row := 1164) (i := 9) rfl (by decide)
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
  have e_1164_10 := arithEq_of_rows h (row := 1164) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_10
  simp only [← c71862, ← c71863, ← c71864] at e_1164_10
  have e_1164_11 := arithEq_of_rows h (row := 1164) (i := 11) rfl (by decide)
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
  have e_1164_12 := arithEq_of_rows h (row := 1164) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_12
  simp only [← c71868, ← c71869, ← c71870] at e_1164_12
  have e_1164_13 := arithEq_of_rows h (row := 1164) (i := 13) rfl (by decide)
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
  have e_1164_14 := arithEq_of_rows h (row := 1164) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_14
  simp only [← c71874, ← c71875, ← c71876] at e_1164_14
  have e_1164_15 := arithEq_of_rows h (row := 1164) (i := 15) rfl (by decide)
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
  have e_1164_16 := arithEq_of_rows h (row := 1164) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_16
  simp only [← c71880, ← c71881, ← c71882] at e_1164_16
  have e_1164_17 := arithEq_of_rows h (row := 1164) (i := 17) rfl (by decide)
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
  have e_1164_18 := arithEq_of_rows h (row := 1164) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1164_18
  simp only [← c71886, ← c71887, ← c71888] at e_1164_18
  have e_1164_19 := arithEq_of_rows h (row := 1164) (i := 19) rfl (by decide)
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
  have e_1165_0 := arithEq_of_rows h (row := 1165) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_0
  simp only [← c71892, ← c71893, ← c71894] at e_1165_0
  have e_1165_1 := arithEq_of_rows h (row := 1165) (i := 1) rfl (by decide)
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
  have e_1165_2 := arithEq_of_rows h (row := 1165) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_2
  simp only [← c71898, ← c71899, ← c71900] at e_1165_2
  have e_1165_3 := arithEq_of_rows h (row := 1165) (i := 3) rfl (by decide)
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
  have e_1165_4 := arithEq_of_rows h (row := 1165) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_4
  simp only [← c71904, ← c71905, ← c71906] at e_1165_4
  have e_1165_5 := arithEq_of_rows h (row := 1165) (i := 5) rfl (by decide)
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
  have e_1165_6 := arithEq_of_rows h (row := 1165) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_6
  simp only [← c71910, ← c71911, ← c71912] at e_1165_6
  have e_1165_7 := arithEq_of_rows h (row := 1165) (i := 7) rfl (by decide)
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
  have e_1165_8 := arithEq_of_rows h (row := 1165) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_8
  simp only [← c71916, ← c71917, ← c71918] at e_1165_8
  have e_1165_9 := arithEq_of_rows h (row := 1165) (i := 9) rfl (by decide)
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
  have e_1165_10 := arithEq_of_rows h (row := 1165) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_10
  simp only [← c71922, ← c71923, ← c71924] at e_1165_10
  have e_1165_11 := arithEq_of_rows h (row := 1165) (i := 11) rfl (by decide)
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
  have e_1165_12 := arithEq_of_rows h (row := 1165) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_12
  simp only [← c71928, ← c71929, ← c71930] at e_1165_12
  have e_1165_13 := arithEq_of_rows h (row := 1165) (i := 13) rfl (by decide)
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
  have e_1165_14 := arithEq_of_rows h (row := 1165) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_14
  simp only [← c71934, ← c71935, ← c71936] at e_1165_14
  have e_1165_15 := arithEq_of_rows h (row := 1165) (i := 15) rfl (by decide)
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
  have e_1165_16 := arithEq_of_rows h (row := 1165) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_16
  simp only [← c71940, ← c71941, ← c71942] at e_1165_16
  have e_1165_17 := arithEq_of_rows h (row := 1165) (i := 17) rfl (by decide)
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
  have e_1165_18 := arithEq_of_rows h (row := 1165) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1165_18
  simp only [← c71946, ← c71947, ← c71948] at e_1165_18
  have e_1165_19 := arithEq_of_rows h (row := 1165) (i := 19) rfl (by decide)
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
  have e_1166_0 := arithEq_of_rows h (row := 1166) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_0
  simp only [← c71952, ← c71953, ← c71954] at e_1166_0
  have e_1166_1 := arithEq_of_rows h (row := 1166) (i := 1) rfl (by decide)
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
  have e_1166_2 := arithEq_of_rows h (row := 1166) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_2
  simp only [← c71958, ← c71959, ← c71960] at e_1166_2
  have e_1166_3 := arithEq_of_rows h (row := 1166) (i := 3) rfl (by decide)
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
  have e_1166_4 := arithEq_of_rows h (row := 1166) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_4
  simp only [← c71964, ← c71965, ← c71966] at e_1166_4
  have e_1166_5 := arithEq_of_rows h (row := 1166) (i := 5) rfl (by decide)
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
  have e_1166_6 := arithEq_of_rows h (row := 1166) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_6
  simp only [← c71970, ← c71971, ← c71972] at e_1166_6
  have e_1166_7 := arithEq_of_rows h (row := 1166) (i := 7) rfl (by decide)
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
  have e_1166_8 := arithEq_of_rows h (row := 1166) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_8
  simp only [← c71976, ← c71977, ← c71978] at e_1166_8
  have e_1166_9 := arithEq_of_rows h (row := 1166) (i := 9) rfl (by decide)
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
  have e_1166_10 := arithEq_of_rows h (row := 1166) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_10
  simp only [← c71982, ← c71983, ← c71984] at e_1166_10
  have e_1166_11 := arithEq_of_rows h (row := 1166) (i := 11) rfl (by decide)
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
  have e_1166_12 := arithEq_of_rows h (row := 1166) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_12
  simp only [← c71988, ← c71989, ← c71990] at e_1166_12
  have e_1166_13 := arithEq_of_rows h (row := 1166) (i := 13) rfl (by decide)
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
  have e_1166_14 := arithEq_of_rows h (row := 1166) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_14
  simp only [← c71994, ← c71995, ← c71996] at e_1166_14
  have e_1166_15 := arithEq_of_rows h (row := 1166) (i := 15) rfl (by decide)
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
  have e_1166_16 := arithEq_of_rows h (row := 1166) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_16
  simp only [← c72000, ← c72001, ← c72002] at e_1166_16
  have e_1166_17 := arithEq_of_rows h (row := 1166) (i := 17) rfl (by decide)
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
  have e_1166_18 := arithEq_of_rows h (row := 1166) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1166_18
  simp only [← c72006, ← c72007, ← c72008] at e_1166_18
  have e_1166_19 := arithEq_of_rows h (row := 1166) (i := 19) rfl (by decide)
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
  have e_1167_0 := arithEq_of_rows h (row := 1167) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_0
  simp only [← c72012, ← c72013, ← c72014] at e_1167_0
  have e_1167_1 := arithEq_of_rows h (row := 1167) (i := 1) rfl (by decide)
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
  have e_1167_2 := arithEq_of_rows h (row := 1167) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_2
  simp only [← c72018, ← c72019, ← c72020] at e_1167_2
  have e_1167_3 := arithEq_of_rows h (row := 1167) (i := 3) rfl (by decide)
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
  have e_1167_4 := arithEq_of_rows h (row := 1167) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_4
  simp only [← c72024, ← c72025, ← c72026] at e_1167_4
  have e_1167_5 := arithEq_of_rows h (row := 1167) (i := 5) rfl (by decide)
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
  have e_1167_6 := arithEq_of_rows h (row := 1167) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_6
  simp only [← c72030, ← c72031, ← c72032] at e_1167_6
  have e_1167_7 := arithEq_of_rows h (row := 1167) (i := 7) rfl (by decide)
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
  have e_1167_8 := arithEq_of_rows h (row := 1167) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_8
  simp only [← c72036, ← c72037, ← c72038] at e_1167_8
  have e_1167_9 := arithEq_of_rows h (row := 1167) (i := 9) rfl (by decide)
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
  have e_1167_10 := arithEq_of_rows h (row := 1167) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_10
  simp only [← c72042, ← c72043, ← c72044] at e_1167_10
  have e_1167_11 := arithEq_of_rows h (row := 1167) (i := 11) rfl (by decide)
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
  have e_1167_12 := arithEq_of_rows h (row := 1167) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_12
  simp only [← c72048, ← c72049, ← c72050] at e_1167_12
  have e_1167_13 := arithEq_of_rows h (row := 1167) (i := 13) rfl (by decide)
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
  have e_1167_14 := arithEq_of_rows h (row := 1167) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_14
  simp only [← c72054, ← c72055, ← c72056] at e_1167_14
  have e_1167_15 := arithEq_of_rows h (row := 1167) (i := 15) rfl (by decide)
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
  have e_1167_16 := arithEq_of_rows h (row := 1167) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_16
  simp only [← c72060, ← c72061, ← c72062] at e_1167_16
  have e_1167_17 := arithEq_of_rows h (row := 1167) (i := 17) rfl (by decide)
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
  have e_1167_18 := arithEq_of_rows h (row := 1167) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1167_18
  simp only [← c72066, ← c72067, ← c72068] at e_1167_18
  have e_1167_19 := arithEq_of_rows h (row := 1167) (i := 19) rfl (by decide)
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
  have e_1168_0 := arithEq_of_rows h (row := 1168) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_0
  simp only [← c72072, ← c72073, ← c72074] at e_1168_0
  have e_1168_1 := arithEq_of_rows h (row := 1168) (i := 1) rfl (by decide)
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
  have e_1168_2 := arithEq_of_rows h (row := 1168) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_2
  simp only [← c72078, ← c72079, ← c72080] at e_1168_2
  have e_1168_3 := arithEq_of_rows h (row := 1168) (i := 3) rfl (by decide)
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
  have e_1168_4 := arithEq_of_rows h (row := 1168) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_4
  simp only [← c72084, ← c72085, ← c72086] at e_1168_4
  have e_1168_5 := arithEq_of_rows h (row := 1168) (i := 5) rfl (by decide)
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
  have e_1168_6 := arithEq_of_rows h (row := 1168) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_6
  simp only [← c72090, ← c72091, ← c72092] at e_1168_6
  have e_1168_7 := arithEq_of_rows h (row := 1168) (i := 7) rfl (by decide)
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
  have e_1168_8 := arithEq_of_rows h (row := 1168) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_8
  simp only [← c72096, ← c72097, ← c72098] at e_1168_8
  have e_1168_9 := arithEq_of_rows h (row := 1168) (i := 9) rfl (by decide)
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
  have e_1168_10 := arithEq_of_rows h (row := 1168) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_10
  simp only [← c72102, ← c72103, ← c72104] at e_1168_10
  have e_1168_11 := arithEq_of_rows h (row := 1168) (i := 11) rfl (by decide)
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
  have e_1168_12 := arithEq_of_rows h (row := 1168) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_12
  simp only [← c72108, ← c72109, ← c72110] at e_1168_12
  have e_1168_13 := arithEq_of_rows h (row := 1168) (i := 13) rfl (by decide)
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
  have e_1168_14 := arithEq_of_rows h (row := 1168) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_14
  simp only [← c72114, ← c72115, ← c72116] at e_1168_14
  have e_1168_15 := arithEq_of_rows h (row := 1168) (i := 15) rfl (by decide)
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
  have e_1168_16 := arithEq_of_rows h (row := 1168) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_16
  simp only [← c72120, ← c72121, ← c72122] at e_1168_16
  have e_1168_17 := arithEq_of_rows h (row := 1168) (i := 17) rfl (by decide)
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
  have e_1168_18 := arithEq_of_rows h (row := 1168) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1168_18
  simp only [← c72126, ← c72127, ← c72128] at e_1168_18
  have e_1168_19 := arithEq_of_rows h (row := 1168) (i := 19) rfl (by decide)
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
  have e_1169_0 := arithEq_of_rows h (row := 1169) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_0
  simp only [← c72132, ← c72133, ← c72134] at e_1169_0
  have e_1169_1 := arithEq_of_rows h (row := 1169) (i := 1) rfl (by decide)
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
  have e_1169_2 := arithEq_of_rows h (row := 1169) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_2
  simp only [← c72138, ← c72139, ← c72140] at e_1169_2
  have e_1169_3 := arithEq_of_rows h (row := 1169) (i := 3) rfl (by decide)
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
  have e_1169_4 := arithEq_of_rows h (row := 1169) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_4
  simp only [← c72144, ← c72145, ← c72146] at e_1169_4
  have e_1169_5 := arithEq_of_rows h (row := 1169) (i := 5) rfl (by decide)
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
  have e_1169_6 := arithEq_of_rows h (row := 1169) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_6
  simp only [← c72150, ← c72151, ← c72152] at e_1169_6
  have e_1169_7 := arithEq_of_rows h (row := 1169) (i := 7) rfl (by decide)
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
  have e_1169_8 := arithEq_of_rows h (row := 1169) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_8
  simp only [← c72156, ← c72157, ← c72158] at e_1169_8
  have e_1169_9 := arithEq_of_rows h (row := 1169) (i := 9) rfl (by decide)
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
  have e_1169_10 := arithEq_of_rows h (row := 1169) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_10
  simp only [← c72162, ← c72163, ← c72164] at e_1169_10
  have e_1169_11 := arithEq_of_rows h (row := 1169) (i := 11) rfl (by decide)
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
  have e_1169_12 := arithEq_of_rows h (row := 1169) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_12
  simp only [← c72168, ← c72169, ← c72170] at e_1169_12
  have e_1169_13 := arithEq_of_rows h (row := 1169) (i := 13) rfl (by decide)
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
  have e_1169_14 := arithEq_of_rows h (row := 1169) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_14
  simp only [← c72174, ← c72175, ← c72176] at e_1169_14
  have e_1169_15 := arithEq_of_rows h (row := 1169) (i := 15) rfl (by decide)
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
  have e_1169_16 := arithEq_of_rows h (row := 1169) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_16
  simp only [← c72180, ← c72181, ← c72182] at e_1169_16
  have e_1169_17 := arithEq_of_rows h (row := 1169) (i := 17) rfl (by decide)
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
  have e_1169_18 := arithEq_of_rows h (row := 1169) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1169_18
  simp only [← c72186, ← c72187, ← c72188] at e_1169_18
  have e_1169_19 := arithEq_of_rows h (row := 1169) (i := 19) rfl (by decide)
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
  have e_1170_0 := arithEq_of_rows h (row := 1170) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_0
  simp only [← c72192, ← c72193, ← c72194] at e_1170_0
  have e_1170_1 := arithEq_of_rows h (row := 1170) (i := 1) rfl (by decide)
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
  have e_1170_2 := arithEq_of_rows h (row := 1170) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_2
  simp only [← c72198, ← c72199, ← c72200] at e_1170_2
  have e_1170_3 := arithEq_of_rows h (row := 1170) (i := 3) rfl (by decide)
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
  have e_1170_4 := arithEq_of_rows h (row := 1170) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_4
  simp only [← c72204, ← c72205, ← c72206] at e_1170_4
  have e_1170_5 := arithEq_of_rows h (row := 1170) (i := 5) rfl (by decide)
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
  have e_1170_6 := arithEq_of_rows h (row := 1170) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_6
  simp only [← c72210, ← c72211, ← c72212] at e_1170_6
  have e_1170_7 := arithEq_of_rows h (row := 1170) (i := 7) rfl (by decide)
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
  have e_1170_8 := arithEq_of_rows h (row := 1170) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_8
  simp only [← c72216, ← c72217, ← c72218] at e_1170_8
  have e_1170_9 := arithEq_of_rows h (row := 1170) (i := 9) rfl (by decide)
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
  have e_1170_10 := arithEq_of_rows h (row := 1170) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_10
  simp only [← c72222, ← c72223, ← c72224] at e_1170_10
  have e_1170_11 := arithEq_of_rows h (row := 1170) (i := 11) rfl (by decide)
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
  have e_1170_12 := arithEq_of_rows h (row := 1170) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_12
  simp only [← c72228, ← c72229, ← c72230] at e_1170_12
  have e_1170_13 := arithEq_of_rows h (row := 1170) (i := 13) rfl (by decide)
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
  have e_1170_14 := arithEq_of_rows h (row := 1170) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_14
  simp only [← c72234, ← c72235, ← c72236] at e_1170_14
  have e_1170_15 := arithEq_of_rows h (row := 1170) (i := 15) rfl (by decide)
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
  have e_1170_16 := arithEq_of_rows h (row := 1170) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_16
  simp only [← c72240, ← c72241, ← c72242] at e_1170_16
  have e_1170_17 := arithEq_of_rows h (row := 1170) (i := 17) rfl (by decide)
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
  have e_1170_18 := arithEq_of_rows h (row := 1170) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1170_18
  simp only [← c72246, ← c72247, ← c72248] at e_1170_18
  have e_1170_19 := arithEq_of_rows h (row := 1170) (i := 19) rfl (by decide)
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
  have e_1171_0 := arithEq_of_rows h (row := 1171) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_0
  simp only [← c72252, ← c72253, ← c72254] at e_1171_0
  have e_1171_1 := arithEq_of_rows h (row := 1171) (i := 1) rfl (by decide)
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
  have e_1171_2 := arithEq_of_rows h (row := 1171) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_2
  simp only [← c72258, ← c72259, ← c72260] at e_1171_2
  have e_1171_3 := arithEq_of_rows h (row := 1171) (i := 3) rfl (by decide)
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
  have e_1171_4 := arithEq_of_rows h (row := 1171) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_4
  simp only [← c72264, ← c72265, ← c72266] at e_1171_4
  have e_1171_5 := arithEq_of_rows h (row := 1171) (i := 5) rfl (by decide)
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
  have e_1171_6 := arithEq_of_rows h (row := 1171) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_6
  simp only [← c72270, ← c72271, ← c72272] at e_1171_6
  have e_1171_7 := arithEq_of_rows h (row := 1171) (i := 7) rfl (by decide)
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
  have e_1171_8 := arithEq_of_rows h (row := 1171) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_8
  simp only [← c72276, ← c72277, ← c72278] at e_1171_8
  have e_1171_9 := arithEq_of_rows h (row := 1171) (i := 9) rfl (by decide)
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
  have e_1171_10 := arithEq_of_rows h (row := 1171) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_10
  simp only [← c72282, ← c72283, ← c72284] at e_1171_10
  have e_1171_11 := arithEq_of_rows h (row := 1171) (i := 11) rfl (by decide)
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
  have e_1171_12 := arithEq_of_rows h (row := 1171) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_12
  simp only [← c72288, ← c72289, ← c72290] at e_1171_12
  have e_1171_13 := arithEq_of_rows h (row := 1171) (i := 13) rfl (by decide)
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
  have e_1171_14 := arithEq_of_rows h (row := 1171) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_14
  simp only [← c72294, ← c72295, ← c72296] at e_1171_14
  have e_1171_15 := arithEq_of_rows h (row := 1171) (i := 15) rfl (by decide)
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
  have e_1171_16 := arithEq_of_rows h (row := 1171) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_16
  simp only [← c72300, ← c72301, ← c72302] at e_1171_16
  have e_1171_17 := arithEq_of_rows h (row := 1171) (i := 17) rfl (by decide)
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
  have e_1171_18 := arithEq_of_rows h (row := 1171) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1171_18
  simp only [← c72306, ← c72307, ← c72308] at e_1171_18
  have e_1171_19 := arithEq_of_rows h (row := 1171) (i := 19) rfl (by decide)
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
  have e_1172_0 := arithEq_of_rows h (row := 1172) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_0
  simp only [← c72312, ← c72313, ← c72314] at e_1172_0
  have e_1172_1 := arithEq_of_rows h (row := 1172) (i := 1) rfl (by decide)
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
  have e_1172_2 := arithEq_of_rows h (row := 1172) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_2
  simp only [← c72318, ← c72319, ← c72320] at e_1172_2
  have e_1172_3 := arithEq_of_rows h (row := 1172) (i := 3) rfl (by decide)
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
  have e_1172_4 := arithEq_of_rows h (row := 1172) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_4
  simp only [← c72324, ← c72325, ← c72326] at e_1172_4
  have e_1172_5 := arithEq_of_rows h (row := 1172) (i := 5) rfl (by decide)
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
  have e_1172_6 := arithEq_of_rows h (row := 1172) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_6
  simp only [← c72330, ← c72331, ← c72332] at e_1172_6
  have e_1172_7 := arithEq_of_rows h (row := 1172) (i := 7) rfl (by decide)
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
  have e_1172_8 := arithEq_of_rows h (row := 1172) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_8
  simp only [← c72336, ← c72337, ← c72338] at e_1172_8
  have e_1172_9 := arithEq_of_rows h (row := 1172) (i := 9) rfl (by decide)
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
  have e_1172_10 := arithEq_of_rows h (row := 1172) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_10
  simp only [← c72342, ← c72343, ← c72344] at e_1172_10
  have e_1172_11 := arithEq_of_rows h (row := 1172) (i := 11) rfl (by decide)
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
  have e_1172_12 := arithEq_of_rows h (row := 1172) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_12
  simp only [← c72348, ← c72349, ← c72350] at e_1172_12
  have e_1172_13 := arithEq_of_rows h (row := 1172) (i := 13) rfl (by decide)
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
  have e_1172_14 := arithEq_of_rows h (row := 1172) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_14
  simp only [← c72354, ← c72355, ← c72356] at e_1172_14
  have e_1172_15 := arithEq_of_rows h (row := 1172) (i := 15) rfl (by decide)
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
  have e_1172_16 := arithEq_of_rows h (row := 1172) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_16
  simp only [← c72360, ← c72361, ← c72362] at e_1172_16
  have e_1172_17 := arithEq_of_rows h (row := 1172) (i := 17) rfl (by decide)
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
  have e_1172_18 := arithEq_of_rows h (row := 1172) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1172_18
  simp only [← c72366, ← c72367, ← c72368] at e_1172_18
  have e_1172_19 := arithEq_of_rows h (row := 1172) (i := 19) rfl (by decide)
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
  have e_1173_0 := arithEq_of_rows h (row := 1173) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_0
  simp only [← c72372, ← c72373, ← c72374] at e_1173_0
  have e_1173_1 := arithEq_of_rows h (row := 1173) (i := 1) rfl (by decide)
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
  have e_1173_2 := arithEq_of_rows h (row := 1173) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_2
  simp only [← c72378, ← c72379, ← c72380] at e_1173_2
  have e_1173_3 := arithEq_of_rows h (row := 1173) (i := 3) rfl (by decide)
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
  have e_1173_4 := arithEq_of_rows h (row := 1173) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_4
  simp only [← c72384, ← c72385, ← c72386] at e_1173_4
  have e_1173_5 := arithEq_of_rows h (row := 1173) (i := 5) rfl (by decide)
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
  have e_1173_6 := arithEq_of_rows h (row := 1173) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_6
  simp only [← c72390, ← c72391, ← c72392] at e_1173_6
  have e_1173_7 := arithEq_of_rows h (row := 1173) (i := 7) rfl (by decide)
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
  have e_1173_8 := arithEq_of_rows h (row := 1173) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_8
  simp only [← c72396, ← c72397, ← c72398] at e_1173_8
  have e_1173_9 := arithEq_of_rows h (row := 1173) (i := 9) rfl (by decide)
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
  have e_1173_10 := arithEq_of_rows h (row := 1173) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_10
  simp only [← c72402, ← c72403, ← c72404] at e_1173_10
  have e_1173_11 := arithEq_of_rows h (row := 1173) (i := 11) rfl (by decide)
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
  have e_1173_12 := arithEq_of_rows h (row := 1173) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_12
  simp only [← c72408, ← c72409, ← c72410] at e_1173_12
  have e_1173_13 := arithEq_of_rows h (row := 1173) (i := 13) rfl (by decide)
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
  have e_1173_14 := arithEq_of_rows h (row := 1173) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_14
  simp only [← c72414, ← c72415, ← c72416] at e_1173_14
  have e_1173_15 := arithEq_of_rows h (row := 1173) (i := 15) rfl (by decide)
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
  have e_1173_16 := arithEq_of_rows h (row := 1173) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_16
  simp only [← c72420, ← c72421, ← c72422] at e_1173_16
  have e_1173_17 := arithEq_of_rows h (row := 1173) (i := 17) rfl (by decide)
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
  have e_1173_18 := arithEq_of_rows h (row := 1173) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1173_18
  simp only [← c72426, ← c72427, ← c72428] at e_1173_18
  have e_1173_19 := arithEq_of_rows h (row := 1173) (i := 19) rfl (by decide)
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
  have e_1174_0 := arithEq_of_rows h (row := 1174) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_0
  simp only [← c72432, ← c72433, ← c72434] at e_1174_0
  have e_1174_1 := arithEq_of_rows h (row := 1174) (i := 1) rfl (by decide)
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
  have e_1174_2 := arithEq_of_rows h (row := 1174) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_2
  simp only [← c72438, ← c72439, ← c72440] at e_1174_2
  have e_1174_3 := arithEq_of_rows h (row := 1174) (i := 3) rfl (by decide)
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
  have e_1174_4 := arithEq_of_rows h (row := 1174) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_4
  simp only [← c72444, ← c72445, ← c72446] at e_1174_4
  have e_1174_5 := arithEq_of_rows h (row := 1174) (i := 5) rfl (by decide)
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
  have e_1174_6 := arithEq_of_rows h (row := 1174) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_6
  simp only [← c72450, ← c72451, ← c72452] at e_1174_6
  have e_1174_7 := arithEq_of_rows h (row := 1174) (i := 7) rfl (by decide)
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
  have e_1174_8 := arithEq_of_rows h (row := 1174) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_8
  simp only [← c72456, ← c72457, ← c72458] at e_1174_8
  have e_1174_9 := arithEq_of_rows h (row := 1174) (i := 9) rfl (by decide)
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
  have e_1174_10 := arithEq_of_rows h (row := 1174) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_10
  simp only [← c72462, ← c72463, ← c72464] at e_1174_10
  have e_1174_11 := arithEq_of_rows h (row := 1174) (i := 11) rfl (by decide)
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
  have e_1174_12 := arithEq_of_rows h (row := 1174) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_12
  simp only [← c72468, ← c72469, ← c72470] at e_1174_12
  have e_1174_13 := arithEq_of_rows h (row := 1174) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_13
  simp only [← c72471, ← c72472, k2, ← c72473] at e_1174_13
  have hr := e_1174_13
  simp only [e_1174_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10352)) := by
  have c72474 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72475 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72476 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72477 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72478 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72479 := (publicBatchWrapper89_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_14 := arithEq_of_rows h (row := 1174) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_14
  simp only [← c72474, ← c72475, ← c72476] at e_1174_14
  have e_1174_15 := arithEq_of_rows h (row := 1174) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_15
  simp only [← c72477, ← c72478, k2, ← c72479] at e_1174_15
  have hr := e_1174_15
  simp only [e_1174_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10353)) := by
  have c72480 := (publicBatchWrapper89_copies2265 a h).1
  have c72481 := (publicBatchWrapper89_copies2265 a h).2.1
  have c72482 := (publicBatchWrapper89_copies2265 a h).2.2.1
  have c72483 := (publicBatchWrapper89_copies2265 a h).2.2.2.1
  have c72484 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.1
  have c72485 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_16 := arithEq_of_rows h (row := 1174) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_16
  simp only [← c72480, ← c72481, ← c72482] at e_1174_16
  have e_1174_17 := arithEq_of_rows h (row := 1174) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_17
  simp only [← c72483, ← c72484, k2, ← c72485] at e_1174_17
  have hr := e_1174_17
  simp only [e_1174_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1174 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10354)) := by
  have c72486 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.1
  have c72487 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.1
  have c72488 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.1
  have c72489 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.1
  have c72490 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72491 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1174_18 := arithEq_of_rows h (row := 1174) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_18
  simp only [← c72486, ← c72487, ← c72488] at e_1174_18
  have e_1174_19 := arithEq_of_rows h (row := 1174) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_19
  simp only [← c72489, ← c72490, k2, ← c72491] at e_1174_19
  have hr := e_1174_19
  simp only [e_1174_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10355)) := by
  have c72492 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72493 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72494 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72495 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72496 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72497 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_0 := arithEq_of_rows h (row := 1175) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_0
  simp only [← c72492, ← c72493, ← c72494] at e_1175_0
  have e_1175_1 := arithEq_of_rows h (row := 1175) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_1
  simp only [← c72495, ← c72496, k2, ← c72497] at e_1175_1
  have hr := e_1175_1
  simp only [e_1175_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10356)) := by
  have c72498 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72499 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72500 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72501 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72502 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72503 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_2 := arithEq_of_rows h (row := 1175) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_2
  simp only [← c72498, ← c72499, ← c72500] at e_1175_2
  have e_1175_3 := arithEq_of_rows h (row := 1175) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_3
  simp only [← c72501, ← c72502, k2, ← c72503] at e_1175_3
  have hr := e_1175_3
  simp only [e_1175_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10357)) := by
  have c72504 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72505 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72506 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72507 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72508 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72509 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_4 := arithEq_of_rows h (row := 1175) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_4
  simp only [← c72504, ← c72505, ← c72506] at e_1175_4
  have e_1175_5 := arithEq_of_rows h (row := 1175) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_5
  simp only [← c72507, ← c72508, k2, ← c72509] at e_1175_5
  have hr := e_1175_5
  simp only [e_1175_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10358)) := by
  have c72510 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72511 := (publicBatchWrapper89_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72512 := (publicBatchWrapper89_copies2266 a h).1
  have c72513 := (publicBatchWrapper89_copies2266 a h).2.1
  have c72514 := (publicBatchWrapper89_copies2266 a h).2.2.1
  have c72515 := (publicBatchWrapper89_copies2266 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_6 := arithEq_of_rows h (row := 1175) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_6
  simp only [← c72510, ← c72511, ← c72512] at e_1175_6
  have e_1175_7 := arithEq_of_rows h (row := 1175) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_7
  simp only [← c72513, ← c72514, k2, ← c72515] at e_1175_7
  have hr := e_1175_7
  simp only [e_1175_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10359)) := by
  have c72516 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.1
  have c72517 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.1
  have c72518 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.1
  have c72519 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.1
  have c72520 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.1
  have c72521 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_8 := arithEq_of_rows h (row := 1175) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_8
  simp only [← c72516, ← c72517, ← c72518] at e_1175_8
  have e_1175_9 := arithEq_of_rows h (row := 1175) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_9
  simp only [← c72519, ← c72520, k2, ← c72521] at e_1175_9
  have hr := e_1175_9
  simp only [e_1175_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10360)) := by
  have c72522 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72523 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72524 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72525 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72526 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72527 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_10 := arithEq_of_rows h (row := 1175) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_10
  simp only [← c72522, ← c72523, ← c72524] at e_1175_10
  have e_1175_11 := arithEq_of_rows h (row := 1175) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_11
  simp only [← c72525, ← c72526, k2, ← c72527] at e_1175_11
  have hr := e_1175_11
  simp only [e_1175_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10361)) := by
  have c72528 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72529 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72530 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72531 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72532 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72533 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_12 := arithEq_of_rows h (row := 1175) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_12
  simp only [← c72528, ← c72529, ← c72530] at e_1175_12
  have e_1175_13 := arithEq_of_rows h (row := 1175) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_13
  simp only [← c72531, ← c72532, k2, ← c72533] at e_1175_13
  have hr := e_1175_13
  simp only [e_1175_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10362)) := by
  have c72534 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72535 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72536 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72537 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72538 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72539 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_14 := arithEq_of_rows h (row := 1175) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_14
  simp only [← c72534, ← c72535, ← c72536] at e_1175_14
  have e_1175_15 := arithEq_of_rows h (row := 1175) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_15
  simp only [← c72537, ← c72538, k2, ← c72539] at e_1175_15
  have hr := e_1175_15
  simp only [e_1175_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10363)) := by
  have c72540 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72541 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72542 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72543 := (publicBatchWrapper89_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72544 := (publicBatchWrapper89_copies2267 a h).1
  have c72545 := (publicBatchWrapper89_copies2267 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_16 := arithEq_of_rows h (row := 1175) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_16
  simp only [← c72540, ← c72541, ← c72542] at e_1175_16
  have e_1175_17 := arithEq_of_rows h (row := 1175) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_17
  simp only [← c72543, ← c72544, k2, ← c72545] at e_1175_17
  have hr := e_1175_17
  simp only [e_1175_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1175 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10364)) := by
  have c72546 := (publicBatchWrapper89_copies2267 a h).2.2.1
  have c72547 := (publicBatchWrapper89_copies2267 a h).2.2.2.1
  have c72548 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.1
  have c72549 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.1
  have c72550 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.1
  have c72551 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1175_18 := arithEq_of_rows h (row := 1175) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_18
  simp only [← c72546, ← c72547, ← c72548] at e_1175_18
  have e_1175_19 := arithEq_of_rows h (row := 1175) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_19
  simp only [← c72549, ← c72550, k2, ← c72551] at e_1175_19
  have hr := e_1175_19
  simp only [e_1175_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10365)) := by
  have c72552 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.1
  have c72553 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.1
  have c72554 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72555 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72556 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72557 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_0 := arithEq_of_rows h (row := 1176) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_0
  simp only [← c72552, ← c72553, ← c72554] at e_1176_0
  have e_1176_1 := arithEq_of_rows h (row := 1176) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_1
  simp only [← c72555, ← c72556, k2, ← c72557] at e_1176_1
  have hr := e_1176_1
  simp only [e_1176_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10366)) := by
  have c72558 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72559 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72560 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72561 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72562 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72563 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_2 := arithEq_of_rows h (row := 1176) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_2
  simp only [← c72558, ← c72559, ← c72560] at e_1176_2
  have e_1176_3 := arithEq_of_rows h (row := 1176) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_3
  simp only [← c72561, ← c72562, k2, ← c72563] at e_1176_3
  have hr := e_1176_3
  simp only [e_1176_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10367)) := by
  have c72564 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72565 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72566 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72567 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72568 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72569 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_4 := arithEq_of_rows h (row := 1176) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_4
  simp only [← c72564, ← c72565, ← c72566] at e_1176_4
  have e_1176_5 := arithEq_of_rows h (row := 1176) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_5
  simp only [← c72567, ← c72568, k2, ← c72569] at e_1176_5
  have hr := e_1176_5
  simp only [e_1176_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10368)) := by
  have c72570 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72571 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72572 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72573 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72574 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72575 := (publicBatchWrapper89_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_6 := arithEq_of_rows h (row := 1176) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_6
  simp only [← c72570, ← c72571, ← c72572] at e_1176_6
  have e_1176_7 := arithEq_of_rows h (row := 1176) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_7
  simp only [← c72573, ← c72574, k2, ← c72575] at e_1176_7
  have hr := e_1176_7
  simp only [e_1176_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10369)) := by
  have c72576 := (publicBatchWrapper89_copies2268 a h).1
  have c72577 := (publicBatchWrapper89_copies2268 a h).2.1
  have c72578 := (publicBatchWrapper89_copies2268 a h).2.2.1
  have c72579 := (publicBatchWrapper89_copies2268 a h).2.2.2.1
  have c72580 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.1
  have c72581 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_8 := arithEq_of_rows h (row := 1176) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_8
  simp only [← c72576, ← c72577, ← c72578] at e_1176_8
  have e_1176_9 := arithEq_of_rows h (row := 1176) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_9
  simp only [← c72579, ← c72580, k2, ← c72581] at e_1176_9
  have hr := e_1176_9
  simp only [e_1176_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10410)) := by
  have c72582 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.1
  have c72583 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.1
  have c72584 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.1
  have c72585 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.1
  have c72586 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72587 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_10 := arithEq_of_rows h (row := 1176) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_10
  simp only [← c72582, ← c72583, ← c72584] at e_1176_10
  have e_1176_11 := arithEq_of_rows h (row := 1176) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_11
  simp only [← c72585, ← c72586, k2, ← c72587] at e_1176_11
  have hr := e_1176_11
  simp only [e_1176_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10411)) := by
  have c72588 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72589 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72590 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72591 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72592 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72593 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_12 := arithEq_of_rows h (row := 1176) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_12
  simp only [← c72588, ← c72589, ← c72590] at e_1176_12
  have e_1176_13 := arithEq_of_rows h (row := 1176) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_13
  simp only [← c72591, ← c72592, k2, ← c72593] at e_1176_13
  have hr := e_1176_13
  simp only [e_1176_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10412)) := by
  have c72594 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72595 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72596 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72597 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72598 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72599 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_14 := arithEq_of_rows h (row := 1176) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_14
  simp only [← c72594, ← c72595, ← c72596] at e_1176_14
  have e_1176_15 := arithEq_of_rows h (row := 1176) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_15
  simp only [← c72597, ← c72598, k2, ← c72599] at e_1176_15
  have hr := e_1176_15
  simp only [e_1176_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10413)) := by
  have c72600 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72601 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72602 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72603 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72604 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72605 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_16 := arithEq_of_rows h (row := 1176) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_16
  simp only [← c72600, ← c72601, ← c72602] at e_1176_16
  have e_1176_17 := arithEq_of_rows h (row := 1176) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_17
  simp only [← c72603, ← c72604, k2, ← c72605] at e_1176_17
  have hr := e_1176_17
  simp only [e_1176_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1176 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10414)) := by
  have c72606 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72607 := (publicBatchWrapper89_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72608 := (publicBatchWrapper89_copies2269 a h).1
  have c72609 := (publicBatchWrapper89_copies2269 a h).2.1
  have c72610 := (publicBatchWrapper89_copies2269 a h).2.2.1
  have c72611 := (publicBatchWrapper89_copies2269 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1176_18 := arithEq_of_rows h (row := 1176) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_18
  simp only [← c72606, ← c72607, ← c72608] at e_1176_18
  have e_1176_19 := arithEq_of_rows h (row := 1176) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_19
  simp only [← c72609, ← c72610, k2, ← c72611] at e_1176_19
  have hr := e_1176_19
  simp only [e_1176_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10415)) := by
  have c72612 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.1
  have c72613 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.1
  have c72614 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.1
  have c72615 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.1
  have c72616 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.1
  have c72617 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_0 := arithEq_of_rows h (row := 1177) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_0
  simp only [← c72612, ← c72613, ← c72614] at e_1177_0
  have e_1177_1 := arithEq_of_rows h (row := 1177) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_1
  simp only [← c72615, ← c72616, k2, ← c72617] at e_1177_1
  have hr := e_1177_1
  simp only [e_1177_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10416)) := by
  have c72618 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72619 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72620 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72621 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72622 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72623 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_2 := arithEq_of_rows h (row := 1177) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_2
  simp only [← c72618, ← c72619, ← c72620] at e_1177_2
  have e_1177_3 := arithEq_of_rows h (row := 1177) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_3
  simp only [← c72621, ← c72622, k2, ← c72623] at e_1177_3
  have hr := e_1177_3
  simp only [e_1177_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10417)) := by
  have c72624 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72625 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72626 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72627 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72628 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72629 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_4 := arithEq_of_rows h (row := 1177) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_4
  simp only [← c72624, ← c72625, ← c72626] at e_1177_4
  have e_1177_5 := arithEq_of_rows h (row := 1177) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_5
  simp only [← c72627, ← c72628, k2, ← c72629] at e_1177_5
  have hr := e_1177_5
  simp only [e_1177_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10418)) := by
  have c72630 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72631 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72632 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72633 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72634 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72635 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_6 := arithEq_of_rows h (row := 1177) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_6
  simp only [← c72630, ← c72631, ← c72632] at e_1177_6
  have e_1177_7 := arithEq_of_rows h (row := 1177) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_7
  simp only [← c72633, ← c72634, k2, ← c72635] at e_1177_7
  have hr := e_1177_7
  simp only [e_1177_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10419)) := by
  have c72636 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72637 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72638 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72639 := (publicBatchWrapper89_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72640 := (publicBatchWrapper89_copies2270 a h).1
  have c72641 := (publicBatchWrapper89_copies2270 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_8 := arithEq_of_rows h (row := 1177) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_8
  simp only [← c72636, ← c72637, ← c72638] at e_1177_8
  have e_1177_9 := arithEq_of_rows h (row := 1177) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_9
  simp only [← c72639, ← c72640, k2, ← c72641] at e_1177_9
  have hr := e_1177_9
  simp only [e_1177_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10420)) := by
  have c72642 := (publicBatchWrapper89_copies2270 a h).2.2.1
  have c72643 := (publicBatchWrapper89_copies2270 a h).2.2.2.1
  have c72644 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.1
  have c72645 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.1
  have c72646 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.1
  have c72647 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_10 := arithEq_of_rows h (row := 1177) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_10
  simp only [← c72642, ← c72643, ← c72644] at e_1177_10
  have e_1177_11 := arithEq_of_rows h (row := 1177) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_11
  simp only [← c72645, ← c72646, k2, ← c72647] at e_1177_11
  have hr := e_1177_11
  simp only [e_1177_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10421)) := by
  have c72648 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.1
  have c72649 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.1
  have c72650 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72651 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72652 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72653 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_12 := arithEq_of_rows h (row := 1177) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_12
  simp only [← c72648, ← c72649, ← c72650] at e_1177_12
  have e_1177_13 := arithEq_of_rows h (row := 1177) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_13
  simp only [← c72651, ← c72652, k2, ← c72653] at e_1177_13
  have hr := e_1177_13
  simp only [e_1177_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10422)) := by
  have c72654 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72655 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72656 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72657 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72658 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72659 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_14 := arithEq_of_rows h (row := 1177) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_14
  simp only [← c72654, ← c72655, ← c72656] at e_1177_14
  have e_1177_15 := arithEq_of_rows h (row := 1177) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_15
  simp only [← c72657, ← c72658, k2, ← c72659] at e_1177_15
  have hr := e_1177_15
  simp only [e_1177_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10423)) := by
  have c72660 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72661 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72662 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72663 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72664 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72665 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_16 := arithEq_of_rows h (row := 1177) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_16
  simp only [← c72660, ← c72661, ← c72662] at e_1177_16
  have e_1177_17 := arithEq_of_rows h (row := 1177) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_17
  simp only [← c72663, ← c72664, k2, ← c72665] at e_1177_17
  have hr := e_1177_17
  simp only [e_1177_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1177 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10424)) := by
  have c72666 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72667 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72668 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72669 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72670 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72671 := (publicBatchWrapper89_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1177_18 := arithEq_of_rows h (row := 1177) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_18
  simp only [← c72666, ← c72667, ← c72668] at e_1177_18
  have e_1177_19 := arithEq_of_rows h (row := 1177) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_19
  simp only [← c72669, ← c72670, k2, ← c72671] at e_1177_19
  have hr := e_1177_19
  simp only [e_1177_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10425)) := by
  have c72672 := (publicBatchWrapper89_copies2271 a h).1
  have c72673 := (publicBatchWrapper89_copies2271 a h).2.1
  have c72674 := (publicBatchWrapper89_copies2271 a h).2.2.1
  have c72675 := (publicBatchWrapper89_copies2271 a h).2.2.2.1
  have c72676 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.1
  have c72677 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_0 := arithEq_of_rows h (row := 1178) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_0
  simp only [← c72672, ← c72673, ← c72674] at e_1178_0
  have e_1178_1 := arithEq_of_rows h (row := 1178) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_1
  simp only [← c72675, ← c72676, k2, ← c72677] at e_1178_1
  have hr := e_1178_1
  simp only [e_1178_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10426)) := by
  have c72678 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.1
  have c72679 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.1
  have c72680 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.1
  have c72681 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.1
  have c72682 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72683 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_2 := arithEq_of_rows h (row := 1178) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_2
  simp only [← c72678, ← c72679, ← c72680] at e_1178_2
  have e_1178_3 := arithEq_of_rows h (row := 1178) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_3
  simp only [← c72681, ← c72682, k2, ← c72683] at e_1178_3
  have hr := e_1178_3
  simp only [e_1178_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10427)) := by
  have c72684 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72685 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72686 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72687 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72688 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72689 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_4 := arithEq_of_rows h (row := 1178) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_4
  simp only [← c72684, ← c72685, ← c72686] at e_1178_4
  have e_1178_5 := arithEq_of_rows h (row := 1178) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_5
  simp only [← c72687, ← c72688, k2, ← c72689] at e_1178_5
  have hr := e_1178_5
  simp only [e_1178_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10428)) := by
  have c72690 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72691 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72692 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72693 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72694 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72695 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_6 := arithEq_of_rows h (row := 1178) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_6
  simp only [← c72690, ← c72691, ← c72692] at e_1178_6
  have e_1178_7 := arithEq_of_rows h (row := 1178) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_7
  simp only [← c72693, ← c72694, k2, ← c72695] at e_1178_7
  have hr := e_1178_7
  simp only [e_1178_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10429)) := by
  have c72696 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72697 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72698 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72699 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72700 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72701 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_8 := arithEq_of_rows h (row := 1178) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_8
  simp only [← c72696, ← c72697, ← c72698] at e_1178_8
  have e_1178_9 := arithEq_of_rows h (row := 1178) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_9
  simp only [← c72699, ← c72700, k2, ← c72701] at e_1178_9
  have hr := e_1178_9
  simp only [e_1178_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10430)) := by
  have c72702 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72703 := (publicBatchWrapper89_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72704 := (publicBatchWrapper89_copies2272 a h).1
  have c72705 := (publicBatchWrapper89_copies2272 a h).2.1
  have c72706 := (publicBatchWrapper89_copies2272 a h).2.2.1
  have c72707 := (publicBatchWrapper89_copies2272 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_10 := arithEq_of_rows h (row := 1178) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_10
  simp only [← c72702, ← c72703, ← c72704] at e_1178_10
  have e_1178_11 := arithEq_of_rows h (row := 1178) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_11
  simp only [← c72705, ← c72706, k2, ← c72707] at e_1178_11
  have hr := e_1178_11
  simp only [e_1178_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10431)) := by
  have c72708 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.1
  have c72709 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.1
  have c72710 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.1
  have c72711 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.1
  have c72712 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.1
  have c72713 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_12 := arithEq_of_rows h (row := 1178) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_12
  simp only [← c72708, ← c72709, ← c72710] at e_1178_12
  have e_1178_13 := arithEq_of_rows h (row := 1178) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_13
  simp only [← c72711, ← c72712, k2, ← c72713] at e_1178_13
  have hr := e_1178_13
  simp only [e_1178_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10432)) := by
  have c72714 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72715 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72716 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72717 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72718 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72719 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_14 := arithEq_of_rows h (row := 1178) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_14
  simp only [← c72714, ← c72715, ← c72716] at e_1178_14
  have e_1178_15 := arithEq_of_rows h (row := 1178) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_15
  simp only [← c72717, ← c72718, k2, ← c72719] at e_1178_15
  have hr := e_1178_15
  simp only [e_1178_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10433)) := by
  have c72720 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72721 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72722 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72723 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72724 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72725 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_16 := arithEq_of_rows h (row := 1178) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_16
  simp only [← c72720, ← c72721, ← c72722] at e_1178_16
  have e_1178_17 := arithEq_of_rows h (row := 1178) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_17
  simp only [← c72723, ← c72724, k2, ← c72725] at e_1178_17
  have hr := e_1178_17
  simp only [e_1178_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1178 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10434)) := by
  have c72726 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72727 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72728 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72729 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72730 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72731 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1178_18 := arithEq_of_rows h (row := 1178) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_18
  simp only [← c72726, ← c72727, ← c72728] at e_1178_18
  have e_1178_19 := arithEq_of_rows h (row := 1178) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_19
  simp only [← c72729, ← c72730, k2, ← c72731] at e_1178_19
  have hr := e_1178_19
  simp only [e_1178_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10435)) := by
  have c72732 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72733 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72734 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72735 := (publicBatchWrapper89_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72736 := (publicBatchWrapper89_copies2273 a h).1
  have c72737 := (publicBatchWrapper89_copies2273 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_0 := arithEq_of_rows h (row := 1179) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_0
  simp only [← c72732, ← c72733, ← c72734] at e_1179_0
  have e_1179_1 := arithEq_of_rows h (row := 1179) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_1
  simp only [← c72735, ← c72736, k2, ← c72737] at e_1179_1
  have hr := e_1179_1
  simp only [e_1179_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10436)) := by
  have c72738 := (publicBatchWrapper89_copies2273 a h).2.2.1
  have c72739 := (publicBatchWrapper89_copies2273 a h).2.2.2.1
  have c72740 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.1
  have c72741 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.1
  have c72742 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.1
  have c72743 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_2 := arithEq_of_rows h (row := 1179) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_2
  simp only [← c72738, ← c72739, ← c72740] at e_1179_2
  have e_1179_3 := arithEq_of_rows h (row := 1179) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_3
  simp only [← c72741, ← c72742, k2, ← c72743] at e_1179_3
  have hr := e_1179_3
  simp only [e_1179_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10437)) := by
  have c72744 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.1
  have c72745 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.1
  have c72746 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72747 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72748 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72749 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_4 := arithEq_of_rows h (row := 1179) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_4
  simp only [← c72744, ← c72745, ← c72746] at e_1179_4
  have e_1179_5 := arithEq_of_rows h (row := 1179) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_5
  simp only [← c72747, ← c72748, k2, ← c72749] at e_1179_5
  have hr := e_1179_5
  simp only [e_1179_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10438)) := by
  have c72750 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72751 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72752 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72753 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72754 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72755 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_6 := arithEq_of_rows h (row := 1179) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_6
  simp only [← c72750, ← c72751, ← c72752] at e_1179_6
  have e_1179_7 := arithEq_of_rows h (row := 1179) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_7
  simp only [← c72753, ← c72754, k2, ← c72755] at e_1179_7
  have hr := e_1179_7
  simp only [e_1179_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10439)) := by
  have c72756 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72757 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72758 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72759 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72760 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72761 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_8 := arithEq_of_rows h (row := 1179) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_8
  simp only [← c72756, ← c72757, ← c72758] at e_1179_8
  have e_1179_9 := arithEq_of_rows h (row := 1179) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_9
  simp only [← c72759, ← c72760, k2, ← c72761] at e_1179_9
  have hr := e_1179_9
  simp only [e_1179_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10440)) := by
  have c72762 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72763 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72764 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72765 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72766 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72767 := (publicBatchWrapper89_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_10 := arithEq_of_rows h (row := 1179) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_10
  simp only [← c72762, ← c72763, ← c72764] at e_1179_10
  have e_1179_11 := arithEq_of_rows h (row := 1179) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_11
  simp only [← c72765, ← c72766, k2, ← c72767] at e_1179_11
  have hr := e_1179_11
  simp only [e_1179_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10441)) := by
  have c72768 := (publicBatchWrapper89_copies2274 a h).1
  have c72769 := (publicBatchWrapper89_copies2274 a h).2.1
  have c72770 := (publicBatchWrapper89_copies2274 a h).2.2.1
  have c72771 := (publicBatchWrapper89_copies2274 a h).2.2.2.1
  have c72772 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.1
  have c72773 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_12 := arithEq_of_rows h (row := 1179) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_12
  simp only [← c72768, ← c72769, ← c72770] at e_1179_12
  have e_1179_13 := arithEq_of_rows h (row := 1179) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_13
  simp only [← c72771, ← c72772, k2, ← c72773] at e_1179_13
  have hr := e_1179_13
  simp only [e_1179_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10442)) := by
  have c72774 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.1
  have c72775 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.1
  have c72776 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.1
  have c72777 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.1
  have c72778 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72779 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_14 := arithEq_of_rows h (row := 1179) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_14
  simp only [← c72774, ← c72775, ← c72776] at e_1179_14
  have e_1179_15 := arithEq_of_rows h (row := 1179) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_15
  simp only [← c72777, ← c72778, k2, ← c72779] at e_1179_15
  have hr := e_1179_15
  simp only [e_1179_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10443)) := by
  have c72780 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72781 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72782 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72783 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72784 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72785 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_16 := arithEq_of_rows h (row := 1179) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_16
  simp only [← c72780, ← c72781, ← c72782] at e_1179_16
  have e_1179_17 := arithEq_of_rows h (row := 1179) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_17
  simp only [← c72783, ← c72784, k2, ← c72785] at e_1179_17
  have hr := e_1179_17
  simp only [e_1179_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1179 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10444)) := by
  have c72786 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72787 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72788 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72789 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72790 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72791 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1179_18 := arithEq_of_rows h (row := 1179) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_18
  simp only [← c72786, ← c72787, ← c72788] at e_1179_18
  have e_1179_19 := arithEq_of_rows h (row := 1179) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_19
  simp only [← c72789, ← c72790, k2, ← c72791] at e_1179_19
  have hr := e_1179_19
  simp only [e_1179_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10445)) := by
  have c72792 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72793 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72794 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72795 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72796 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72797 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_0 := arithEq_of_rows h (row := 1180) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_0
  simp only [← c72792, ← c72793, ← c72794] at e_1180_0
  have e_1180_1 := arithEq_of_rows h (row := 1180) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_1
  simp only [← c72795, ← c72796, k2, ← c72797] at e_1180_1
  have hr := e_1180_1
  simp only [e_1180_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10446)) := by
  have c72798 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72799 := (publicBatchWrapper89_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72800 := (publicBatchWrapper89_copies2275 a h).1
  have c72801 := (publicBatchWrapper89_copies2275 a h).2.1
  have c72802 := (publicBatchWrapper89_copies2275 a h).2.2.1
  have c72803 := (publicBatchWrapper89_copies2275 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_2 := arithEq_of_rows h (row := 1180) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_2
  simp only [← c72798, ← c72799, ← c72800] at e_1180_2
  have e_1180_3 := arithEq_of_rows h (row := 1180) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_3
  simp only [← c72801, ← c72802, k2, ← c72803] at e_1180_3
  have hr := e_1180_3
  simp only [e_1180_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10447)) := by
  have c72804 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.1
  have c72805 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.1
  have c72806 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.1
  have c72807 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.1
  have c72808 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.1
  have c72809 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_4 := arithEq_of_rows h (row := 1180) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_4
  simp only [← c72804, ← c72805, ← c72806] at e_1180_4
  have e_1180_5 := arithEq_of_rows h (row := 1180) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_5
  simp only [← c72807, ← c72808, k2, ← c72809] at e_1180_5
  have hr := e_1180_5
  simp only [e_1180_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10448)) := by
  have c72810 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72811 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72812 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72813 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72814 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72815 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_6 := arithEq_of_rows h (row := 1180) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_6
  simp only [← c72810, ← c72811, ← c72812] at e_1180_6
  have e_1180_7 := arithEq_of_rows h (row := 1180) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_7
  simp only [← c72813, ← c72814, k2, ← c72815] at e_1180_7
  have hr := e_1180_7
  simp only [e_1180_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10449)) := by
  have c72816 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72817 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72818 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72819 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72820 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72821 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_8 := arithEq_of_rows h (row := 1180) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_8
  simp only [← c72816, ← c72817, ← c72818] at e_1180_8
  have e_1180_9 := arithEq_of_rows h (row := 1180) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_9
  simp only [← c72819, ← c72820, k2, ← c72821] at e_1180_9
  have hr := e_1180_9
  simp only [e_1180_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10450)) := by
  have c72822 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72823 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72824 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72825 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72826 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72827 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_10 := arithEq_of_rows h (row := 1180) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_10
  simp only [← c72822, ← c72823, ← c72824] at e_1180_10
  have e_1180_11 := arithEq_of_rows h (row := 1180) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_11
  simp only [← c72825, ← c72826, k2, ← c72827] at e_1180_11
  have hr := e_1180_11
  simp only [e_1180_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10451)) := by
  have c72828 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72829 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72830 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72831 := (publicBatchWrapper89_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72832 := (publicBatchWrapper89_copies2276 a h).1
  have c72833 := (publicBatchWrapper89_copies2276 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_12 := arithEq_of_rows h (row := 1180) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_12
  simp only [← c72828, ← c72829, ← c72830] at e_1180_12
  have e_1180_13 := arithEq_of_rows h (row := 1180) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_13
  simp only [← c72831, ← c72832, k2, ← c72833] at e_1180_13
  have hr := e_1180_13
  simp only [e_1180_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10452)) := by
  have c72834 := (publicBatchWrapper89_copies2276 a h).2.2.1
  have c72835 := (publicBatchWrapper89_copies2276 a h).2.2.2.1
  have c72836 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.1
  have c72837 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.1
  have c72838 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.1
  have c72839 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_14 := arithEq_of_rows h (row := 1180) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_14
  simp only [← c72834, ← c72835, ← c72836] at e_1180_14
  have e_1180_15 := arithEq_of_rows h (row := 1180) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_15
  simp only [← c72837, ← c72838, k2, ← c72839] at e_1180_15
  have hr := e_1180_15
  simp only [e_1180_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10453)) := by
  have c72840 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.1
  have c72841 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.1
  have c72842 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72843 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72844 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72845 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_16 := arithEq_of_rows h (row := 1180) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_16
  simp only [← c72840, ← c72841, ← c72842] at e_1180_16
  have e_1180_17 := arithEq_of_rows h (row := 1180) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_17
  simp only [← c72843, ← c72844, k2, ← c72845] at e_1180_17
  have hr := e_1180_17
  simp only [e_1180_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1180 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10454)) := by
  have c72846 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72847 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72848 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72849 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72850 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72851 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1180_18 := arithEq_of_rows h (row := 1180) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_18
  simp only [← c72846, ← c72847, ← c72848] at e_1180_18
  have e_1180_19 := arithEq_of_rows h (row := 1180) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_19
  simp only [← c72849, ← c72850, k2, ← c72851] at e_1180_19
  have hr := e_1180_19
  simp only [e_1180_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10455)) := by
  have c72852 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72853 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72854 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72855 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72856 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72857 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_0 := arithEq_of_rows h (row := 1181) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_0
  simp only [← c72852, ← c72853, ← c72854] at e_1181_0
  have e_1181_1 := arithEq_of_rows h (row := 1181) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_1
  simp only [← c72855, ← c72856, k2, ← c72857] at e_1181_1
  have hr := e_1181_1
  simp only [e_1181_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10456)) := by
  have c72858 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72859 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72860 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72861 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72862 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72863 := (publicBatchWrapper89_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_2 := arithEq_of_rows h (row := 1181) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_2
  simp only [← c72858, ← c72859, ← c72860] at e_1181_2
  have e_1181_3 := arithEq_of_rows h (row := 1181) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_3
  simp only [← c72861, ← c72862, k2, ← c72863] at e_1181_3
  have hr := e_1181_3
  simp only [e_1181_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10457)) := by
  have c72864 := (publicBatchWrapper89_copies2277 a h).1
  have c72865 := (publicBatchWrapper89_copies2277 a h).2.1
  have c72866 := (publicBatchWrapper89_copies2277 a h).2.2.1
  have c72867 := (publicBatchWrapper89_copies2277 a h).2.2.2.1
  have c72868 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.1
  have c72869 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_4 := arithEq_of_rows h (row := 1181) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_4
  simp only [← c72864, ← c72865, ← c72866] at e_1181_4
  have e_1181_5 := arithEq_of_rows h (row := 1181) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_5
  simp only [← c72867, ← c72868, k2, ← c72869] at e_1181_5
  have hr := e_1181_5
  simp only [e_1181_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10458)) := by
  have c72870 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.1
  have c72871 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.1
  have c72872 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.1
  have c72873 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.1
  have c72874 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72875 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_6 := arithEq_of_rows h (row := 1181) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_6
  simp only [← c72870, ← c72871, ← c72872] at e_1181_6
  have e_1181_7 := arithEq_of_rows h (row := 1181) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_7
  simp only [← c72873, ← c72874, k2, ← c72875] at e_1181_7
  have hr := e_1181_7
  simp only [e_1181_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10459)) := by
  have c72876 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72877 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72878 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72879 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72880 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72881 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_8 := arithEq_of_rows h (row := 1181) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_8
  simp only [← c72876, ← c72877, ← c72878] at e_1181_8
  have e_1181_9 := arithEq_of_rows h (row := 1181) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_9
  simp only [← c72879, ← c72880, k2, ← c72881] at e_1181_9
  have hr := e_1181_9
  simp only [e_1181_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10460)) := by
  have c72882 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72883 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72884 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72885 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72886 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72887 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_10 := arithEq_of_rows h (row := 1181) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_10
  simp only [← c72882, ← c72883, ← c72884] at e_1181_10
  have e_1181_11 := arithEq_of_rows h (row := 1181) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_11
  simp only [← c72885, ← c72886, k2, ← c72887] at e_1181_11
  have hr := e_1181_11
  simp only [e_1181_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10461)) := by
  have c72888 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72889 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72890 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72891 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72892 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72893 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_12 := arithEq_of_rows h (row := 1181) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_12
  simp only [← c72888, ← c72889, ← c72890] at e_1181_12
  have e_1181_13 := arithEq_of_rows h (row := 1181) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_13
  simp only [← c72891, ← c72892, k2, ← c72893] at e_1181_13
  have hr := e_1181_13
  simp only [e_1181_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10462)) := by
  have c72894 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72895 := (publicBatchWrapper89_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72896 := (publicBatchWrapper89_copies2278 a h).1
  have c72897 := (publicBatchWrapper89_copies2278 a h).2.1
  have c72898 := (publicBatchWrapper89_copies2278 a h).2.2.1
  have c72899 := (publicBatchWrapper89_copies2278 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_14 := arithEq_of_rows h (row := 1181) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_14
  simp only [← c72894, ← c72895, ← c72896] at e_1181_14
  have e_1181_15 := arithEq_of_rows h (row := 1181) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_15
  simp only [← c72897, ← c72898, k2, ← c72899] at e_1181_15
  have hr := e_1181_15
  simp only [e_1181_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10463)) := by
  have c72900 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.1
  have c72901 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.1
  have c72902 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.1
  have c72903 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.1
  have c72904 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.1
  have c72905 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_16 := arithEq_of_rows h (row := 1181) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_16
  simp only [← c72900, ← c72901, ← c72902] at e_1181_16
  have e_1181_17 := arithEq_of_rows h (row := 1181) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_17
  simp only [← c72903, ← c72904, k2, ← c72905] at e_1181_17
  have hr := e_1181_17
  simp only [e_1181_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1181 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10464)) := by
  have c72906 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72907 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72908 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72909 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72910 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72911 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1181_18 := arithEq_of_rows h (row := 1181) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_18
  simp only [← c72906, ← c72907, ← c72908] at e_1181_18
  have e_1181_19 := arithEq_of_rows h (row := 1181) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_19
  simp only [← c72909, ← c72910, k2, ← c72911] at e_1181_19
  have hr := e_1181_19
  simp only [e_1181_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10465)) := by
  have c72912 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72913 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72914 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72915 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72916 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72917 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_0 := arithEq_of_rows h (row := 1182) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_0
  simp only [← c72912, ← c72913, ← c72914] at e_1182_0
  have e_1182_1 := arithEq_of_rows h (row := 1182) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_1
  simp only [← c72915, ← c72916, k2, ← c72917] at e_1182_1
  have hr := e_1182_1
  simp only [e_1182_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10466)) := by
  have c72918 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72919 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72920 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72921 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72922 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72923 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_2 := arithEq_of_rows h (row := 1182) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_2
  simp only [← c72918, ← c72919, ← c72920] at e_1182_2
  have e_1182_3 := arithEq_of_rows h (row := 1182) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_3
  simp only [← c72921, ← c72922, k2, ← c72923] at e_1182_3
  have hr := e_1182_3
  simp only [e_1182_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10467)) := by
  have c72924 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72925 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72926 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72927 := (publicBatchWrapper89_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72928 := (publicBatchWrapper89_copies2279 a h).1
  have c72929 := (publicBatchWrapper89_copies2279 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_4 := arithEq_of_rows h (row := 1182) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_4
  simp only [← c72924, ← c72925, ← c72926] at e_1182_4
  have e_1182_5 := arithEq_of_rows h (row := 1182) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_5
  simp only [← c72927, ← c72928, k2, ← c72929] at e_1182_5
  have hr := e_1182_5
  simp only [e_1182_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10468)) := by
  have c72930 := (publicBatchWrapper89_copies2279 a h).2.2.1
  have c72931 := (publicBatchWrapper89_copies2279 a h).2.2.2.1
  have c72932 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.1
  have c72933 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.1
  have c72934 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.1
  have c72935 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_6 := arithEq_of_rows h (row := 1182) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_6
  simp only [← c72930, ← c72931, ← c72932] at e_1182_6
  have e_1182_7 := arithEq_of_rows h (row := 1182) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_7
  simp only [← c72933, ← c72934, k2, ← c72935] at e_1182_7
  have hr := e_1182_7
  simp only [e_1182_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10469)) := by
  have c72936 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.1
  have c72937 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.1
  have c72938 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72939 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72940 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72941 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_8 := arithEq_of_rows h (row := 1182) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_8
  simp only [← c72936, ← c72937, ← c72938] at e_1182_8
  have e_1182_9 := arithEq_of_rows h (row := 1182) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_9
  simp only [← c72939, ← c72940, k2, ← c72941] at e_1182_9
  have hr := e_1182_9
  simp only [e_1182_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10470)) := by
  have c72942 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72943 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72944 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72945 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72946 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72947 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_10 := arithEq_of_rows h (row := 1182) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_10
  simp only [← c72942, ← c72943, ← c72944] at e_1182_10
  have e_1182_11 := arithEq_of_rows h (row := 1182) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_11
  simp only [← c72945, ← c72946, k2, ← c72947] at e_1182_11
  have hr := e_1182_11
  simp only [e_1182_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10471)) := by
  have c72948 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72949 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72950 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72951 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72952 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72953 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_12 := arithEq_of_rows h (row := 1182) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_12
  simp only [← c72948, ← c72949, ← c72950] at e_1182_12
  have e_1182_13 := arithEq_of_rows h (row := 1182) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_13
  simp only [← c72951, ← c72952, k2, ← c72953] at e_1182_13
  have hr := e_1182_13
  simp only [e_1182_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10472)) := by
  have c72954 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72955 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72956 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72957 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72958 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72959 := (publicBatchWrapper89_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_14 := arithEq_of_rows h (row := 1182) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_14
  simp only [← c72954, ← c72955, ← c72956] at e_1182_14
  have e_1182_15 := arithEq_of_rows h (row := 1182) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_15
  simp only [← c72957, ← c72958, k2, ← c72959] at e_1182_15
  have hr := e_1182_15
  simp only [e_1182_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10473)) := by
  have c72960 := (publicBatchWrapper89_copies2280 a h).1
  have c72961 := (publicBatchWrapper89_copies2280 a h).2.1
  have c72962 := (publicBatchWrapper89_copies2280 a h).2.2.1
  have c72963 := (publicBatchWrapper89_copies2280 a h).2.2.2.1
  have c72964 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.1
  have c72965 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_16 := arithEq_of_rows h (row := 1182) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_16
  simp only [← c72960, ← c72961, ← c72962] at e_1182_16
  have e_1182_17 := arithEq_of_rows h (row := 1182) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_17
  simp only [← c72963, ← c72964, k2, ← c72965] at e_1182_17
  have hr := e_1182_17
  simp only [e_1182_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1182 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10474)) := by
  have c72966 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.1
  have c72967 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.1
  have c72968 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.1
  have c72969 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.1
  have c72970 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72971 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1182_18 := arithEq_of_rows h (row := 1182) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_18
  simp only [← c72966, ← c72967, ← c72968] at e_1182_18
  have e_1182_19 := arithEq_of_rows h (row := 1182) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_19
  simp only [← c72969, ← c72970, k2, ← c72971] at e_1182_19
  have hr := e_1182_19
  simp only [e_1182_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10475)) := by
  have c72972 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72973 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72974 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72975 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72976 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72977 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_0 := arithEq_of_rows h (row := 1183) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_0
  simp only [← c72972, ← c72973, ← c72974] at e_1183_0
  have e_1183_1 := arithEq_of_rows h (row := 1183) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_1
  simp only [← c72975, ← c72976, k2, ← c72977] at e_1183_1
  have hr := e_1183_1
  simp only [e_1183_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10476)) := by
  have c72978 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72979 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72980 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72981 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72982 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72983 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_2 := arithEq_of_rows h (row := 1183) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_2
  simp only [← c72978, ← c72979, ← c72980] at e_1183_2
  have e_1183_3 := arithEq_of_rows h (row := 1183) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_3
  simp only [← c72981, ← c72982, k2, ← c72983] at e_1183_3
  have hr := e_1183_3
  simp only [e_1183_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10477)) := by
  have c72984 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72985 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72986 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72987 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72988 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72989 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_4 := arithEq_of_rows h (row := 1183) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_4
  simp only [← c72984, ← c72985, ← c72986] at e_1183_4
  have e_1183_5 := arithEq_of_rows h (row := 1183) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_5
  simp only [← c72987, ← c72988, k2, ← c72989] at e_1183_5
  have hr := e_1183_5
  simp only [e_1183_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10478)) := by
  have c72990 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72991 := (publicBatchWrapper89_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72992 := (publicBatchWrapper89_copies2281 a h).1
  have c72993 := (publicBatchWrapper89_copies2281 a h).2.1
  have c72994 := (publicBatchWrapper89_copies2281 a h).2.2.1
  have c72995 := (publicBatchWrapper89_copies2281 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_6 := arithEq_of_rows h (row := 1183) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_6
  simp only [← c72990, ← c72991, ← c72992] at e_1183_6
  have e_1183_7 := arithEq_of_rows h (row := 1183) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_7
  simp only [← c72993, ← c72994, k2, ← c72995] at e_1183_7
  have hr := e_1183_7
  simp only [e_1183_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10479)) := by
  have c72996 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.1
  have c72997 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.1
  have c72998 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.1
  have c72999 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.1
  have c73000 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.1
  have c73001 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_8 := arithEq_of_rows h (row := 1183) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_8
  simp only [← c72996, ← c72997, ← c72998] at e_1183_8
  have e_1183_9 := arithEq_of_rows h (row := 1183) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_9
  simp only [← c72999, ← c73000, k2, ← c73001] at e_1183_9
  have hr := e_1183_9
  simp only [e_1183_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10480)) := by
  have c73002 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73003 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73004 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73005 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73006 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73007 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_10 := arithEq_of_rows h (row := 1183) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_10
  simp only [← c73002, ← c73003, ← c73004] at e_1183_10
  have e_1183_11 := arithEq_of_rows h (row := 1183) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_11
  simp only [← c73005, ← c73006, k2, ← c73007] at e_1183_11
  have hr := e_1183_11
  simp only [e_1183_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10481)) := by
  have c73008 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73009 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73010 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73011 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73012 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73013 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_12 := arithEq_of_rows h (row := 1183) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_12
  simp only [← c73008, ← c73009, ← c73010] at e_1183_12
  have e_1183_13 := arithEq_of_rows h (row := 1183) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_13
  simp only [← c73011, ← c73012, k2, ← c73013] at e_1183_13
  have hr := e_1183_13
  simp only [e_1183_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10482)) := by
  have c73014 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73015 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73016 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73017 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73018 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73019 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_14 := arithEq_of_rows h (row := 1183) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_14
  simp only [← c73014, ← c73015, ← c73016] at e_1183_14
  have e_1183_15 := arithEq_of_rows h (row := 1183) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_15
  simp only [← c73017, ← c73018, k2, ← c73019] at e_1183_15
  have hr := e_1183_15
  simp only [e_1183_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10483)) := by
  have c73020 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73021 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73022 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73023 := (publicBatchWrapper89_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73024 := (publicBatchWrapper89_copies2282 a h).1
  have c73025 := (publicBatchWrapper89_copies2282 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_16 := arithEq_of_rows h (row := 1183) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_16
  simp only [← c73020, ← c73021, ← c73022] at e_1183_16
  have e_1183_17 := arithEq_of_rows h (row := 1183) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_17
  simp only [← c73023, ← c73024, k2, ← c73025] at e_1183_17
  have hr := e_1183_17
  simp only [e_1183_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1183 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10484)) := by
  have c73026 := (publicBatchWrapper89_copies2282 a h).2.2.1
  have c73027 := (publicBatchWrapper89_copies2282 a h).2.2.2.1
  have c73028 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.1
  have c73029 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.1
  have c73030 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.1
  have c73031 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1183_18 := arithEq_of_rows h (row := 1183) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_18
  simp only [← c73026, ← c73027, ← c73028] at e_1183_18
  have e_1183_19 := arithEq_of_rows h (row := 1183) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_19
  simp only [← c73029, ← c73030, k2, ← c73031] at e_1183_19
  have hr := e_1183_19
  simp only [e_1183_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10485)) := by
  have c73032 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.1
  have c73033 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.1
  have c73034 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73035 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73036 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73037 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_0 := arithEq_of_rows h (row := 1184) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_0
  simp only [← c73032, ← c73033, ← c73034] at e_1184_0
  have e_1184_1 := arithEq_of_rows h (row := 1184) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_1
  simp only [← c73035, ← c73036, k2, ← c73037] at e_1184_1
  have hr := e_1184_1
  simp only [e_1184_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10486)) := by
  have c73038 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73039 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73040 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73041 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73042 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73043 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_2 := arithEq_of_rows h (row := 1184) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_2
  simp only [← c73038, ← c73039, ← c73040] at e_1184_2
  have e_1184_3 := arithEq_of_rows h (row := 1184) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_3
  simp only [← c73041, ← c73042, k2, ← c73043] at e_1184_3
  have hr := e_1184_3
  simp only [e_1184_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10487)) := by
  have c73044 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73045 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73046 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73047 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73048 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73049 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_4 := arithEq_of_rows h (row := 1184) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_4
  simp only [← c73044, ← c73045, ← c73046] at e_1184_4
  have e_1184_5 := arithEq_of_rows h (row := 1184) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_5
  simp only [← c73047, ← c73048, k2, ← c73049] at e_1184_5
  have hr := e_1184_5
  simp only [e_1184_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10488)) := by
  have c73050 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73051 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73052 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73053 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73054 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73055 := (publicBatchWrapper89_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_6 := arithEq_of_rows h (row := 1184) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_6
  simp only [← c73050, ← c73051, ← c73052] at e_1184_6
  have e_1184_7 := arithEq_of_rows h (row := 1184) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_7
  simp only [← c73053, ← c73054, k2, ← c73055] at e_1184_7
  have hr := e_1184_7
  simp only [e_1184_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10489)) := by
  have c73056 := (publicBatchWrapper89_copies2283 a h).1
  have c73057 := (publicBatchWrapper89_copies2283 a h).2.1
  have c73058 := (publicBatchWrapper89_copies2283 a h).2.2.1
  have c73059 := (publicBatchWrapper89_copies2283 a h).2.2.2.1
  have c73060 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.1
  have c73061 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_8 := arithEq_of_rows h (row := 1184) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_8
  simp only [← c73056, ← c73057, ← c73058] at e_1184_8
  have e_1184_9 := arithEq_of_rows h (row := 1184) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_9
  simp only [← c73059, ← c73060, k2, ← c73061] at e_1184_9
  have hr := e_1184_9
  simp only [e_1184_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10490)) := by
  have c73062 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.1
  have c73063 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.1
  have c73064 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.1
  have c73065 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.1
  have c73066 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73067 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_10 := arithEq_of_rows h (row := 1184) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_10
  simp only [← c73062, ← c73063, ← c73064] at e_1184_10
  have e_1184_11 := arithEq_of_rows h (row := 1184) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_11
  simp only [← c73065, ← c73066, k2, ← c73067] at e_1184_11
  have hr := e_1184_11
  simp only [e_1184_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10491)) := by
  have c73068 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73069 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73070 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73071 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73072 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73073 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_12 := arithEq_of_rows h (row := 1184) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_12
  simp only [← c73068, ← c73069, ← c73070] at e_1184_12
  have e_1184_13 := arithEq_of_rows h (row := 1184) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_13
  simp only [← c73071, ← c73072, k2, ← c73073] at e_1184_13
  have hr := e_1184_13
  simp only [e_1184_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10492)) := by
  have c73074 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73075 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73076 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73077 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73078 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73079 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_14 := arithEq_of_rows h (row := 1184) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_14
  simp only [← c73074, ← c73075, ← c73076] at e_1184_14
  have e_1184_15 := arithEq_of_rows h (row := 1184) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_15
  simp only [← c73077, ← c73078, k2, ← c73079] at e_1184_15
  have hr := e_1184_15
  simp only [e_1184_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10493)) := by
  have c73080 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73081 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73082 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73083 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73084 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73085 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_16 := arithEq_of_rows h (row := 1184) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_16
  simp only [← c73080, ← c73081, ← c73082] at e_1184_16
  have e_1184_17 := arithEq_of_rows h (row := 1184) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_17
  simp only [← c73083, ← c73084, k2, ← c73085] at e_1184_17
  have hr := e_1184_17
  simp only [e_1184_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1184 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10494)) := by
  have c73086 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73087 := (publicBatchWrapper89_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73088 := (publicBatchWrapper89_copies2284 a h).1
  have c73089 := (publicBatchWrapper89_copies2284 a h).2.1
  have c73090 := (publicBatchWrapper89_copies2284 a h).2.2.1
  have c73091 := (publicBatchWrapper89_copies2284 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1184_18 := arithEq_of_rows h (row := 1184) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_18
  simp only [← c73086, ← c73087, ← c73088] at e_1184_18
  have e_1184_19 := arithEq_of_rows h (row := 1184) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_19
  simp only [← c73089, ← c73090, k2, ← c73091] at e_1184_19
  have hr := e_1184_19
  simp only [e_1184_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10495)) := by
  have c73092 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.1
  have c73093 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.1
  have c73094 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.1
  have c73095 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.1
  have c73096 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.1
  have c73097 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_0 := arithEq_of_rows h (row := 1185) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_0
  simp only [← c73092, ← c73093, ← c73094] at e_1185_0
  have e_1185_1 := arithEq_of_rows h (row := 1185) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_1
  simp only [← c73095, ← c73096, k2, ← c73097] at e_1185_1
  have hr := e_1185_1
  simp only [e_1185_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10496)) := by
  have c73098 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73099 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73100 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73101 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73102 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73103 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_2 := arithEq_of_rows h (row := 1185) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_2
  simp only [← c73098, ← c73099, ← c73100] at e_1185_2
  have e_1185_3 := arithEq_of_rows h (row := 1185) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_3
  simp only [← c73101, ← c73102, k2, ← c73103] at e_1185_3
  have hr := e_1185_3
  simp only [e_1185_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10497)) := by
  have c73104 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73105 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73106 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73107 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73108 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73109 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_4 := arithEq_of_rows h (row := 1185) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_4
  simp only [← c73104, ← c73105, ← c73106] at e_1185_4
  have e_1185_5 := arithEq_of_rows h (row := 1185) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_5
  simp only [← c73107, ← c73108, k2, ← c73109] at e_1185_5
  have hr := e_1185_5
  simp only [e_1185_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10498)) := by
  have c73110 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73111 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73112 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73113 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73114 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73115 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_6 := arithEq_of_rows h (row := 1185) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_6
  simp only [← c73110, ← c73111, ← c73112] at e_1185_6
  have e_1185_7 := arithEq_of_rows h (row := 1185) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_7
  simp only [← c73113, ← c73114, k2, ← c73115] at e_1185_7
  have hr := e_1185_7
  simp only [e_1185_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10499)) := by
  have c73116 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73117 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73118 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73119 := (publicBatchWrapper89_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73120 := (publicBatchWrapper89_copies2285 a h).1
  have c73121 := (publicBatchWrapper89_copies2285 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_8 := arithEq_of_rows h (row := 1185) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_8
  simp only [← c73116, ← c73117, ← c73118] at e_1185_8
  have e_1185_9 := arithEq_of_rows h (row := 1185) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_9
  simp only [← c73119, ← c73120, k2, ← c73121] at e_1185_9
  have hr := e_1185_9
  simp only [e_1185_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10500)) := by
  have c73122 := (publicBatchWrapper89_copies2285 a h).2.2.1
  have c73123 := (publicBatchWrapper89_copies2285 a h).2.2.2.1
  have c73124 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.1
  have c73125 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.1
  have c73126 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.1
  have c73127 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_10 := arithEq_of_rows h (row := 1185) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_10
  simp only [← c73122, ← c73123, ← c73124] at e_1185_10
  have e_1185_11 := arithEq_of_rows h (row := 1185) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_11
  simp only [← c73125, ← c73126, k2, ← c73127] at e_1185_11
  have hr := e_1185_11
  simp only [e_1185_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10501)) := by
  have c73128 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.1
  have c73129 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.1
  have c73130 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73131 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73132 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73133 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_12 := arithEq_of_rows h (row := 1185) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_12
  simp only [← c73128, ← c73129, ← c73130] at e_1185_12
  have e_1185_13 := arithEq_of_rows h (row := 1185) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_13
  simp only [← c73131, ← c73132, k2, ← c73133] at e_1185_13
  have hr := e_1185_13
  simp only [e_1185_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10502)) := by
  have c73134 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73135 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73136 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73137 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73138 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73139 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_14 := arithEq_of_rows h (row := 1185) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_14
  simp only [← c73134, ← c73135, ← c73136] at e_1185_14
  have e_1185_15 := arithEq_of_rows h (row := 1185) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_15
  simp only [← c73137, ← c73138, k2, ← c73139] at e_1185_15
  have hr := e_1185_15
  simp only [e_1185_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10503)) := by
  have c73140 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73141 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73142 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73143 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73144 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73145 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_16 := arithEq_of_rows h (row := 1185) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_16
  simp only [← c73140, ← c73141, ← c73142] at e_1185_16
  have e_1185_17 := arithEq_of_rows h (row := 1185) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_17
  simp only [← c73143, ← c73144, k2, ← c73145] at e_1185_17
  have hr := e_1185_17
  simp only [e_1185_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1185 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10504)) := by
  have c73146 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73147 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73148 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73149 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73150 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73151 := (publicBatchWrapper89_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1185_18 := arithEq_of_rows h (row := 1185) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_18
  simp only [← c73146, ← c73147, ← c73148] at e_1185_18
  have e_1185_19 := arithEq_of_rows h (row := 1185) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_19
  simp only [← c73149, ← c73150, k2, ← c73151] at e_1185_19
  have hr := e_1185_19
  simp only [e_1185_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10505)) := by
  have c73152 := (publicBatchWrapper89_copies2286 a h).1
  have c73153 := (publicBatchWrapper89_copies2286 a h).2.1
  have c73154 := (publicBatchWrapper89_copies2286 a h).2.2.1
  have c73155 := (publicBatchWrapper89_copies2286 a h).2.2.2.1
  have c73156 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.1
  have c73157 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_0 := arithEq_of_rows h (row := 1186) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_0
  simp only [← c73152, ← c73153, ← c73154] at e_1186_0
  have e_1186_1 := arithEq_of_rows h (row := 1186) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_1
  simp only [← c73155, ← c73156, k2, ← c73157] at e_1186_1
  have hr := e_1186_1
  simp only [e_1186_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10506)) := by
  have c73158 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.1
  have c73159 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.1
  have c73160 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.1
  have c73161 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.1
  have c73162 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73163 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_2 := arithEq_of_rows h (row := 1186) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_2
  simp only [← c73158, ← c73159, ← c73160] at e_1186_2
  have e_1186_3 := arithEq_of_rows h (row := 1186) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_3
  simp only [← c73161, ← c73162, k2, ← c73163] at e_1186_3
  have hr := e_1186_3
  simp only [e_1186_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10507)) := by
  have c73164 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73165 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73166 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73167 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73168 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73169 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_4 := arithEq_of_rows h (row := 1186) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_4
  simp only [← c73164, ← c73165, ← c73166] at e_1186_4
  have e_1186_5 := arithEq_of_rows h (row := 1186) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_5
  simp only [← c73167, ← c73168, k2, ← c73169] at e_1186_5
  have hr := e_1186_5
  simp only [e_1186_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10508)) := by
  have c73170 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73171 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73172 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73173 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73174 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73175 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_6 := arithEq_of_rows h (row := 1186) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_6
  simp only [← c73170, ← c73171, ← c73172] at e_1186_6
  have e_1186_7 := arithEq_of_rows h (row := 1186) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_7
  simp only [← c73173, ← c73174, k2, ← c73175] at e_1186_7
  have hr := e_1186_7
  simp only [e_1186_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10509)) := by
  have c73176 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73177 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73178 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73179 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73180 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73181 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_8 := arithEq_of_rows h (row := 1186) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_8
  simp only [← c73176, ← c73177, ← c73178] at e_1186_8
  have e_1186_9 := arithEq_of_rows h (row := 1186) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_9
  simp only [← c73179, ← c73180, k2, ← c73181] at e_1186_9
  have hr := e_1186_9
  simp only [e_1186_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10510)) := by
  have c73182 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73183 := (publicBatchWrapper89_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73184 := (publicBatchWrapper89_copies2287 a h).1
  have c73185 := (publicBatchWrapper89_copies2287 a h).2.1
  have c73186 := (publicBatchWrapper89_copies2287 a h).2.2.1
  have c73187 := (publicBatchWrapper89_copies2287 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_10 := arithEq_of_rows h (row := 1186) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_10
  simp only [← c73182, ← c73183, ← c73184] at e_1186_10
  have e_1186_11 := arithEq_of_rows h (row := 1186) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_11
  simp only [← c73185, ← c73186, k2, ← c73187] at e_1186_11
  have hr := e_1186_11
  simp only [e_1186_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10511)) := by
  have c73188 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.1
  have c73189 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.1
  have c73190 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.1
  have c73191 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.1
  have c73192 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.1
  have c73193 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_12 := arithEq_of_rows h (row := 1186) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_12
  simp only [← c73188, ← c73189, ← c73190] at e_1186_12
  have e_1186_13 := arithEq_of_rows h (row := 1186) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_13
  simp only [← c73191, ← c73192, k2, ← c73193] at e_1186_13
  have hr := e_1186_13
  simp only [e_1186_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10512)) := by
  have c73194 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73195 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73196 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73197 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73198 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73199 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_14 := arithEq_of_rows h (row := 1186) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_14
  simp only [← c73194, ← c73195, ← c73196] at e_1186_14
  have e_1186_15 := arithEq_of_rows h (row := 1186) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_15
  simp only [← c73197, ← c73198, k2, ← c73199] at e_1186_15
  have hr := e_1186_15
  simp only [e_1186_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10513)) := by
  have c73200 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73201 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73202 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73203 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73204 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73205 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_16 := arithEq_of_rows h (row := 1186) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_16
  simp only [← c73200, ← c73201, ← c73202] at e_1186_16
  have e_1186_17 := arithEq_of_rows h (row := 1186) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_17
  simp only [← c73203, ← c73204, k2, ← c73205] at e_1186_17
  have hr := e_1186_17
  simp only [e_1186_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1186 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10514)) := by
  have c73206 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73207 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73208 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73209 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73210 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73211 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1186_18 := arithEq_of_rows h (row := 1186) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_18
  simp only [← c73206, ← c73207, ← c73208] at e_1186_18
  have e_1186_19 := arithEq_of_rows h (row := 1186) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_19
  simp only [← c73209, ← c73210, k2, ← c73211] at e_1186_19
  have hr := e_1186_19
  simp only [e_1186_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10515)) := by
  have c73212 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73213 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73214 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73215 := (publicBatchWrapper89_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73216 := (publicBatchWrapper89_copies2288 a h).1
  have c73217 := (publicBatchWrapper89_copies2288 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_0 := arithEq_of_rows h (row := 1187) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_0
  simp only [← c73212, ← c73213, ← c73214] at e_1187_0
  have e_1187_1 := arithEq_of_rows h (row := 1187) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_1
  simp only [← c73215, ← c73216, k2, ← c73217] at e_1187_1
  have hr := e_1187_1
  simp only [e_1187_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10516)) := by
  have c73218 := (publicBatchWrapper89_copies2288 a h).2.2.1
  have c73219 := (publicBatchWrapper89_copies2288 a h).2.2.2.1
  have c73220 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.1
  have c73221 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.1
  have c73222 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.1
  have c73223 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_2 := arithEq_of_rows h (row := 1187) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_2
  simp only [← c73218, ← c73219, ← c73220] at e_1187_2
  have e_1187_3 := arithEq_of_rows h (row := 1187) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_3
  simp only [← c73221, ← c73222, k2, ← c73223] at e_1187_3
  have hr := e_1187_3
  simp only [e_1187_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10517)) := by
  have c73224 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.1
  have c73225 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.1
  have c73226 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73227 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73228 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73229 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_4 := arithEq_of_rows h (row := 1187) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_4
  simp only [← c73224, ← c73225, ← c73226] at e_1187_4
  have e_1187_5 := arithEq_of_rows h (row := 1187) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_5
  simp only [← c73227, ← c73228, k2, ← c73229] at e_1187_5
  have hr := e_1187_5
  simp only [e_1187_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10518)) := by
  have c73230 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73231 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73232 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73233 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73234 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73235 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_6 := arithEq_of_rows h (row := 1187) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_6
  simp only [← c73230, ← c73231, ← c73232] at e_1187_6
  have e_1187_7 := arithEq_of_rows h (row := 1187) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_7
  simp only [← c73233, ← c73234, k2, ← c73235] at e_1187_7
  have hr := e_1187_7
  simp only [e_1187_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10519)) := by
  have c73236 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73237 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73238 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73239 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73240 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73241 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_8 := arithEq_of_rows h (row := 1187) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_8
  simp only [← c73236, ← c73237, ← c73238] at e_1187_8
  have e_1187_9 := arithEq_of_rows h (row := 1187) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_9
  simp only [← c73239, ← c73240, k2, ← c73241] at e_1187_9
  have hr := e_1187_9
  simp only [e_1187_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10520)) := by
  have c73242 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73243 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73244 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73245 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73246 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73247 := (publicBatchWrapper89_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_10 := arithEq_of_rows h (row := 1187) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_10
  simp only [← c73242, ← c73243, ← c73244] at e_1187_10
  have e_1187_11 := arithEq_of_rows h (row := 1187) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_11
  simp only [← c73245, ← c73246, k2, ← c73247] at e_1187_11
  have hr := e_1187_11
  simp only [e_1187_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10521)) := by
  have c73248 := (publicBatchWrapper89_copies2289 a h).1
  have c73249 := (publicBatchWrapper89_copies2289 a h).2.1
  have c73250 := (publicBatchWrapper89_copies2289 a h).2.2.1
  have c73251 := (publicBatchWrapper89_copies2289 a h).2.2.2.1
  have c73252 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.1
  have c73253 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_12 := arithEq_of_rows h (row := 1187) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_12
  simp only [← c73248, ← c73249, ← c73250] at e_1187_12
  have e_1187_13 := arithEq_of_rows h (row := 1187) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_13
  simp only [← c73251, ← c73252, k2, ← c73253] at e_1187_13
  have hr := e_1187_13
  simp only [e_1187_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10522)) := by
  have c73254 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.1
  have c73255 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.1
  have c73256 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.1
  have c73257 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.1
  have c73258 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73259 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_14 := arithEq_of_rows h (row := 1187) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_14
  simp only [← c73254, ← c73255, ← c73256] at e_1187_14
  have e_1187_15 := arithEq_of_rows h (row := 1187) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_15
  simp only [← c73257, ← c73258, k2, ← c73259] at e_1187_15
  have hr := e_1187_15
  simp only [e_1187_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10523)) := by
  have c73260 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73261 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73262 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73263 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73264 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73265 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_16 := arithEq_of_rows h (row := 1187) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_16
  simp only [← c73260, ← c73261, ← c73262] at e_1187_16
  have e_1187_17 := arithEq_of_rows h (row := 1187) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_17
  simp only [← c73263, ← c73264, k2, ← c73265] at e_1187_17
  have hr := e_1187_17
  simp only [e_1187_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1187 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10524)) := by
  have c73266 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73267 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73268 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73269 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73270 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73271 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1187_18 := arithEq_of_rows h (row := 1187) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_18
  simp only [← c73266, ← c73267, ← c73268] at e_1187_18
  have e_1187_19 := arithEq_of_rows h (row := 1187) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_19
  simp only [← c73269, ← c73270, k2, ← c73271] at e_1187_19
  have hr := e_1187_19
  simp only [e_1187_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10525)) := by
  have c73272 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73273 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73274 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73275 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73276 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73277 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_0 := arithEq_of_rows h (row := 1188) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_0
  simp only [← c73272, ← c73273, ← c73274] at e_1188_0
  have e_1188_1 := arithEq_of_rows h (row := 1188) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_1
  simp only [← c73275, ← c73276, k2, ← c73277] at e_1188_1
  have hr := e_1188_1
  simp only [e_1188_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10526)) := by
  have c73278 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73279 := (publicBatchWrapper89_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73280 := (publicBatchWrapper89_copies2290 a h).1
  have c73281 := (publicBatchWrapper89_copies2290 a h).2.1
  have c73282 := (publicBatchWrapper89_copies2290 a h).2.2.1
  have c73283 := (publicBatchWrapper89_copies2290 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_2 := arithEq_of_rows h (row := 1188) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_2
  simp only [← c73278, ← c73279, ← c73280] at e_1188_2
  have e_1188_3 := arithEq_of_rows h (row := 1188) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_3
  simp only [← c73281, ← c73282, k2, ← c73283] at e_1188_3
  have hr := e_1188_3
  simp only [e_1188_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10527)) := by
  have c73284 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.1
  have c73285 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.1
  have c73286 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.1
  have c73287 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.1
  have c73288 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.1
  have c73289 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_4 := arithEq_of_rows h (row := 1188) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_4
  simp only [← c73284, ← c73285, ← c73286] at e_1188_4
  have e_1188_5 := arithEq_of_rows h (row := 1188) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_5
  simp only [← c73287, ← c73288, k2, ← c73289] at e_1188_5
  have hr := e_1188_5
  simp only [e_1188_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10528)) := by
  have c73290 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73291 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73292 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73293 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73294 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73295 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_6 := arithEq_of_rows h (row := 1188) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_6
  simp only [← c73290, ← c73291, ← c73292] at e_1188_6
  have e_1188_7 := arithEq_of_rows h (row := 1188) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_7
  simp only [← c73293, ← c73294, k2, ← c73295] at e_1188_7
  have hr := e_1188_7
  simp only [e_1188_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10529)) := by
  have c73296 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73297 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73298 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73299 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73300 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73301 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_8 := arithEq_of_rows h (row := 1188) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_8
  simp only [← c73296, ← c73297, ← c73298] at e_1188_8
  have e_1188_9 := arithEq_of_rows h (row := 1188) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_9
  simp only [← c73299, ← c73300, k2, ← c73301] at e_1188_9
  have hr := e_1188_9
  simp only [e_1188_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10530)) := by
  have c73302 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73303 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73304 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73305 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73306 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73307 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_10 := arithEq_of_rows h (row := 1188) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_10
  simp only [← c73302, ← c73303, ← c73304] at e_1188_10
  have e_1188_11 := arithEq_of_rows h (row := 1188) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_11
  simp only [← c73305, ← c73306, k2, ← c73307] at e_1188_11
  have hr := e_1188_11
  simp only [e_1188_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10531)) := by
  have c73308 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73309 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73310 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73311 := (publicBatchWrapper89_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73312 := (publicBatchWrapper89_copies2291 a h).1
  have c73313 := (publicBatchWrapper89_copies2291 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_12 := arithEq_of_rows h (row := 1188) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_12
  simp only [← c73308, ← c73309, ← c73310] at e_1188_12
  have e_1188_13 := arithEq_of_rows h (row := 1188) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_13
  simp only [← c73311, ← c73312, k2, ← c73313] at e_1188_13
  have hr := e_1188_13
  simp only [e_1188_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10532)) := by
  have c73314 := (publicBatchWrapper89_copies2291 a h).2.2.1
  have c73315 := (publicBatchWrapper89_copies2291 a h).2.2.2.1
  have c73316 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.1
  have c73317 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.1
  have c73318 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.1
  have c73319 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_14 := arithEq_of_rows h (row := 1188) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_14
  simp only [← c73314, ← c73315, ← c73316] at e_1188_14
  have e_1188_15 := arithEq_of_rows h (row := 1188) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_15
  simp only [← c73317, ← c73318, k2, ← c73319] at e_1188_15
  have hr := e_1188_15
  simp only [e_1188_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10533)) := by
  have c73320 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.1
  have c73321 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.1
  have c73322 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73323 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73324 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73325 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_16 := arithEq_of_rows h (row := 1188) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_16
  simp only [← c73320, ← c73321, ← c73322] at e_1188_16
  have e_1188_17 := arithEq_of_rows h (row := 1188) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_17
  simp only [← c73323, ← c73324, k2, ← c73325] at e_1188_17
  have hr := e_1188_17
  simp only [e_1188_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1188 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10534)) := by
  have c73326 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73327 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73328 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73329 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73330 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73331 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1188_18 := arithEq_of_rows h (row := 1188) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_18
  simp only [← c73326, ← c73327, ← c73328] at e_1188_18
  have e_1188_19 := arithEq_of_rows h (row := 1188) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_19
  simp only [← c73329, ← c73330, k2, ← c73331] at e_1188_19
  have hr := e_1188_19
  simp only [e_1188_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10535)) := by
  have c73332 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73333 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73334 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73335 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73336 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73337 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_0 := arithEq_of_rows h (row := 1189) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_0
  simp only [← c73332, ← c73333, ← c73334] at e_1189_0
  have e_1189_1 := arithEq_of_rows h (row := 1189) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_1
  simp only [← c73335, ← c73336, k2, ← c73337] at e_1189_1
  have hr := e_1189_1
  simp only [e_1189_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10536)) := by
  have c73338 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73339 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73340 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73341 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73342 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73343 := (publicBatchWrapper89_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_2 := arithEq_of_rows h (row := 1189) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_2
  simp only [← c73338, ← c73339, ← c73340] at e_1189_2
  have e_1189_3 := arithEq_of_rows h (row := 1189) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_3
  simp only [← c73341, ← c73342, k2, ← c73343] at e_1189_3
  have hr := e_1189_3
  simp only [e_1189_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10537)) := by
  have c73344 := (publicBatchWrapper89_copies2292 a h).1
  have c73345 := (publicBatchWrapper89_copies2292 a h).2.1
  have c73346 := (publicBatchWrapper89_copies2292 a h).2.2.1
  have c73347 := (publicBatchWrapper89_copies2292 a h).2.2.2.1
  have c73348 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.1
  have c73349 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_4 := arithEq_of_rows h (row := 1189) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_4
  simp only [← c73344, ← c73345, ← c73346] at e_1189_4
  have e_1189_5 := arithEq_of_rows h (row := 1189) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_5
  simp only [← c73347, ← c73348, k2, ← c73349] at e_1189_5
  have hr := e_1189_5
  simp only [e_1189_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10538)) := by
  have c73350 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.1
  have c73351 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.1
  have c73352 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.1
  have c73353 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.1
  have c73354 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73355 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_6 := arithEq_of_rows h (row := 1189) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_6
  simp only [← c73350, ← c73351, ← c73352] at e_1189_6
  have e_1189_7 := arithEq_of_rows h (row := 1189) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_7
  simp only [← c73353, ← c73354, k2, ← c73355] at e_1189_7
  have hr := e_1189_7
  simp only [e_1189_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10539)) := by
  have c73356 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73357 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73358 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73359 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73360 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73361 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_8 := arithEq_of_rows h (row := 1189) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_8
  simp only [← c73356, ← c73357, ← c73358] at e_1189_8
  have e_1189_9 := arithEq_of_rows h (row := 1189) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_9
  simp only [← c73359, ← c73360, k2, ← c73361] at e_1189_9
  have hr := e_1189_9
  simp only [e_1189_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10540)) := by
  have c73362 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73363 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73364 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73365 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73366 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73367 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_10 := arithEq_of_rows h (row := 1189) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_10
  simp only [← c73362, ← c73363, ← c73364] at e_1189_10
  have e_1189_11 := arithEq_of_rows h (row := 1189) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_11
  simp only [← c73365, ← c73366, k2, ← c73367] at e_1189_11
  have hr := e_1189_11
  simp only [e_1189_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10541)) := by
  have c73368 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73369 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73370 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73371 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73372 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73373 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_12 := arithEq_of_rows h (row := 1189) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_12
  simp only [← c73368, ← c73369, ← c73370] at e_1189_12
  have e_1189_13 := arithEq_of_rows h (row := 1189) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_13
  simp only [← c73371, ← c73372, k2, ← c73373] at e_1189_13
  have hr := e_1189_13
  simp only [e_1189_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10542)) := by
  have c73374 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73375 := (publicBatchWrapper89_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73376 := (publicBatchWrapper89_copies2293 a h).1
  have c73377 := (publicBatchWrapper89_copies2293 a h).2.1
  have c73378 := (publicBatchWrapper89_copies2293 a h).2.2.1
  have c73379 := (publicBatchWrapper89_copies2293 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_14 := arithEq_of_rows h (row := 1189) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_14
  simp only [← c73374, ← c73375, ← c73376] at e_1189_14
  have e_1189_15 := arithEq_of_rows h (row := 1189) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_15
  simp only [← c73377, ← c73378, k2, ← c73379] at e_1189_15
  have hr := e_1189_15
  simp only [e_1189_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10543)) := by
  have c73380 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.1
  have c73381 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.1
  have c73382 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.1
  have c73383 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.1
  have c73384 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.1
  have c73385 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_16 := arithEq_of_rows h (row := 1189) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_16
  simp only [← c73380, ← c73381, ← c73382] at e_1189_16
  have e_1189_17 := arithEq_of_rows h (row := 1189) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_17
  simp only [← c73383, ← c73384, k2, ← c73385] at e_1189_17
  have hr := e_1189_17
  simp only [e_1189_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1189 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10544)) := by
  have c73386 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73387 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73388 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73389 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73390 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73391 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1189_18 := arithEq_of_rows h (row := 1189) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_18
  simp only [← c73386, ← c73387, ← c73388] at e_1189_18
  have e_1189_19 := arithEq_of_rows h (row := 1189) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_19
  simp only [← c73389, ← c73390, k2, ← c73391] at e_1189_19
  have hr := e_1189_19
  simp only [e_1189_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10545)) := by
  have c73392 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73393 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73394 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73395 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73396 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73397 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_0 := arithEq_of_rows h (row := 1190) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_0
  simp only [← c73392, ← c73393, ← c73394] at e_1190_0
  have e_1190_1 := arithEq_of_rows h (row := 1190) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_1
  simp only [← c73395, ← c73396, k2, ← c73397] at e_1190_1
  have hr := e_1190_1
  simp only [e_1190_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10546)) := by
  have c73398 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73399 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73400 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73401 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73402 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73403 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_2 := arithEq_of_rows h (row := 1190) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_2
  simp only [← c73398, ← c73399, ← c73400] at e_1190_2
  have e_1190_3 := arithEq_of_rows h (row := 1190) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_3
  simp only [← c73401, ← c73402, k2, ← c73403] at e_1190_3
  have hr := e_1190_3
  simp only [e_1190_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10547)) := by
  have c73404 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73405 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73406 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73407 := (publicBatchWrapper89_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73408 := (publicBatchWrapper89_copies2294 a h).1
  have c73409 := (publicBatchWrapper89_copies2294 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_4 := arithEq_of_rows h (row := 1190) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_4
  simp only [← c73404, ← c73405, ← c73406] at e_1190_4
  have e_1190_5 := arithEq_of_rows h (row := 1190) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_5
  simp only [← c73407, ← c73408, k2, ← c73409] at e_1190_5
  have hr := e_1190_5
  simp only [e_1190_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10548)) := by
  have c73410 := (publicBatchWrapper89_copies2294 a h).2.2.1
  have c73411 := (publicBatchWrapper89_copies2294 a h).2.2.2.1
  have c73412 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.1
  have c73413 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.1
  have c73414 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.1
  have c73415 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_6 := arithEq_of_rows h (row := 1190) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_6
  simp only [← c73410, ← c73411, ← c73412] at e_1190_6
  have e_1190_7 := arithEq_of_rows h (row := 1190) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_7
  simp only [← c73413, ← c73414, k2, ← c73415] at e_1190_7
  have hr := e_1190_7
  simp only [e_1190_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10549)) := by
  have c73416 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.1
  have c73417 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.1
  have c73418 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73419 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73420 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73421 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_8 := arithEq_of_rows h (row := 1190) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_8
  simp only [← c73416, ← c73417, ← c73418] at e_1190_8
  have e_1190_9 := arithEq_of_rows h (row := 1190) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_9
  simp only [← c73419, ← c73420, k2, ← c73421] at e_1190_9
  have hr := e_1190_9
  simp only [e_1190_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10550)) := by
  have c73422 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73423 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73424 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73425 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73426 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73427 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_10 := arithEq_of_rows h (row := 1190) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_10
  simp only [← c73422, ← c73423, ← c73424] at e_1190_10
  have e_1190_11 := arithEq_of_rows h (row := 1190) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_11
  simp only [← c73425, ← c73426, k2, ← c73427] at e_1190_11
  have hr := e_1190_11
  simp only [e_1190_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10551)) := by
  have c73428 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73429 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73430 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73431 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73432 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73433 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_12 := arithEq_of_rows h (row := 1190) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_12
  simp only [← c73428, ← c73429, ← c73430] at e_1190_12
  have e_1190_13 := arithEq_of_rows h (row := 1190) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_13
  simp only [← c73431, ← c73432, k2, ← c73433] at e_1190_13
  have hr := e_1190_13
  simp only [e_1190_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10552)) := by
  have c73434 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73435 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73436 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73437 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73438 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73439 := (publicBatchWrapper89_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_14 := arithEq_of_rows h (row := 1190) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_14
  simp only [← c73434, ← c73435, ← c73436] at e_1190_14
  have e_1190_15 := arithEq_of_rows h (row := 1190) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_15
  simp only [← c73437, ← c73438, k2, ← c73439] at e_1190_15
  have hr := e_1190_15
  simp only [e_1190_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10553)) := by
  have c73440 := (publicBatchWrapper89_copies2295 a h).1
  have c73441 := (publicBatchWrapper89_copies2295 a h).2.1
  have c73442 := (publicBatchWrapper89_copies2295 a h).2.2.1
  have c73443 := (publicBatchWrapper89_copies2295 a h).2.2.2.1
  have c73444 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.1
  have c73445 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_16 := arithEq_of_rows h (row := 1190) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_16
  simp only [← c73440, ← c73441, ← c73442] at e_1190_16
  have e_1190_17 := arithEq_of_rows h (row := 1190) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_17
  simp only [← c73443, ← c73444, k2, ← c73445] at e_1190_17
  have hr := e_1190_17
  simp only [e_1190_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1190 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10554)) := by
  have c73446 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.1
  have c73447 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.1
  have c73448 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.1
  have c73449 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.1
  have c73450 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73451 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1190_18 := arithEq_of_rows h (row := 1190) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_18
  simp only [← c73446, ← c73447, ← c73448] at e_1190_18
  have e_1190_19 := arithEq_of_rows h (row := 1190) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_19
  simp only [← c73449, ← c73450, k2, ← c73451] at e_1190_19
  have hr := e_1190_19
  simp only [e_1190_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10555)) := by
  have c73452 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73453 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73454 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73455 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73456 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73457 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_0 := arithEq_of_rows h (row := 1191) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_0
  simp only [← c73452, ← c73453, ← c73454] at e_1191_0
  have e_1191_1 := arithEq_of_rows h (row := 1191) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_1
  simp only [← c73455, ← c73456, k2, ← c73457] at e_1191_1
  have hr := e_1191_1
  simp only [e_1191_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10556)) := by
  have c73458 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73459 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73460 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73461 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73462 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73463 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_2 := arithEq_of_rows h (row := 1191) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_2
  simp only [← c73458, ← c73459, ← c73460] at e_1191_2
  have e_1191_3 := arithEq_of_rows h (row := 1191) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_3
  simp only [← c73461, ← c73462, k2, ← c73463] at e_1191_3
  have hr := e_1191_3
  simp only [e_1191_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10557)) := by
  have c73464 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73465 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73466 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73467 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73468 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73469 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_4 := arithEq_of_rows h (row := 1191) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_4
  simp only [← c73464, ← c73465, ← c73466] at e_1191_4
  have e_1191_5 := arithEq_of_rows h (row := 1191) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_5
  simp only [← c73467, ← c73468, k2, ← c73469] at e_1191_5
  have hr := e_1191_5
  simp only [e_1191_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10558)) := by
  have c73470 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73471 := (publicBatchWrapper89_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73472 := (publicBatchWrapper89_copies2296 a h).1
  have c73473 := (publicBatchWrapper89_copies2296 a h).2.1
  have c73474 := (publicBatchWrapper89_copies2296 a h).2.2.1
  have c73475 := (publicBatchWrapper89_copies2296 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_6 := arithEq_of_rows h (row := 1191) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_6
  simp only [← c73470, ← c73471, ← c73472] at e_1191_6
  have e_1191_7 := arithEq_of_rows h (row := 1191) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_7
  simp only [← c73473, ← c73474, k2, ← c73475] at e_1191_7
  have hr := e_1191_7
  simp only [e_1191_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10559)) := by
  have c73476 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.1
  have c73477 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.1
  have c73478 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.1
  have c73479 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.1
  have c73480 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.1
  have c73481 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_8 := arithEq_of_rows h (row := 1191) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_8
  simp only [← c73476, ← c73477, ← c73478] at e_1191_8
  have e_1191_9 := arithEq_of_rows h (row := 1191) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_9
  simp only [← c73479, ← c73480, k2, ← c73481] at e_1191_9
  have hr := e_1191_9
  simp only [e_1191_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10560)) := by
  have c73482 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73483 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73484 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73485 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73486 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73487 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_10 := arithEq_of_rows h (row := 1191) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_10
  simp only [← c73482, ← c73483, ← c73484] at e_1191_10
  have e_1191_11 := arithEq_of_rows h (row := 1191) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_11
  simp only [← c73485, ← c73486, k2, ← c73487] at e_1191_11
  have hr := e_1191_11
  simp only [e_1191_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10561)) := by
  have c73488 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73489 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73490 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73491 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73492 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73493 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_12 := arithEq_of_rows h (row := 1191) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_12
  simp only [← c73488, ← c73489, ← c73490] at e_1191_12
  have e_1191_13 := arithEq_of_rows h (row := 1191) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_13
  simp only [← c73491, ← c73492, k2, ← c73493] at e_1191_13
  have hr := e_1191_13
  simp only [e_1191_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10562)) := by
  have c73494 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73495 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73496 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73497 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73498 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73499 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_14 := arithEq_of_rows h (row := 1191) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_14
  simp only [← c73494, ← c73495, ← c73496] at e_1191_14
  have e_1191_15 := arithEq_of_rows h (row := 1191) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_15
  simp only [← c73497, ← c73498, k2, ← c73499] at e_1191_15
  have hr := e_1191_15
  simp only [e_1191_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10563)) := by
  have c73500 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73501 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73502 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73503 := (publicBatchWrapper89_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73504 := (publicBatchWrapper89_copies2297 a h).1
  have c73505 := (publicBatchWrapper89_copies2297 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_16 := arithEq_of_rows h (row := 1191) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_16
  simp only [← c73500, ← c73501, ← c73502] at e_1191_16
  have e_1191_17 := arithEq_of_rows h (row := 1191) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_17
  simp only [← c73503, ← c73504, k2, ← c73505] at e_1191_17
  have hr := e_1191_17
  simp only [e_1191_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1191 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10564)) := by
  have c73506 := (publicBatchWrapper89_copies2297 a h).2.2.1
  have c73507 := (publicBatchWrapper89_copies2297 a h).2.2.2.1
  have c73508 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.1
  have c73509 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.1
  have c73510 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.1
  have c73511 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1191_18 := arithEq_of_rows h (row := 1191) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_18
  simp only [← c73506, ← c73507, ← c73508] at e_1191_18
  have e_1191_19 := arithEq_of_rows h (row := 1191) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_19
  simp only [← c73509, ← c73510, k2, ← c73511] at e_1191_19
  have hr := e_1191_19
  simp only [e_1191_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10565)) := by
  have c73512 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.1
  have c73513 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.1
  have c73514 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73515 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73516 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73517 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_0 := arithEq_of_rows h (row := 1192) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_0
  simp only [← c73512, ← c73513, ← c73514] at e_1192_0
  have e_1192_1 := arithEq_of_rows h (row := 1192) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_1
  simp only [← c73515, ← c73516, k2, ← c73517] at e_1192_1
  have hr := e_1192_1
  simp only [e_1192_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10566)) := by
  have c73518 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73519 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73520 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73521 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73522 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73523 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_2 := arithEq_of_rows h (row := 1192) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_2
  simp only [← c73518, ← c73519, ← c73520] at e_1192_2
  have e_1192_3 := arithEq_of_rows h (row := 1192) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_3
  simp only [← c73521, ← c73522, k2, ← c73523] at e_1192_3
  have hr := e_1192_3
  simp only [e_1192_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10567)) := by
  have c73524 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73525 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73526 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73527 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73528 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73529 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_4 := arithEq_of_rows h (row := 1192) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_4
  simp only [← c73524, ← c73525, ← c73526] at e_1192_4
  have e_1192_5 := arithEq_of_rows h (row := 1192) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_5
  simp only [← c73527, ← c73528, k2, ← c73529] at e_1192_5
  have hr := e_1192_5
  simp only [e_1192_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10568)) := by
  have c73530 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73531 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73532 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73533 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73534 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73535 := (publicBatchWrapper89_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_6 := arithEq_of_rows h (row := 1192) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_6
  simp only [← c73530, ← c73531, ← c73532] at e_1192_6
  have e_1192_7 := arithEq_of_rows h (row := 1192) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_7
  simp only [← c73533, ← c73534, k2, ← c73535] at e_1192_7
  have hr := e_1192_7
  simp only [e_1192_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10569)) := by
  have c73536 := (publicBatchWrapper89_copies2298 a h).1
  have c73537 := (publicBatchWrapper89_copies2298 a h).2.1
  have c73538 := (publicBatchWrapper89_copies2298 a h).2.2.1
  have c73539 := (publicBatchWrapper89_copies2298 a h).2.2.2.1
  have c73540 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.1
  have c73541 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_8 := arithEq_of_rows h (row := 1192) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_8
  simp only [← c73536, ← c73537, ← c73538] at e_1192_8
  have e_1192_9 := arithEq_of_rows h (row := 1192) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_9
  simp only [← c73539, ← c73540, k2, ← c73541] at e_1192_9
  have hr := e_1192_9
  simp only [e_1192_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10610)) := by
  have c73542 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.1
  have c73543 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.1
  have c73544 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.1
  have c73545 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.1
  have c73546 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73547 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_10 := arithEq_of_rows h (row := 1192) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_10
  simp only [← c73542, ← c73543, ← c73544] at e_1192_10
  have e_1192_11 := arithEq_of_rows h (row := 1192) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_11
  simp only [← c73545, ← c73546, k2, ← c73547] at e_1192_11
  have hr := e_1192_11
  simp only [e_1192_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10611)) := by
  have c73548 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73549 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73550 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73551 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73552 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73553 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_12 := arithEq_of_rows h (row := 1192) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_12
  simp only [← c73548, ← c73549, ← c73550] at e_1192_12
  have e_1192_13 := arithEq_of_rows h (row := 1192) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_13
  simp only [← c73551, ← c73552, k2, ← c73553] at e_1192_13
  have hr := e_1192_13
  simp only [e_1192_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10612)) := by
  have c73554 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73555 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73556 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73557 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73558 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73559 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_14 := arithEq_of_rows h (row := 1192) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_14
  simp only [← c73554, ← c73555, ← c73556] at e_1192_14
  have e_1192_15 := arithEq_of_rows h (row := 1192) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_15
  simp only [← c73557, ← c73558, k2, ← c73559] at e_1192_15
  have hr := e_1192_15
  simp only [e_1192_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10613)) := by
  have c73560 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73561 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73562 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73563 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73564 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73565 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_16 := arithEq_of_rows h (row := 1192) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_16
  simp only [← c73560, ← c73561, ← c73562] at e_1192_16
  have e_1192_17 := arithEq_of_rows h (row := 1192) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_17
  simp only [← c73563, ← c73564, k2, ← c73565] at e_1192_17
  have hr := e_1192_17
  simp only [e_1192_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1192 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10614)) := by
  have c73566 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73567 := (publicBatchWrapper89_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73568 := (publicBatchWrapper89_copies2299 a h).1
  have c73569 := (publicBatchWrapper89_copies2299 a h).2.1
  have c73570 := (publicBatchWrapper89_copies2299 a h).2.2.1
  have c73571 := (publicBatchWrapper89_copies2299 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1192_18 := arithEq_of_rows h (row := 1192) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_18
  simp only [← c73566, ← c73567, ← c73568] at e_1192_18
  have e_1192_19 := arithEq_of_rows h (row := 1192) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_19
  simp only [← c73569, ← c73570, k2, ← c73571] at e_1192_19
  have hr := e_1192_19
  simp only [e_1192_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10615)) := by
  have c73572 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.1
  have c73573 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.1
  have c73574 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.1
  have c73575 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.1
  have c73576 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.1
  have c73577 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_0 := arithEq_of_rows h (row := 1193) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_0
  simp only [← c73572, ← c73573, ← c73574] at e_1193_0
  have e_1193_1 := arithEq_of_rows h (row := 1193) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_1
  simp only [← c73575, ← c73576, k2, ← c73577] at e_1193_1
  have hr := e_1193_1
  simp only [e_1193_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10616)) := by
  have c73578 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73579 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73580 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73581 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73582 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73583 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_2 := arithEq_of_rows h (row := 1193) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_2
  simp only [← c73578, ← c73579, ← c73580] at e_1193_2
  have e_1193_3 := arithEq_of_rows h (row := 1193) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_3
  simp only [← c73581, ← c73582, k2, ← c73583] at e_1193_3
  have hr := e_1193_3
  simp only [e_1193_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10617)) := by
  have c73584 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73585 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73586 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73587 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73588 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73589 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_4 := arithEq_of_rows h (row := 1193) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_4
  simp only [← c73584, ← c73585, ← c73586] at e_1193_4
  have e_1193_5 := arithEq_of_rows h (row := 1193) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_5
  simp only [← c73587, ← c73588, k2, ← c73589] at e_1193_5
  have hr := e_1193_5
  simp only [e_1193_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10618)) := by
  have c73590 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73591 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73592 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73593 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73594 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73595 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_6 := arithEq_of_rows h (row := 1193) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_6
  simp only [← c73590, ← c73591, ← c73592] at e_1193_6
  have e_1193_7 := arithEq_of_rows h (row := 1193) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_7
  simp only [← c73593, ← c73594, k2, ← c73595] at e_1193_7
  have hr := e_1193_7
  simp only [e_1193_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10619)) := by
  have c73596 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73597 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73598 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73599 := (publicBatchWrapper89_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73600 := (publicBatchWrapper89_copies2300 a h).1
  have c73601 := (publicBatchWrapper89_copies2300 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_8 := arithEq_of_rows h (row := 1193) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_8
  simp only [← c73596, ← c73597, ← c73598] at e_1193_8
  have e_1193_9 := arithEq_of_rows h (row := 1193) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_9
  simp only [← c73599, ← c73600, k2, ← c73601] at e_1193_9
  have hr := e_1193_9
  simp only [e_1193_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10620)) := by
  have c73602 := (publicBatchWrapper89_copies2300 a h).2.2.1
  have c73603 := (publicBatchWrapper89_copies2300 a h).2.2.2.1
  have c73604 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.1
  have c73605 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.1
  have c73606 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.1
  have c73607 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_10 := arithEq_of_rows h (row := 1193) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_10
  simp only [← c73602, ← c73603, ← c73604] at e_1193_10
  have e_1193_11 := arithEq_of_rows h (row := 1193) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_11
  simp only [← c73605, ← c73606, k2, ← c73607] at e_1193_11
  have hr := e_1193_11
  simp only [e_1193_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10621)) := by
  have c73608 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.1
  have c73609 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.1
  have c73610 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73611 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73612 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73613 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_12 := arithEq_of_rows h (row := 1193) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_12
  simp only [← c73608, ← c73609, ← c73610] at e_1193_12
  have e_1193_13 := arithEq_of_rows h (row := 1193) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_13
  simp only [← c73611, ← c73612, k2, ← c73613] at e_1193_13
  have hr := e_1193_13
  simp only [e_1193_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10622)) := by
  have c73614 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73615 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73616 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73617 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73618 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73619 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_14 := arithEq_of_rows h (row := 1193) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_14
  simp only [← c73614, ← c73615, ← c73616] at e_1193_14
  have e_1193_15 := arithEq_of_rows h (row := 1193) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_15
  simp only [← c73617, ← c73618, k2, ← c73619] at e_1193_15
  have hr := e_1193_15
  simp only [e_1193_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10623)) := by
  have c73620 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73621 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73622 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73623 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73624 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73625 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_16 := arithEq_of_rows h (row := 1193) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_16
  simp only [← c73620, ← c73621, ← c73622] at e_1193_16
  have e_1193_17 := arithEq_of_rows h (row := 1193) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_17
  simp only [← c73623, ← c73624, k2, ← c73625] at e_1193_17
  have hr := e_1193_17
  simp only [e_1193_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1193 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10624)) := by
  have c73626 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73627 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73628 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73629 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73630 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73631 := (publicBatchWrapper89_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1193_18 := arithEq_of_rows h (row := 1193) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_18
  simp only [← c73626, ← c73627, ← c73628] at e_1193_18
  have e_1193_19 := arithEq_of_rows h (row := 1193) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_19
  simp only [← c73629, ← c73630, k2, ← c73631] at e_1193_19
  have hr := e_1193_19
  simp only [e_1193_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10625)) := by
  have c73632 := (publicBatchWrapper89_copies2301 a h).1
  have c73633 := (publicBatchWrapper89_copies2301 a h).2.1
  have c73634 := (publicBatchWrapper89_copies2301 a h).2.2.1
  have c73635 := (publicBatchWrapper89_copies2301 a h).2.2.2.1
  have c73636 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.1
  have c73637 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_0 := arithEq_of_rows h (row := 1194) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_0
  simp only [← c73632, ← c73633, ← c73634] at e_1194_0
  have e_1194_1 := arithEq_of_rows h (row := 1194) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_1
  simp only [← c73635, ← c73636, k2, ← c73637] at e_1194_1
  have hr := e_1194_1
  simp only [e_1194_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10626)) := by
  have c73638 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.1
  have c73639 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.1
  have c73640 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.1
  have c73641 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.1
  have c73642 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73643 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_2 := arithEq_of_rows h (row := 1194) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_2
  simp only [← c73638, ← c73639, ← c73640] at e_1194_2
  have e_1194_3 := arithEq_of_rows h (row := 1194) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_3
  simp only [← c73641, ← c73642, k2, ← c73643] at e_1194_3
  have hr := e_1194_3
  simp only [e_1194_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10627)) := by
  have c73644 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73645 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73646 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73647 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73648 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73649 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_4 := arithEq_of_rows h (row := 1194) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_4
  simp only [← c73644, ← c73645, ← c73646] at e_1194_4
  have e_1194_5 := arithEq_of_rows h (row := 1194) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_5
  simp only [← c73647, ← c73648, k2, ← c73649] at e_1194_5
  have hr := e_1194_5
  simp only [e_1194_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f11524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1194 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10628)) := by
  have c73650 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73651 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73652 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73653 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73654 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73655 := (publicBatchWrapper89_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1194_6 := arithEq_of_rows h (row := 1194) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_6
  simp only [← c73650, ← c73651, ← c73652] at e_1194_6
  have e_1194_7 := arithEq_of_rows h (row := 1194) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_7
  simp only [← c73653, ← c73654, k2, ← c73655] at e_1194_7
  have hr := e_1194_7
  simp only [e_1194_6] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
