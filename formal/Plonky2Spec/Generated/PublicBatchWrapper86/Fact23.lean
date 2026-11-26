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

theorem publicBatchWrapper86_f17780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6978)) := by
  have c111033 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111034 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111035 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111036 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111037 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111038 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_16 := arithEq_of_rows h (row := 1818) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_16
  simp only [← c111033, ← c111034, ← c111035] at e_1818_16
  have e_1818_17 := arithEq_of_rows h (row := 1818) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_17
  simp only [← c111036, ← c111037, k2, ← c111038] at e_1818_17
  have hr := e_1818_17
  simp only [e_1818_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6979)) := by
  have c111039 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111040 := (publicBatchWrapper86_copies3470 a h).1
  have c111041 := (publicBatchWrapper86_copies3470 a h).2.1
  have c111042 := (publicBatchWrapper86_copies3470 a h).2.2.1
  have c111043 := (publicBatchWrapper86_copies3470 a h).2.2.2.1
  have c111044 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_18 := arithEq_of_rows h (row := 1818) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_18
  simp only [← c111039, ← c111040, ← c111041] at e_1818_18
  have e_1818_19 := arithEq_of_rows h (row := 1818) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_19
  simp only [← c111042, ← c111043, k2, ← c111044] at e_1818_19
  have hr := e_1818_19
  simp only [e_1818_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6980)) := by
  have c111045 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.1
  have c111046 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.1
  have c111047 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.1
  have c111048 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.1
  have c111049 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.1
  have c111050 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_0 := arithEq_of_rows h (row := 1819) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_0
  simp only [← c111045, ← c111046, ← c111047] at e_1819_0
  have e_1819_1 := arithEq_of_rows h (row := 1819) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_1
  simp only [← c111048, ← c111049, k2, ← c111050] at e_1819_1
  have hr := e_1819_1
  simp only [e_1819_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6981)) := by
  have c111051 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111052 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111053 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111054 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111055 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111056 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_2 := arithEq_of_rows h (row := 1819) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_2
  simp only [← c111051, ← c111052, ← c111053] at e_1819_2
  have e_1819_3 := arithEq_of_rows h (row := 1819) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_3
  simp only [← c111054, ← c111055, k2, ← c111056] at e_1819_3
  have hr := e_1819_3
  simp only [e_1819_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6982)) := by
  have c111057 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111058 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111059 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111060 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111061 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111062 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_4 := arithEq_of_rows h (row := 1819) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_4
  simp only [← c111057, ← c111058, ← c111059] at e_1819_4
  have e_1819_5 := arithEq_of_rows h (row := 1819) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_5
  simp only [← c111060, ← c111061, k2, ← c111062] at e_1819_5
  have hr := e_1819_5
  simp only [e_1819_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6983)) := by
  have c111063 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111064 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111065 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111066 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111067 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111068 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_6 := arithEq_of_rows h (row := 1819) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_6
  simp only [← c111063, ← c111064, ← c111065] at e_1819_6
  have e_1819_7 := arithEq_of_rows h (row := 1819) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_7
  simp only [← c111066, ← c111067, k2, ← c111068] at e_1819_7
  have hr := e_1819_7
  simp only [e_1819_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6984)) := by
  have c111069 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111070 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111071 := (publicBatchWrapper86_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111072 := (publicBatchWrapper86_copies3471 a h).1
  have c111073 := (publicBatchWrapper86_copies3471 a h).2.1
  have c111074 := (publicBatchWrapper86_copies3471 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_8 := arithEq_of_rows h (row := 1819) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_8
  simp only [← c111069, ← c111070, ← c111071] at e_1819_8
  have e_1819_9 := arithEq_of_rows h (row := 1819) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_9
  simp only [← c111072, ← c111073, k2, ← c111074] at e_1819_9
  have hr := e_1819_9
  simp only [e_1819_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6985)) := by
  have c111075 := (publicBatchWrapper86_copies3471 a h).2.2.2.1
  have c111076 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.1
  have c111077 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.1
  have c111078 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.1
  have c111079 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.1
  have c111080 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_10 := arithEq_of_rows h (row := 1819) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_10
  simp only [← c111075, ← c111076, ← c111077] at e_1819_10
  have e_1819_11 := arithEq_of_rows h (row := 1819) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_11
  simp only [← c111078, ← c111079, k2, ← c111080] at e_1819_11
  have hr := e_1819_11
  simp only [e_1819_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6986)) := by
  have c111081 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.1
  have c111082 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111083 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111084 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111085 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111086 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_12 := arithEq_of_rows h (row := 1819) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_12
  simp only [← c111081, ← c111082, ← c111083] at e_1819_12
  have e_1819_13 := arithEq_of_rows h (row := 1819) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_13
  simp only [← c111084, ← c111085, k2, ← c111086] at e_1819_13
  have hr := e_1819_13
  simp only [e_1819_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6987)) := by
  have c111087 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111088 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111089 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111090 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111091 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111092 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_14 := arithEq_of_rows h (row := 1819) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_14
  simp only [← c111087, ← c111088, ← c111089] at e_1819_14
  have e_1819_15 := arithEq_of_rows h (row := 1819) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_15
  simp only [← c111090, ← c111091, k2, ← c111092] at e_1819_15
  have hr := e_1819_15
  simp only [e_1819_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6988)) := by
  have c111093 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111094 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111095 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111096 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111097 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111098 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_16 := arithEq_of_rows h (row := 1819) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_16
  simp only [← c111093, ← c111094, ← c111095] at e_1819_16
  have e_1819_17 := arithEq_of_rows h (row := 1819) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_17
  simp only [← c111096, ← c111097, k2, ← c111098] at e_1819_17
  have hr := e_1819_17
  simp only [e_1819_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1819 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6989)) := by
  have c111099 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111100 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111101 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111102 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111103 := (publicBatchWrapper86_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111104 := (publicBatchWrapper86_copies3472 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1819_18 := arithEq_of_rows h (row := 1819) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_18
  simp only [← c111099, ← c111100, ← c111101] at e_1819_18
  have e_1819_19 := arithEq_of_rows h (row := 1819) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_19
  simp only [← c111102, ← c111103, k2, ← c111104] at e_1819_19
  have hr := e_1819_19
  simp only [e_1819_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6990)) := by
  have c111105 := (publicBatchWrapper86_copies3472 a h).2.1
  have c111106 := (publicBatchWrapper86_copies3472 a h).2.2.1
  have c111107 := (publicBatchWrapper86_copies3472 a h).2.2.2.1
  have c111108 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.1
  have c111109 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.1
  have c111110 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_0 := arithEq_of_rows h (row := 1820) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_0
  simp only [← c111105, ← c111106, ← c111107] at e_1820_0
  have e_1820_1 := arithEq_of_rows h (row := 1820) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_1
  simp only [← c111108, ← c111109, k2, ← c111110] at e_1820_1
  have hr := e_1820_1
  simp only [e_1820_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6991)) := by
  have c111111 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.1
  have c111112 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.1
  have c111113 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.1
  have c111114 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111115 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111116 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_2 := arithEq_of_rows h (row := 1820) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_2
  simp only [← c111111, ← c111112, ← c111113] at e_1820_2
  have e_1820_3 := arithEq_of_rows h (row := 1820) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_3
  simp only [← c111114, ← c111115, k2, ← c111116] at e_1820_3
  have hr := e_1820_3
  simp only [e_1820_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6992)) := by
  have c111117 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111118 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111119 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111120 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111121 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111122 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_4 := arithEq_of_rows h (row := 1820) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_4
  simp only [← c111117, ← c111118, ← c111119] at e_1820_4
  have e_1820_5 := arithEq_of_rows h (row := 1820) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_5
  simp only [← c111120, ← c111121, k2, ← c111122] at e_1820_5
  have hr := e_1820_5
  simp only [e_1820_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6993)) := by
  have c111123 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111124 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111125 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111126 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111127 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111128 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_6 := arithEq_of_rows h (row := 1820) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_6
  simp only [← c111123, ← c111124, ← c111125] at e_1820_6
  have e_1820_7 := arithEq_of_rows h (row := 1820) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_7
  simp only [← c111126, ← c111127, k2, ← c111128] at e_1820_7
  have hr := e_1820_7
  simp only [e_1820_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6994)) := by
  have c111129 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111130 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111131 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111132 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111133 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111134 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_8 := arithEq_of_rows h (row := 1820) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_8
  simp only [← c111129, ← c111130, ← c111131] at e_1820_8
  have e_1820_9 := arithEq_of_rows h (row := 1820) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_9
  simp only [← c111132, ← c111133, k2, ← c111134] at e_1820_9
  have hr := e_1820_9
  simp only [e_1820_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6995)) := by
  have c111135 := (publicBatchWrapper86_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111136 := (publicBatchWrapper86_copies3473 a h).1
  have c111137 := (publicBatchWrapper86_copies3473 a h).2.1
  have c111138 := (publicBatchWrapper86_copies3473 a h).2.2.1
  have c111139 := (publicBatchWrapper86_copies3473 a h).2.2.2.1
  have c111140 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_10 := arithEq_of_rows h (row := 1820) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_10
  simp only [← c111135, ← c111136, ← c111137] at e_1820_10
  have e_1820_11 := arithEq_of_rows h (row := 1820) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_11
  simp only [← c111138, ← c111139, k2, ← c111140] at e_1820_11
  have hr := e_1820_11
  simp only [e_1820_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6996)) := by
  have c111141 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.1
  have c111142 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.1
  have c111143 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.1
  have c111144 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.1
  have c111145 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.1
  have c111146 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_12 := arithEq_of_rows h (row := 1820) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_12
  simp only [← c111141, ← c111142, ← c111143] at e_1820_12
  have e_1820_13 := arithEq_of_rows h (row := 1820) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_13
  simp only [← c111144, ← c111145, k2, ← c111146] at e_1820_13
  have hr := e_1820_13
  simp only [e_1820_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6997)) := by
  have c111147 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111148 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111149 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111150 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111151 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111152 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_14 := arithEq_of_rows h (row := 1820) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_14
  simp only [← c111147, ← c111148, ← c111149] at e_1820_14
  have e_1820_15 := arithEq_of_rows h (row := 1820) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_15
  simp only [← c111150, ← c111151, k2, ← c111152] at e_1820_15
  have hr := e_1820_15
  simp only [e_1820_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6998)) := by
  have c111153 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111154 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111155 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111156 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111157 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111158 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_16 := arithEq_of_rows h (row := 1820) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_16
  simp only [← c111153, ← c111154, ← c111155] at e_1820_16
  have e_1820_17 := arithEq_of_rows h (row := 1820) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_17
  simp only [← c111156, ← c111157, k2, ← c111158] at e_1820_17
  have hr := e_1820_17
  simp only [e_1820_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1820 79) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6999)) := by
  have c111159 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111160 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111161 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111162 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111163 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111164 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1820_18 := arithEq_of_rows h (row := 1820) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_18
  simp only [← c111159, ← c111160, ← c111161] at e_1820_18
  have e_1820_19 := arithEq_of_rows h (row := 1820) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_19
  simp only [← c111162, ← c111163, k2, ← c111164] at e_1820_19
  have hr := e_1820_19
  simp only [e_1820_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 7000)) := by
  have c111165 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111166 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111167 := (publicBatchWrapper86_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111168 := (publicBatchWrapper86_copies3474 a h).1
  have c111169 := (publicBatchWrapper86_copies3474 a h).2.1
  have c111170 := (publicBatchWrapper86_copies3474 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_0 := arithEq_of_rows h (row := 1821) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_0
  simp only [← c111165, ← c111166, ← c111167] at e_1821_0
  have e_1821_1 := arithEq_of_rows h (row := 1821) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_1
  simp only [← c111168, ← c111169, k2, ← c111170] at e_1821_1
  have hr := e_1821_1
  simp only [e_1821_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 7001)) := by
  have c111171 := (publicBatchWrapper86_copies3474 a h).2.2.2.1
  have c111172 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.1
  have c111173 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.1
  have c111174 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.1
  have c111175 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.1
  have c111176 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_2 := arithEq_of_rows h (row := 1821) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_2
  simp only [← c111171, ← c111172, ← c111173] at e_1821_2
  have e_1821_3 := arithEq_of_rows h (row := 1821) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_3
  simp only [← c111174, ← c111175, k2, ← c111176] at e_1821_3
  have hr := e_1821_3
  simp only [e_1821_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7170)) := by
  have c111177 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.1
  have c111178 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111179 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111180 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111181 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111182 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_4 := arithEq_of_rows h (row := 1821) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_4
  simp only [← c111177, ← c111178, ← c111179] at e_1821_4
  have e_1821_5 := arithEq_of_rows h (row := 1821) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_5
  simp only [← c111180, ← c111181, k2, ← c111182] at e_1821_5
  have hr := e_1821_5
  simp only [e_1821_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7171)) := by
  have c111183 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111184 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111185 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111186 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111187 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111188 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_6 := arithEq_of_rows h (row := 1821) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_6
  simp only [← c111183, ← c111184, ← c111185] at e_1821_6
  have e_1821_7 := arithEq_of_rows h (row := 1821) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_7
  simp only [← c111186, ← c111187, k2, ← c111188] at e_1821_7
  have hr := e_1821_7
  simp only [e_1821_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7172)) := by
  have c111189 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111190 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111191 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111192 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111193 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111194 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_8 := arithEq_of_rows h (row := 1821) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_8
  simp only [← c111189, ← c111190, ← c111191] at e_1821_8
  have e_1821_9 := arithEq_of_rows h (row := 1821) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_9
  simp only [← c111192, ← c111193, k2, ← c111194] at e_1821_9
  have hr := e_1821_9
  simp only [e_1821_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7173)) := by
  have c111195 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111196 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111197 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111198 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111199 := (publicBatchWrapper86_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111200 := (publicBatchWrapper86_copies3475 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_10 := arithEq_of_rows h (row := 1821) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_10
  simp only [← c111195, ← c111196, ← c111197] at e_1821_10
  have e_1821_11 := arithEq_of_rows h (row := 1821) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_11
  simp only [← c111198, ← c111199, k2, ← c111200] at e_1821_11
  have hr := e_1821_11
  simp only [e_1821_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7174)) := by
  have c111201 := (publicBatchWrapper86_copies3475 a h).2.1
  have c111202 := (publicBatchWrapper86_copies3475 a h).2.2.1
  have c111203 := (publicBatchWrapper86_copies3475 a h).2.2.2.1
  have c111204 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.1
  have c111205 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.1
  have c111206 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_12 := arithEq_of_rows h (row := 1821) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_12
  simp only [← c111201, ← c111202, ← c111203] at e_1821_12
  have e_1821_13 := arithEq_of_rows h (row := 1821) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_13
  simp only [← c111204, ← c111205, k2, ← c111206] at e_1821_13
  have hr := e_1821_13
  simp only [e_1821_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7175)) := by
  have c111207 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.1
  have c111208 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.1
  have c111209 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.1
  have c111210 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111211 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111212 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_14 := arithEq_of_rows h (row := 1821) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_14
  simp only [← c111207, ← c111208, ← c111209] at e_1821_14
  have e_1821_15 := arithEq_of_rows h (row := 1821) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_15
  simp only [← c111210, ← c111211, k2, ← c111212] at e_1821_15
  have hr := e_1821_15
  simp only [e_1821_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7176)) := by
  have c111213 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111214 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111215 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111216 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111217 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111218 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_16 := arithEq_of_rows h (row := 1821) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_16
  simp only [← c111213, ← c111214, ← c111215] at e_1821_16
  have e_1821_17 := arithEq_of_rows h (row := 1821) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_17
  simp only [← c111216, ← c111217, k2, ← c111218] at e_1821_17
  have hr := e_1821_17
  simp only [e_1821_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1821 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7177)) := by
  have c111219 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111220 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111221 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111222 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111223 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111224 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1821_18 := arithEq_of_rows h (row := 1821) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_18
  simp only [← c111219, ← c111220, ← c111221] at e_1821_18
  have e_1821_19 := arithEq_of_rows h (row := 1821) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_19
  simp only [← c111222, ← c111223, k2, ← c111224] at e_1821_19
  have hr := e_1821_19
  simp only [e_1821_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7178)) := by
  have c111225 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111226 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111227 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111228 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111229 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111230 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_0 := arithEq_of_rows h (row := 1822) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_0
  simp only [← c111225, ← c111226, ← c111227] at e_1822_0
  have e_1822_1 := arithEq_of_rows h (row := 1822) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_1
  simp only [← c111228, ← c111229, k2, ← c111230] at e_1822_1
  have hr := e_1822_1
  simp only [e_1822_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7179)) := by
  have c111231 := (publicBatchWrapper86_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111232 := (publicBatchWrapper86_copies3476 a h).1
  have c111233 := (publicBatchWrapper86_copies3476 a h).2.1
  have c111234 := (publicBatchWrapper86_copies3476 a h).2.2.1
  have c111235 := (publicBatchWrapper86_copies3476 a h).2.2.2.1
  have c111236 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_2 := arithEq_of_rows h (row := 1822) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_2
  simp only [← c111231, ← c111232, ← c111233] at e_1822_2
  have e_1822_3 := arithEq_of_rows h (row := 1822) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_3
  simp only [← c111234, ← c111235, k2, ← c111236] at e_1822_3
  have hr := e_1822_3
  simp only [e_1822_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7180)) := by
  have c111237 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.1
  have c111238 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.1
  have c111239 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.1
  have c111240 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.1
  have c111241 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.1
  have c111242 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_4 := arithEq_of_rows h (row := 1822) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_4
  simp only [← c111237, ← c111238, ← c111239] at e_1822_4
  have e_1822_5 := arithEq_of_rows h (row := 1822) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_5
  simp only [← c111240, ← c111241, k2, ← c111242] at e_1822_5
  have hr := e_1822_5
  simp only [e_1822_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7181)) := by
  have c111243 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111244 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111245 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111246 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111247 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111248 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_6 := arithEq_of_rows h (row := 1822) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_6
  simp only [← c111243, ← c111244, ← c111245] at e_1822_6
  have e_1822_7 := arithEq_of_rows h (row := 1822) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_7
  simp only [← c111246, ← c111247, k2, ← c111248] at e_1822_7
  have hr := e_1822_7
  simp only [e_1822_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7182)) := by
  have c111249 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111250 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111251 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111252 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111253 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111254 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_8 := arithEq_of_rows h (row := 1822) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_8
  simp only [← c111249, ← c111250, ← c111251] at e_1822_8
  have e_1822_9 := arithEq_of_rows h (row := 1822) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_9
  simp only [← c111252, ← c111253, k2, ← c111254] at e_1822_9
  have hr := e_1822_9
  simp only [e_1822_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7183)) := by
  have c111255 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111256 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111257 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111258 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111259 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111260 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_10 := arithEq_of_rows h (row := 1822) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_10
  simp only [← c111255, ← c111256, ← c111257] at e_1822_10
  have e_1822_11 := arithEq_of_rows h (row := 1822) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_11
  simp only [← c111258, ← c111259, k2, ← c111260] at e_1822_11
  have hr := e_1822_11
  simp only [e_1822_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7184)) := by
  have c111261 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111262 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111263 := (publicBatchWrapper86_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111264 := (publicBatchWrapper86_copies3477 a h).1
  have c111265 := (publicBatchWrapper86_copies3477 a h).2.1
  have c111266 := (publicBatchWrapper86_copies3477 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_12 := arithEq_of_rows h (row := 1822) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_12
  simp only [← c111261, ← c111262, ← c111263] at e_1822_12
  have e_1822_13 := arithEq_of_rows h (row := 1822) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_13
  simp only [← c111264, ← c111265, k2, ← c111266] at e_1822_13
  have hr := e_1822_13
  simp only [e_1822_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7185)) := by
  have c111267 := (publicBatchWrapper86_copies3477 a h).2.2.2.1
  have c111268 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.1
  have c111269 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.1
  have c111270 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.1
  have c111271 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.1
  have c111272 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_14 := arithEq_of_rows h (row := 1822) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_14
  simp only [← c111267, ← c111268, ← c111269] at e_1822_14
  have e_1822_15 := arithEq_of_rows h (row := 1822) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_15
  simp only [← c111270, ← c111271, k2, ← c111272] at e_1822_15
  have hr := e_1822_15
  simp only [e_1822_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7186)) := by
  have c111273 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.1
  have c111274 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111275 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111276 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111277 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111278 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_16 := arithEq_of_rows h (row := 1822) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_16
  simp only [← c111273, ← c111274, ← c111275] at e_1822_16
  have e_1822_17 := arithEq_of_rows h (row := 1822) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_17
  simp only [← c111276, ← c111277, k2, ← c111278] at e_1822_17
  have hr := e_1822_17
  simp only [e_1822_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1822 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7187)) := by
  have c111279 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111280 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111281 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111282 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111283 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111284 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1822_18 := arithEq_of_rows h (row := 1822) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_18
  simp only [← c111279, ← c111280, ← c111281] at e_1822_18
  have e_1822_19 := arithEq_of_rows h (row := 1822) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_19
  simp only [← c111282, ← c111283, k2, ← c111284] at e_1822_19
  have hr := e_1822_19
  simp only [e_1822_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7188)) := by
  have c111285 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111286 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111287 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111288 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111289 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111290 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_0 := arithEq_of_rows h (row := 1823) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_0
  simp only [← c111285, ← c111286, ← c111287] at e_1823_0
  have e_1823_1 := arithEq_of_rows h (row := 1823) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_1
  simp only [← c111288, ← c111289, k2, ← c111290] at e_1823_1
  have hr := e_1823_1
  simp only [e_1823_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7189)) := by
  have c111291 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111292 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111293 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111294 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111295 := (publicBatchWrapper86_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111296 := (publicBatchWrapper86_copies3478 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_2 := arithEq_of_rows h (row := 1823) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_2
  simp only [← c111291, ← c111292, ← c111293] at e_1823_2
  have e_1823_3 := arithEq_of_rows h (row := 1823) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_3
  simp only [← c111294, ← c111295, k2, ← c111296] at e_1823_3
  have hr := e_1823_3
  simp only [e_1823_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7190)) := by
  have c111297 := (publicBatchWrapper86_copies3478 a h).2.1
  have c111298 := (publicBatchWrapper86_copies3478 a h).2.2.1
  have c111299 := (publicBatchWrapper86_copies3478 a h).2.2.2.1
  have c111300 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.1
  have c111301 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.1
  have c111302 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_4 := arithEq_of_rows h (row := 1823) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_4
  simp only [← c111297, ← c111298, ← c111299] at e_1823_4
  have e_1823_5 := arithEq_of_rows h (row := 1823) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_5
  simp only [← c111300, ← c111301, k2, ← c111302] at e_1823_5
  have hr := e_1823_5
  simp only [e_1823_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7191)) := by
  have c111303 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.1
  have c111304 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.1
  have c111305 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.1
  have c111306 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111307 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111308 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_6 := arithEq_of_rows h (row := 1823) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_6
  simp only [← c111303, ← c111304, ← c111305] at e_1823_6
  have e_1823_7 := arithEq_of_rows h (row := 1823) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_7
  simp only [← c111306, ← c111307, k2, ← c111308] at e_1823_7
  have hr := e_1823_7
  simp only [e_1823_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 39) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7192)) := by
  have c111309 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111310 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111311 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111312 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111313 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111314 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_8 := arithEq_of_rows h (row := 1823) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_8
  simp only [← c111309, ← c111310, ← c111311] at e_1823_8
  have e_1823_9 := arithEq_of_rows h (row := 1823) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_9
  simp only [← c111312, ← c111313, k2, ← c111314] at e_1823_9
  have hr := e_1823_9
  simp only [e_1823_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 47) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7193)) := by
  have c111315 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111316 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111317 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111318 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111319 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111320 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_10 := arithEq_of_rows h (row := 1823) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_10
  simp only [← c111315, ← c111316, ← c111317] at e_1823_10
  have e_1823_11 := arithEq_of_rows h (row := 1823) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_11
  simp only [← c111318, ← c111319, k2, ← c111320] at e_1823_11
  have hr := e_1823_11
  simp only [e_1823_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 55) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7194)) := by
  have c111321 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111322 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111323 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111324 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111325 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111326 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_12 := arithEq_of_rows h (row := 1823) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_12
  simp only [← c111321, ← c111322, ← c111323] at e_1823_12
  have e_1823_13 := arithEq_of_rows h (row := 1823) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_13
  simp only [← c111324, ← c111325, k2, ← c111326] at e_1823_13
  have hr := e_1823_13
  simp only [e_1823_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 63) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7195)) := by
  have c111327 := (publicBatchWrapper86_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111328 := (publicBatchWrapper86_copies3479 a h).1
  have c111329 := (publicBatchWrapper86_copies3479 a h).2.1
  have c111330 := (publicBatchWrapper86_copies3479 a h).2.2.1
  have c111331 := (publicBatchWrapper86_copies3479 a h).2.2.2.1
  have c111332 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_14 := arithEq_of_rows h (row := 1823) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_14
  simp only [← c111327, ← c111328, ← c111329] at e_1823_14
  have e_1823_15 := arithEq_of_rows h (row := 1823) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_15
  simp only [← c111330, ← c111331, k2, ← c111332] at e_1823_15
  have hr := e_1823_15
  simp only [e_1823_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 71) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7196)) := by
  have c111333 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.1
  have c111334 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.1
  have c111335 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.1
  have c111336 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.1
  have c111337 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.1
  have c111338 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_16 := arithEq_of_rows h (row := 1823) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_16
  simp only [← c111333, ← c111334, ← c111335] at e_1823_16
  have e_1823_17 := arithEq_of_rows h (row := 1823) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_17
  simp only [← c111336, ← c111337, k2, ← c111338] at e_1823_17
  have hr := e_1823_17
  simp only [e_1823_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1823 79) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7197)) := by
  have c111339 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111340 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111341 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111342 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111343 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111344 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1823_18 := arithEq_of_rows h (row := 1823) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_18
  simp only [← c111339, ← c111340, ← c111341] at e_1823_18
  have e_1823_19 := arithEq_of_rows h (row := 1823) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_19
  simp only [← c111342, ← c111343, k2, ← c111344] at e_1823_19
  have hr := e_1823_19
  simp only [e_1823_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 7) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7198)) := by
  have c111345 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111346 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111347 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111348 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111349 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111350 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_0 := arithEq_of_rows h (row := 1824) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_0
  simp only [← c111345, ← c111346, ← c111347] at e_1824_0
  have e_1824_1 := arithEq_of_rows h (row := 1824) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_1
  simp only [← c111348, ← c111349, k2, ← c111350] at e_1824_1
  have hr := e_1824_1
  simp only [e_1824_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 15) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7199)) := by
  have c111351 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111352 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111353 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111354 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111355 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111356 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_2 := arithEq_of_rows h (row := 1824) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_2
  simp only [← c111351, ← c111352, ← c111353] at e_1824_2
  have e_1824_3 := arithEq_of_rows h (row := 1824) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_3
  simp only [← c111354, ← c111355, k2, ← c111356] at e_1824_3
  have hr := e_1824_3
  simp only [e_1824_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 23) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7200)) := by
  have c111357 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111358 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111359 := (publicBatchWrapper86_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111360 := (publicBatchWrapper86_copies3480 a h).1
  have c111361 := (publicBatchWrapper86_copies3480 a h).2.1
  have c111362 := (publicBatchWrapper86_copies3480 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_4 := arithEq_of_rows h (row := 1824) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_4
  simp only [← c111357, ← c111358, ← c111359] at e_1824_4
  have e_1824_5 := arithEq_of_rows h (row := 1824) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_5
  simp only [← c111360, ← c111361, k2, ← c111362] at e_1824_5
  have hr := e_1824_5
  simp only [e_1824_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 31) = bselect (a (.wire 35 47)) (a (.virt 200)) (a (.virt 7201)) := by
  have c111363 := (publicBatchWrapper86_copies3480 a h).2.2.2.1
  have c111364 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.1
  have c111365 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.1
  have c111366 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.1
  have c111367 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.1
  have c111368 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_6 := arithEq_of_rows h (row := 1824) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_6
  simp only [← c111363, ← c111364, ← c111365] at e_1824_6
  have e_1824_7 := arithEq_of_rows h (row := 1824) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_7
  simp only [← c111366, ← c111367, k2, ← c111368] at e_1824_7
  have hr := e_1824_7
  simp only [e_1824_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7370)) := by
  have c111369 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.1
  have c111370 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111371 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111372 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111373 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111374 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_8 := arithEq_of_rows h (row := 1824) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_8
  simp only [← c111369, ← c111370, ← c111371] at e_1824_8
  have e_1824_9 := arithEq_of_rows h (row := 1824) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_9
  simp only [← c111372, ← c111373, k2, ← c111374] at e_1824_9
  have hr := e_1824_9
  simp only [e_1824_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7371)) := by
  have c111375 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111376 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111377 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111378 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111379 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111380 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_10 := arithEq_of_rows h (row := 1824) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_10
  simp only [← c111375, ← c111376, ← c111377] at e_1824_10
  have e_1824_11 := arithEq_of_rows h (row := 1824) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_11
  simp only [← c111378, ← c111379, k2, ← c111380] at e_1824_11
  have hr := e_1824_11
  simp only [e_1824_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7372)) := by
  have c111381 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111382 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111383 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111384 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111385 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111386 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_12 := arithEq_of_rows h (row := 1824) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_12
  simp only [← c111381, ← c111382, ← c111383] at e_1824_12
  have e_1824_13 := arithEq_of_rows h (row := 1824) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_13
  simp only [← c111384, ← c111385, k2, ← c111386] at e_1824_13
  have hr := e_1824_13
  simp only [e_1824_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7373)) := by
  have c111387 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111388 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111389 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111390 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111391 := (publicBatchWrapper86_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111392 := (publicBatchWrapper86_copies3481 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_14 := arithEq_of_rows h (row := 1824) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_14
  simp only [← c111387, ← c111388, ← c111389] at e_1824_14
  have e_1824_15 := arithEq_of_rows h (row := 1824) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_15
  simp only [← c111390, ← c111391, k2, ← c111392] at e_1824_15
  have hr := e_1824_15
  simp only [e_1824_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7374)) := by
  have c111393 := (publicBatchWrapper86_copies3481 a h).2.1
  have c111394 := (publicBatchWrapper86_copies3481 a h).2.2.1
  have c111395 := (publicBatchWrapper86_copies3481 a h).2.2.2.1
  have c111396 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.1
  have c111397 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.1
  have c111398 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_16 := arithEq_of_rows h (row := 1824) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_16
  simp only [← c111393, ← c111394, ← c111395] at e_1824_16
  have e_1824_17 := arithEq_of_rows h (row := 1824) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_17
  simp only [← c111396, ← c111397, k2, ← c111398] at e_1824_17
  have hr := e_1824_17
  simp only [e_1824_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1824 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7375)) := by
  have c111399 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.1
  have c111400 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.1
  have c111401 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.1
  have c111402 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111403 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111404 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1824_18 := arithEq_of_rows h (row := 1824) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_18
  simp only [← c111399, ← c111400, ← c111401] at e_1824_18
  have e_1824_19 := arithEq_of_rows h (row := 1824) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_19
  simp only [← c111402, ← c111403, k2, ← c111404] at e_1824_19
  have hr := e_1824_19
  simp only [e_1824_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7376)) := by
  have c111405 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111406 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111407 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111408 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111409 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111410 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_0 := arithEq_of_rows h (row := 1825) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_0
  simp only [← c111405, ← c111406, ← c111407] at e_1825_0
  have e_1825_1 := arithEq_of_rows h (row := 1825) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_1
  simp only [← c111408, ← c111409, k2, ← c111410] at e_1825_1
  have hr := e_1825_1
  simp only [e_1825_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7377)) := by
  have c111411 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111412 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111413 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111414 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111415 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111416 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_2 := arithEq_of_rows h (row := 1825) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_2
  simp only [← c111411, ← c111412, ← c111413] at e_1825_2
  have e_1825_3 := arithEq_of_rows h (row := 1825) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_3
  simp only [← c111414, ← c111415, k2, ← c111416] at e_1825_3
  have hr := e_1825_3
  simp only [e_1825_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7378)) := by
  have c111417 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111418 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111419 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111420 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111421 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111422 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_4 := arithEq_of_rows h (row := 1825) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_4
  simp only [← c111417, ← c111418, ← c111419] at e_1825_4
  have e_1825_5 := arithEq_of_rows h (row := 1825) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_5
  simp only [← c111420, ← c111421, k2, ← c111422] at e_1825_5
  have hr := e_1825_5
  simp only [e_1825_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7379)) := by
  have c111423 := (publicBatchWrapper86_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111424 := (publicBatchWrapper86_copies3482 a h).1
  have c111425 := (publicBatchWrapper86_copies3482 a h).2.1
  have c111426 := (publicBatchWrapper86_copies3482 a h).2.2.1
  have c111427 := (publicBatchWrapper86_copies3482 a h).2.2.2.1
  have c111428 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_6 := arithEq_of_rows h (row := 1825) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_6
  simp only [← c111423, ← c111424, ← c111425] at e_1825_6
  have e_1825_7 := arithEq_of_rows h (row := 1825) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_7
  simp only [← c111426, ← c111427, k2, ← c111428] at e_1825_7
  have hr := e_1825_7
  simp only [e_1825_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7380)) := by
  have c111429 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.1
  have c111430 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.1
  have c111431 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.1
  have c111432 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.1
  have c111433 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.1
  have c111434 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_8 := arithEq_of_rows h (row := 1825) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_8
  simp only [← c111429, ← c111430, ← c111431] at e_1825_8
  have e_1825_9 := arithEq_of_rows h (row := 1825) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_9
  simp only [← c111432, ← c111433, k2, ← c111434] at e_1825_9
  have hr := e_1825_9
  simp only [e_1825_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7381)) := by
  have c111435 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111436 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111437 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111438 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111439 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111440 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_10 := arithEq_of_rows h (row := 1825) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_10
  simp only [← c111435, ← c111436, ← c111437] at e_1825_10
  have e_1825_11 := arithEq_of_rows h (row := 1825) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_11
  simp only [← c111438, ← c111439, k2, ← c111440] at e_1825_11
  have hr := e_1825_11
  simp only [e_1825_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7382)) := by
  have c111441 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111442 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111443 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111444 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111445 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111446 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_12 := arithEq_of_rows h (row := 1825) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_12
  simp only [← c111441, ← c111442, ← c111443] at e_1825_12
  have e_1825_13 := arithEq_of_rows h (row := 1825) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_13
  simp only [← c111444, ← c111445, k2, ← c111446] at e_1825_13
  have hr := e_1825_13
  simp only [e_1825_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7383)) := by
  have c111447 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111448 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111449 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111450 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111451 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111452 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_14 := arithEq_of_rows h (row := 1825) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_14
  simp only [← c111447, ← c111448, ← c111449] at e_1825_14
  have e_1825_15 := arithEq_of_rows h (row := 1825) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_15
  simp only [← c111450, ← c111451, k2, ← c111452] at e_1825_15
  have hr := e_1825_15
  simp only [e_1825_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7384)) := by
  have c111453 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111454 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111455 := (publicBatchWrapper86_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111456 := (publicBatchWrapper86_copies3483 a h).1
  have c111457 := (publicBatchWrapper86_copies3483 a h).2.1
  have c111458 := (publicBatchWrapper86_copies3483 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_16 := arithEq_of_rows h (row := 1825) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_16
  simp only [← c111453, ← c111454, ← c111455] at e_1825_16
  have e_1825_17 := arithEq_of_rows h (row := 1825) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_17
  simp only [← c111456, ← c111457, k2, ← c111458] at e_1825_17
  have hr := e_1825_17
  simp only [e_1825_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1825 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7385)) := by
  have c111459 := (publicBatchWrapper86_copies3483 a h).2.2.2.1
  have c111460 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.1
  have c111461 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.1
  have c111462 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.1
  have c111463 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.1
  have c111464 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1825_18 := arithEq_of_rows h (row := 1825) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_18
  simp only [← c111459, ← c111460, ← c111461] at e_1825_18
  have e_1825_19 := arithEq_of_rows h (row := 1825) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_19
  simp only [← c111462, ← c111463, k2, ← c111464] at e_1825_19
  have hr := e_1825_19
  simp only [e_1825_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7386)) := by
  have c111465 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.1
  have c111466 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111467 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111468 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111469 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111470 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_0 := arithEq_of_rows h (row := 1826) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_0
  simp only [← c111465, ← c111466, ← c111467] at e_1826_0
  have e_1826_1 := arithEq_of_rows h (row := 1826) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_1
  simp only [← c111468, ← c111469, k2, ← c111470] at e_1826_1
  have hr := e_1826_1
  simp only [e_1826_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7387)) := by
  have c111471 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111472 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111473 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111474 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111475 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111476 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_2 := arithEq_of_rows h (row := 1826) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_2
  simp only [← c111471, ← c111472, ← c111473] at e_1826_2
  have e_1826_3 := arithEq_of_rows h (row := 1826) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_3
  simp only [← c111474, ← c111475, k2, ← c111476] at e_1826_3
  have hr := e_1826_3
  simp only [e_1826_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7388)) := by
  have c111477 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111478 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111479 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111480 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111481 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111482 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_4 := arithEq_of_rows h (row := 1826) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_4
  simp only [← c111477, ← c111478, ← c111479] at e_1826_4
  have e_1826_5 := arithEq_of_rows h (row := 1826) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_5
  simp only [← c111480, ← c111481, k2, ← c111482] at e_1826_5
  have hr := e_1826_5
  simp only [e_1826_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7389)) := by
  have c111483 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111484 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111485 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111486 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111487 := (publicBatchWrapper86_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111488 := (publicBatchWrapper86_copies3484 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_6 := arithEq_of_rows h (row := 1826) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_6
  simp only [← c111483, ← c111484, ← c111485] at e_1826_6
  have e_1826_7 := arithEq_of_rows h (row := 1826) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_7
  simp only [← c111486, ← c111487, k2, ← c111488] at e_1826_7
  have hr := e_1826_7
  simp only [e_1826_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7390)) := by
  have c111489 := (publicBatchWrapper86_copies3484 a h).2.1
  have c111490 := (publicBatchWrapper86_copies3484 a h).2.2.1
  have c111491 := (publicBatchWrapper86_copies3484 a h).2.2.2.1
  have c111492 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.1
  have c111493 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.1
  have c111494 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_8 := arithEq_of_rows h (row := 1826) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_8
  simp only [← c111489, ← c111490, ← c111491] at e_1826_8
  have e_1826_9 := arithEq_of_rows h (row := 1826) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_9
  simp only [← c111492, ← c111493, k2, ← c111494] at e_1826_9
  have hr := e_1826_9
  simp only [e_1826_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7391)) := by
  have c111495 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.1
  have c111496 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.1
  have c111497 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.1
  have c111498 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111499 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111500 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_10 := arithEq_of_rows h (row := 1826) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_10
  simp only [← c111495, ← c111496, ← c111497] at e_1826_10
  have e_1826_11 := arithEq_of_rows h (row := 1826) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_11
  simp only [← c111498, ← c111499, k2, ← c111500] at e_1826_11
  have hr := e_1826_11
  simp only [e_1826_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 55) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7392)) := by
  have c111501 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111502 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111503 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111504 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111505 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111506 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_12 := arithEq_of_rows h (row := 1826) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_12
  simp only [← c111501, ← c111502, ← c111503] at e_1826_12
  have e_1826_13 := arithEq_of_rows h (row := 1826) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_13
  simp only [← c111504, ← c111505, k2, ← c111506] at e_1826_13
  have hr := e_1826_13
  simp only [e_1826_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 63) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7393)) := by
  have c111507 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111508 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111509 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111510 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111511 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111512 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_14 := arithEq_of_rows h (row := 1826) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_14
  simp only [← c111507, ← c111508, ← c111509] at e_1826_14
  have e_1826_15 := arithEq_of_rows h (row := 1826) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_15
  simp only [← c111510, ← c111511, k2, ← c111512] at e_1826_15
  have hr := e_1826_15
  simp only [e_1826_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 71) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7394)) := by
  have c111513 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111514 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111515 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111516 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111517 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111518 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_16 := arithEq_of_rows h (row := 1826) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_16
  simp only [← c111513, ← c111514, ← c111515] at e_1826_16
  have e_1826_17 := arithEq_of_rows h (row := 1826) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_17
  simp only [← c111516, ← c111517, k2, ← c111518] at e_1826_17
  have hr := e_1826_17
  simp only [e_1826_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1826 79) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7395)) := by
  have c111519 := (publicBatchWrapper86_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111520 := (publicBatchWrapper86_copies3485 a h).1
  have c111521 := (publicBatchWrapper86_copies3485 a h).2.1
  have c111522 := (publicBatchWrapper86_copies3485 a h).2.2.1
  have c111523 := (publicBatchWrapper86_copies3485 a h).2.2.2.1
  have c111524 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1826_18 := arithEq_of_rows h (row := 1826) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_18
  simp only [← c111519, ← c111520, ← c111521] at e_1826_18
  have e_1826_19 := arithEq_of_rows h (row := 1826) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_19
  simp only [← c111522, ← c111523, k2, ← c111524] at e_1826_19
  have hr := e_1826_19
  simp only [e_1826_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 7) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7396)) := by
  have c111525 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.1
  have c111526 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.1
  have c111527 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.1
  have c111528 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.1
  have c111529 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.1
  have c111530 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_0 := arithEq_of_rows h (row := 1827) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_0
  simp only [← c111525, ← c111526, ← c111527] at e_1827_0
  have e_1827_1 := arithEq_of_rows h (row := 1827) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_1
  simp only [← c111528, ← c111529, k2, ← c111530] at e_1827_1
  have hr := e_1827_1
  simp only [e_1827_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 15) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7397)) := by
  have c111531 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111532 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111533 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111534 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111535 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111536 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_2 := arithEq_of_rows h (row := 1827) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_2
  simp only [← c111531, ← c111532, ← c111533] at e_1827_2
  have e_1827_3 := arithEq_of_rows h (row := 1827) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_3
  simp only [← c111534, ← c111535, k2, ← c111536] at e_1827_3
  have hr := e_1827_3
  simp only [e_1827_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 23) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7398)) := by
  have c111537 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111538 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111539 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111540 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111541 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111542 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_4 := arithEq_of_rows h (row := 1827) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_4
  simp only [← c111537, ← c111538, ← c111539] at e_1827_4
  have e_1827_5 := arithEq_of_rows h (row := 1827) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_5
  simp only [← c111540, ← c111541, k2, ← c111542] at e_1827_5
  have hr := e_1827_5
  simp only [e_1827_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 31) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7399)) := by
  have c111543 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111544 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111545 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111546 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111547 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111548 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_6 := arithEq_of_rows h (row := 1827) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_6
  simp only [← c111543, ← c111544, ← c111545] at e_1827_6
  have e_1827_7 := arithEq_of_rows h (row := 1827) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_7
  simp only [← c111546, ← c111547, k2, ← c111548] at e_1827_7
  have hr := e_1827_7
  simp only [e_1827_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 39) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7400)) := by
  have c111549 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111550 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111551 := (publicBatchWrapper86_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111552 := (publicBatchWrapper86_copies3486 a h).1
  have c111553 := (publicBatchWrapper86_copies3486 a h).2.1
  have c111554 := (publicBatchWrapper86_copies3486 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_8 := arithEq_of_rows h (row := 1827) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_8
  simp only [← c111549, ← c111550, ← c111551] at e_1827_8
  have e_1827_9 := arithEq_of_rows h (row := 1827) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_9
  simp only [← c111552, ← c111553, k2, ← c111554] at e_1827_9
  have hr := e_1827_9
  simp only [e_1827_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 47) = bselect (a (.wire 37 11)) (a (.virt 200)) (a (.virt 7401)) := by
  have c111555 := (publicBatchWrapper86_copies3486 a h).2.2.2.1
  have c111556 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.1
  have c111557 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.1
  have c111558 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.1
  have c111559 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.1
  have c111560 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_10 := arithEq_of_rows h (row := 1827) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_10
  simp only [← c111555, ← c111556, ← c111557] at e_1827_10
  have e_1827_11 := arithEq_of_rows h (row := 1827) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_11
  simp only [← c111558, ← c111559, k2, ← c111560] at e_1827_11
  have hr := e_1827_11
  simp only [e_1827_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7570)) := by
  have c111561 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.1
  have c111562 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111563 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111564 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111565 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111566 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_12 := arithEq_of_rows h (row := 1827) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_12
  simp only [← c111561, ← c111562, ← c111563] at e_1827_12
  have e_1827_13 := arithEq_of_rows h (row := 1827) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_13
  simp only [← c111564, ← c111565, k2, ← c111566] at e_1827_13
  have hr := e_1827_13
  simp only [e_1827_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7571)) := by
  have c111567 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111568 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111569 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111570 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111571 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111572 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_14 := arithEq_of_rows h (row := 1827) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_14
  simp only [← c111567, ← c111568, ← c111569] at e_1827_14
  have e_1827_15 := arithEq_of_rows h (row := 1827) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_15
  simp only [← c111570, ← c111571, k2, ← c111572] at e_1827_15
  have hr := e_1827_15
  simp only [e_1827_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7572)) := by
  have c111573 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111574 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111575 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111576 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111577 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111578 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_16 := arithEq_of_rows h (row := 1827) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_16
  simp only [← c111573, ← c111574, ← c111575] at e_1827_16
  have e_1827_17 := arithEq_of_rows h (row := 1827) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_17
  simp only [← c111576, ← c111577, k2, ← c111578] at e_1827_17
  have hr := e_1827_17
  simp only [e_1827_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1827 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7573)) := by
  have c111579 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111580 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111581 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111582 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111583 := (publicBatchWrapper86_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111584 := (publicBatchWrapper86_copies3487 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1827_18 := arithEq_of_rows h (row := 1827) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_18
  simp only [← c111579, ← c111580, ← c111581] at e_1827_18
  have e_1827_19 := arithEq_of_rows h (row := 1827) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_19
  simp only [← c111582, ← c111583, k2, ← c111584] at e_1827_19
  have hr := e_1827_19
  simp only [e_1827_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7574)) := by
  have c111585 := (publicBatchWrapper86_copies3487 a h).2.1
  have c111586 := (publicBatchWrapper86_copies3487 a h).2.2.1
  have c111587 := (publicBatchWrapper86_copies3487 a h).2.2.2.1
  have c111588 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.1
  have c111589 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.1
  have c111590 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_0 := arithEq_of_rows h (row := 1828) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_0
  simp only [← c111585, ← c111586, ← c111587] at e_1828_0
  have e_1828_1 := arithEq_of_rows h (row := 1828) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_1
  simp only [← c111588, ← c111589, k2, ← c111590] at e_1828_1
  have hr := e_1828_1
  simp only [e_1828_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7575)) := by
  have c111591 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.1
  have c111592 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.1
  have c111593 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.1
  have c111594 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111595 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111596 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_2 := arithEq_of_rows h (row := 1828) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_2
  simp only [← c111591, ← c111592, ← c111593] at e_1828_2
  have e_1828_3 := arithEq_of_rows h (row := 1828) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_3
  simp only [← c111594, ← c111595, k2, ← c111596] at e_1828_3
  have hr := e_1828_3
  simp only [e_1828_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7576)) := by
  have c111597 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111598 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111599 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111600 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111601 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111602 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_4 := arithEq_of_rows h (row := 1828) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_4
  simp only [← c111597, ← c111598, ← c111599] at e_1828_4
  have e_1828_5 := arithEq_of_rows h (row := 1828) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_5
  simp only [← c111600, ← c111601, k2, ← c111602] at e_1828_5
  have hr := e_1828_5
  simp only [e_1828_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7577)) := by
  have c111603 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111604 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111605 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111606 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111607 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111608 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_6 := arithEq_of_rows h (row := 1828) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_6
  simp only [← c111603, ← c111604, ← c111605] at e_1828_6
  have e_1828_7 := arithEq_of_rows h (row := 1828) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_7
  simp only [← c111606, ← c111607, k2, ← c111608] at e_1828_7
  have hr := e_1828_7
  simp only [e_1828_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7578)) := by
  have c111609 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111610 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111611 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111612 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111613 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111614 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_8 := arithEq_of_rows h (row := 1828) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_8
  simp only [← c111609, ← c111610, ← c111611] at e_1828_8
  have e_1828_9 := arithEq_of_rows h (row := 1828) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_9
  simp only [← c111612, ← c111613, k2, ← c111614] at e_1828_9
  have hr := e_1828_9
  simp only [e_1828_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7579)) := by
  have c111615 := (publicBatchWrapper86_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111616 := (publicBatchWrapper86_copies3488 a h).1
  have c111617 := (publicBatchWrapper86_copies3488 a h).2.1
  have c111618 := (publicBatchWrapper86_copies3488 a h).2.2.1
  have c111619 := (publicBatchWrapper86_copies3488 a h).2.2.2.1
  have c111620 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_10 := arithEq_of_rows h (row := 1828) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_10
  simp only [← c111615, ← c111616, ← c111617] at e_1828_10
  have e_1828_11 := arithEq_of_rows h (row := 1828) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_11
  simp only [← c111618, ← c111619, k2, ← c111620] at e_1828_11
  have hr := e_1828_11
  simp only [e_1828_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7580)) := by
  have c111621 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.1
  have c111622 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.1
  have c111623 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.1
  have c111624 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.1
  have c111625 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.1
  have c111626 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_12 := arithEq_of_rows h (row := 1828) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_12
  simp only [← c111621, ← c111622, ← c111623] at e_1828_12
  have e_1828_13 := arithEq_of_rows h (row := 1828) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_13
  simp only [← c111624, ← c111625, k2, ← c111626] at e_1828_13
  have hr := e_1828_13
  simp only [e_1828_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7581)) := by
  have c111627 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111628 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111629 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111630 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111631 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111632 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_14 := arithEq_of_rows h (row := 1828) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_14
  simp only [← c111627, ← c111628, ← c111629] at e_1828_14
  have e_1828_15 := arithEq_of_rows h (row := 1828) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_15
  simp only [← c111630, ← c111631, k2, ← c111632] at e_1828_15
  have hr := e_1828_15
  simp only [e_1828_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7582)) := by
  have c111633 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111634 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111635 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111636 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111637 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111638 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_16 := arithEq_of_rows h (row := 1828) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_16
  simp only [← c111633, ← c111634, ← c111635] at e_1828_16
  have e_1828_17 := arithEq_of_rows h (row := 1828) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_17
  simp only [← c111636, ← c111637, k2, ← c111638] at e_1828_17
  have hr := e_1828_17
  simp only [e_1828_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1828 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7583)) := by
  have c111639 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111640 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111641 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111642 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111643 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111644 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1828_18 := arithEq_of_rows h (row := 1828) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_18
  simp only [← c111639, ← c111640, ← c111641] at e_1828_18
  have e_1828_19 := arithEq_of_rows h (row := 1828) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_19
  simp only [← c111642, ← c111643, k2, ← c111644] at e_1828_19
  have hr := e_1828_19
  simp only [e_1828_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7584)) := by
  have c111645 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111646 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111647 := (publicBatchWrapper86_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111648 := (publicBatchWrapper86_copies3489 a h).1
  have c111649 := (publicBatchWrapper86_copies3489 a h).2.1
  have c111650 := (publicBatchWrapper86_copies3489 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_0 := arithEq_of_rows h (row := 1829) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_0
  simp only [← c111645, ← c111646, ← c111647] at e_1829_0
  have e_1829_1 := arithEq_of_rows h (row := 1829) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_1
  simp only [← c111648, ← c111649, k2, ← c111650] at e_1829_1
  have hr := e_1829_1
  simp only [e_1829_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7585)) := by
  have c111651 := (publicBatchWrapper86_copies3489 a h).2.2.2.1
  have c111652 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.1
  have c111653 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.1
  have c111654 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.1
  have c111655 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.1
  have c111656 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_2 := arithEq_of_rows h (row := 1829) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_2
  simp only [← c111651, ← c111652, ← c111653] at e_1829_2
  have e_1829_3 := arithEq_of_rows h (row := 1829) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_3
  simp only [← c111654, ← c111655, k2, ← c111656] at e_1829_3
  have hr := e_1829_3
  simp only [e_1829_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7586)) := by
  have c111657 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.1
  have c111658 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111659 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111660 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111661 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111662 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_4 := arithEq_of_rows h (row := 1829) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_4
  simp only [← c111657, ← c111658, ← c111659] at e_1829_4
  have e_1829_5 := arithEq_of_rows h (row := 1829) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_5
  simp only [← c111660, ← c111661, k2, ← c111662] at e_1829_5
  have hr := e_1829_5
  simp only [e_1829_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7587)) := by
  have c111663 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111664 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111665 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111666 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111667 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111668 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_6 := arithEq_of_rows h (row := 1829) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_6
  simp only [← c111663, ← c111664, ← c111665] at e_1829_6
  have e_1829_7 := arithEq_of_rows h (row := 1829) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_7
  simp only [← c111666, ← c111667, k2, ← c111668] at e_1829_7
  have hr := e_1829_7
  simp only [e_1829_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7588)) := by
  have c111669 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111670 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111671 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111672 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111673 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111674 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_8 := arithEq_of_rows h (row := 1829) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_8
  simp only [← c111669, ← c111670, ← c111671] at e_1829_8
  have e_1829_9 := arithEq_of_rows h (row := 1829) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_9
  simp only [← c111672, ← c111673, k2, ← c111674] at e_1829_9
  have hr := e_1829_9
  simp only [e_1829_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7589)) := by
  have c111675 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111676 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111677 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111678 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111679 := (publicBatchWrapper86_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111680 := (publicBatchWrapper86_copies3490 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_10 := arithEq_of_rows h (row := 1829) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_10
  simp only [← c111675, ← c111676, ← c111677] at e_1829_10
  have e_1829_11 := arithEq_of_rows h (row := 1829) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_11
  simp only [← c111678, ← c111679, k2, ← c111680] at e_1829_11
  have hr := e_1829_11
  simp only [e_1829_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7590)) := by
  have c111681 := (publicBatchWrapper86_copies3490 a h).2.1
  have c111682 := (publicBatchWrapper86_copies3490 a h).2.2.1
  have c111683 := (publicBatchWrapper86_copies3490 a h).2.2.2.1
  have c111684 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.1
  have c111685 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.1
  have c111686 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_12 := arithEq_of_rows h (row := 1829) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_12
  simp only [← c111681, ← c111682, ← c111683] at e_1829_12
  have e_1829_13 := arithEq_of_rows h (row := 1829) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_13
  simp only [← c111684, ← c111685, k2, ← c111686] at e_1829_13
  have hr := e_1829_13
  simp only [e_1829_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7591)) := by
  have c111687 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.1
  have c111688 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.1
  have c111689 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.1
  have c111690 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111691 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111692 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_14 := arithEq_of_rows h (row := 1829) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_14
  simp only [← c111687, ← c111688, ← c111689] at e_1829_14
  have e_1829_15 := arithEq_of_rows h (row := 1829) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_15
  simp only [← c111690, ← c111691, k2, ← c111692] at e_1829_15
  have hr := e_1829_15
  simp only [e_1829_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 71) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7592)) := by
  have c111693 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111694 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111695 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111696 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111697 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111698 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_16 := arithEq_of_rows h (row := 1829) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_16
  simp only [← c111693, ← c111694, ← c111695] at e_1829_16
  have e_1829_17 := arithEq_of_rows h (row := 1829) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_17
  simp only [← c111696, ← c111697, k2, ← c111698] at e_1829_17
  have hr := e_1829_17
  simp only [e_1829_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1829 79) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7593)) := by
  have c111699 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111700 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111701 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111702 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111703 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111704 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1829_18 := arithEq_of_rows h (row := 1829) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_18
  simp only [← c111699, ← c111700, ← c111701] at e_1829_18
  have e_1829_19 := arithEq_of_rows h (row := 1829) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_19
  simp only [← c111702, ← c111703, k2, ← c111704] at e_1829_19
  have hr := e_1829_19
  simp only [e_1829_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 7) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7594)) := by
  have c111705 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111706 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111707 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111708 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111709 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111710 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_0 := arithEq_of_rows h (row := 1830) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_0
  simp only [← c111705, ← c111706, ← c111707] at e_1830_0
  have e_1830_1 := arithEq_of_rows h (row := 1830) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_1
  simp only [← c111708, ← c111709, k2, ← c111710] at e_1830_1
  have hr := e_1830_1
  simp only [e_1830_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 15) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7595)) := by
  have c111711 := (publicBatchWrapper86_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111712 := (publicBatchWrapper86_copies3491 a h).1
  have c111713 := (publicBatchWrapper86_copies3491 a h).2.1
  have c111714 := (publicBatchWrapper86_copies3491 a h).2.2.1
  have c111715 := (publicBatchWrapper86_copies3491 a h).2.2.2.1
  have c111716 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_2 := arithEq_of_rows h (row := 1830) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_2
  simp only [← c111711, ← c111712, ← c111713] at e_1830_2
  have e_1830_3 := arithEq_of_rows h (row := 1830) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_3
  simp only [← c111714, ← c111715, k2, ← c111716] at e_1830_3
  have hr := e_1830_3
  simp only [e_1830_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 23) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7596)) := by
  have c111717 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.1
  have c111718 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.1
  have c111719 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.1
  have c111720 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.1
  have c111721 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.1
  have c111722 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_4 := arithEq_of_rows h (row := 1830) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_4
  simp only [← c111717, ← c111718, ← c111719] at e_1830_4
  have e_1830_5 := arithEq_of_rows h (row := 1830) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_5
  simp only [← c111720, ← c111721, k2, ← c111722] at e_1830_5
  have hr := e_1830_5
  simp only [e_1830_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 31) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7597)) := by
  have c111723 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111724 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111725 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111726 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111727 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111728 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_6 := arithEq_of_rows h (row := 1830) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_6
  simp only [← c111723, ← c111724, ← c111725] at e_1830_6
  have e_1830_7 := arithEq_of_rows h (row := 1830) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_7
  simp only [← c111726, ← c111727, k2, ← c111728] at e_1830_7
  have hr := e_1830_7
  simp only [e_1830_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 39) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7598)) := by
  have c111729 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111730 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111731 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111732 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111733 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111734 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_8 := arithEq_of_rows h (row := 1830) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_8
  simp only [← c111729, ← c111730, ← c111731] at e_1830_8
  have e_1830_9 := arithEq_of_rows h (row := 1830) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_9
  simp only [← c111732, ← c111733, k2, ← c111734] at e_1830_9
  have hr := e_1830_9
  simp only [e_1830_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 47) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7599)) := by
  have c111735 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111736 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111737 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111738 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111739 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111740 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_10 := arithEq_of_rows h (row := 1830) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_10
  simp only [← c111735, ← c111736, ← c111737] at e_1830_10
  have e_1830_11 := arithEq_of_rows h (row := 1830) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_11
  simp only [← c111738, ← c111739, k2, ← c111740] at e_1830_11
  have hr := e_1830_11
  simp only [e_1830_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 55) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7600)) := by
  have c111741 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111742 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111743 := (publicBatchWrapper86_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111744 := (publicBatchWrapper86_copies3492 a h).1
  have c111745 := (publicBatchWrapper86_copies3492 a h).2.1
  have c111746 := (publicBatchWrapper86_copies3492 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_12 := arithEq_of_rows h (row := 1830) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_12
  simp only [← c111741, ← c111742, ← c111743] at e_1830_12
  have e_1830_13 := arithEq_of_rows h (row := 1830) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_13
  simp only [← c111744, ← c111745, k2, ← c111746] at e_1830_13
  have hr := e_1830_13
  simp only [e_1830_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 63) = bselect (a (.wire 37 55)) (a (.virt 200)) (a (.virt 7601)) := by
  have c111747 := (publicBatchWrapper86_copies3492 a h).2.2.2.1
  have c111748 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.1
  have c111749 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.1
  have c111750 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.1
  have c111751 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.1
  have c111752 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_14 := arithEq_of_rows h (row := 1830) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_14
  simp only [← c111747, ← c111748, ← c111749] at e_1830_14
  have e_1830_15 := arithEq_of_rows h (row := 1830) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_15
  simp only [← c111750, ← c111751, k2, ← c111752] at e_1830_15
  have hr := e_1830_15
  simp only [e_1830_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7770)) := by
  have c111753 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.1
  have c111754 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111755 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111756 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111757 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111758 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_16 := arithEq_of_rows h (row := 1830) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_16
  simp only [← c111753, ← c111754, ← c111755] at e_1830_16
  have e_1830_17 := arithEq_of_rows h (row := 1830) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_17
  simp only [← c111756, ← c111757, k2, ← c111758] at e_1830_17
  have hr := e_1830_17
  simp only [e_1830_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1830 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7771)) := by
  have c111759 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111760 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111761 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111762 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111763 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111764 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1830_18 := arithEq_of_rows h (row := 1830) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_18
  simp only [← c111759, ← c111760, ← c111761] at e_1830_18
  have e_1830_19 := arithEq_of_rows h (row := 1830) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_19
  simp only [← c111762, ← c111763, k2, ← c111764] at e_1830_19
  have hr := e_1830_19
  simp only [e_1830_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7772)) := by
  have c111765 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111766 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111767 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111768 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111769 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111770 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_0 := arithEq_of_rows h (row := 1831) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_0
  simp only [← c111765, ← c111766, ← c111767] at e_1831_0
  have e_1831_1 := arithEq_of_rows h (row := 1831) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_1
  simp only [← c111768, ← c111769, k2, ← c111770] at e_1831_1
  have hr := e_1831_1
  simp only [e_1831_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7773)) := by
  have c111771 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111772 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111773 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111774 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111775 := (publicBatchWrapper86_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111776 := (publicBatchWrapper86_copies3493 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_2 := arithEq_of_rows h (row := 1831) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_2
  simp only [← c111771, ← c111772, ← c111773] at e_1831_2
  have e_1831_3 := arithEq_of_rows h (row := 1831) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_3
  simp only [← c111774, ← c111775, k2, ← c111776] at e_1831_3
  have hr := e_1831_3
  simp only [e_1831_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7774)) := by
  have c111777 := (publicBatchWrapper86_copies3493 a h).2.1
  have c111778 := (publicBatchWrapper86_copies3493 a h).2.2.1
  have c111779 := (publicBatchWrapper86_copies3493 a h).2.2.2.1
  have c111780 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.1
  have c111781 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.1
  have c111782 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_4 := arithEq_of_rows h (row := 1831) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_4
  simp only [← c111777, ← c111778, ← c111779] at e_1831_4
  have e_1831_5 := arithEq_of_rows h (row := 1831) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_5
  simp only [← c111780, ← c111781, k2, ← c111782] at e_1831_5
  have hr := e_1831_5
  simp only [e_1831_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7775)) := by
  have c111783 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.1
  have c111784 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.1
  have c111785 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.1
  have c111786 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111787 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111788 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_6 := arithEq_of_rows h (row := 1831) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_6
  simp only [← c111783, ← c111784, ← c111785] at e_1831_6
  have e_1831_7 := arithEq_of_rows h (row := 1831) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_7
  simp only [← c111786, ← c111787, k2, ← c111788] at e_1831_7
  have hr := e_1831_7
  simp only [e_1831_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7776)) := by
  have c111789 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111790 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111791 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111792 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111793 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111794 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_8 := arithEq_of_rows h (row := 1831) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_8
  simp only [← c111789, ← c111790, ← c111791] at e_1831_8
  have e_1831_9 := arithEq_of_rows h (row := 1831) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_9
  simp only [← c111792, ← c111793, k2, ← c111794] at e_1831_9
  have hr := e_1831_9
  simp only [e_1831_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7777)) := by
  have c111795 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111796 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111797 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111798 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111799 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111800 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_10 := arithEq_of_rows h (row := 1831) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_10
  simp only [← c111795, ← c111796, ← c111797] at e_1831_10
  have e_1831_11 := arithEq_of_rows h (row := 1831) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_11
  simp only [← c111798, ← c111799, k2, ← c111800] at e_1831_11
  have hr := e_1831_11
  simp only [e_1831_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7778)) := by
  have c111801 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111802 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111803 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111804 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111805 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111806 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_12 := arithEq_of_rows h (row := 1831) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_12
  simp only [← c111801, ← c111802, ← c111803] at e_1831_12
  have e_1831_13 := arithEq_of_rows h (row := 1831) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_13
  simp only [← c111804, ← c111805, k2, ← c111806] at e_1831_13
  have hr := e_1831_13
  simp only [e_1831_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7779)) := by
  have c111807 := (publicBatchWrapper86_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111808 := (publicBatchWrapper86_copies3494 a h).1
  have c111809 := (publicBatchWrapper86_copies3494 a h).2.1
  have c111810 := (publicBatchWrapper86_copies3494 a h).2.2.1
  have c111811 := (publicBatchWrapper86_copies3494 a h).2.2.2.1
  have c111812 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_14 := arithEq_of_rows h (row := 1831) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_14
  simp only [← c111807, ← c111808, ← c111809] at e_1831_14
  have e_1831_15 := arithEq_of_rows h (row := 1831) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_15
  simp only [← c111810, ← c111811, k2, ← c111812] at e_1831_15
  have hr := e_1831_15
  simp only [e_1831_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7780)) := by
  have c111813 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.1
  have c111814 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.1
  have c111815 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.1
  have c111816 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.1
  have c111817 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.1
  have c111818 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_16 := arithEq_of_rows h (row := 1831) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_16
  simp only [← c111813, ← c111814, ← c111815] at e_1831_16
  have e_1831_17 := arithEq_of_rows h (row := 1831) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_17
  simp only [← c111816, ← c111817, k2, ← c111818] at e_1831_17
  have hr := e_1831_17
  simp only [e_1831_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1831 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7781)) := by
  have c111819 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111820 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111821 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111822 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111823 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111824 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1831_18 := arithEq_of_rows h (row := 1831) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_18
  simp only [← c111819, ← c111820, ← c111821] at e_1831_18
  have e_1831_19 := arithEq_of_rows h (row := 1831) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_19
  simp only [← c111822, ← c111823, k2, ← c111824] at e_1831_19
  have hr := e_1831_19
  simp only [e_1831_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7782)) := by
  have c111825 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111826 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111827 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111828 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111829 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111830 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_0 := arithEq_of_rows h (row := 1832) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_0
  simp only [← c111825, ← c111826, ← c111827] at e_1832_0
  have e_1832_1 := arithEq_of_rows h (row := 1832) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_1
  simp only [← c111828, ← c111829, k2, ← c111830] at e_1832_1
  have hr := e_1832_1
  simp only [e_1832_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7783)) := by
  have c111831 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111832 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111833 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111834 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111835 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111836 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_2 := arithEq_of_rows h (row := 1832) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_2
  simp only [← c111831, ← c111832, ← c111833] at e_1832_2
  have e_1832_3 := arithEq_of_rows h (row := 1832) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_3
  simp only [← c111834, ← c111835, k2, ← c111836] at e_1832_3
  have hr := e_1832_3
  simp only [e_1832_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7784)) := by
  have c111837 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111838 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111839 := (publicBatchWrapper86_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111840 := (publicBatchWrapper86_copies3495 a h).1
  have c111841 := (publicBatchWrapper86_copies3495 a h).2.1
  have c111842 := (publicBatchWrapper86_copies3495 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_4 := arithEq_of_rows h (row := 1832) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_4
  simp only [← c111837, ← c111838, ← c111839] at e_1832_4
  have e_1832_5 := arithEq_of_rows h (row := 1832) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_5
  simp only [← c111840, ← c111841, k2, ← c111842] at e_1832_5
  have hr := e_1832_5
  simp only [e_1832_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7785)) := by
  have c111843 := (publicBatchWrapper86_copies3495 a h).2.2.2.1
  have c111844 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.1
  have c111845 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.1
  have c111846 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.1
  have c111847 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.1
  have c111848 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_6 := arithEq_of_rows h (row := 1832) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_6
  simp only [← c111843, ← c111844, ← c111845] at e_1832_6
  have e_1832_7 := arithEq_of_rows h (row := 1832) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_7
  simp only [← c111846, ← c111847, k2, ← c111848] at e_1832_7
  have hr := e_1832_7
  simp only [e_1832_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7786)) := by
  have c111849 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.1
  have c111850 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111851 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111852 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111853 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111854 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_8 := arithEq_of_rows h (row := 1832) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_8
  simp only [← c111849, ← c111850, ← c111851] at e_1832_8
  have e_1832_9 := arithEq_of_rows h (row := 1832) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_9
  simp only [← c111852, ← c111853, k2, ← c111854] at e_1832_9
  have hr := e_1832_9
  simp only [e_1832_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7787)) := by
  have c111855 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111856 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111857 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111858 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111859 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111860 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_10 := arithEq_of_rows h (row := 1832) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_10
  simp only [← c111855, ← c111856, ← c111857] at e_1832_10
  have e_1832_11 := arithEq_of_rows h (row := 1832) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_11
  simp only [← c111858, ← c111859, k2, ← c111860] at e_1832_11
  have hr := e_1832_11
  simp only [e_1832_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7788)) := by
  have c111861 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111862 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111863 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111864 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111865 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111866 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_12 := arithEq_of_rows h (row := 1832) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_12
  simp only [← c111861, ← c111862, ← c111863] at e_1832_12
  have e_1832_13 := arithEq_of_rows h (row := 1832) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_13
  simp only [← c111864, ← c111865, k2, ← c111866] at e_1832_13
  have hr := e_1832_13
  simp only [e_1832_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7789)) := by
  have c111867 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111868 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111869 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111870 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111871 := (publicBatchWrapper86_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111872 := (publicBatchWrapper86_copies3496 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_14 := arithEq_of_rows h (row := 1832) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_14
  simp only [← c111867, ← c111868, ← c111869] at e_1832_14
  have e_1832_15 := arithEq_of_rows h (row := 1832) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_15
  simp only [← c111870, ← c111871, k2, ← c111872] at e_1832_15
  have hr := e_1832_15
  simp only [e_1832_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7790)) := by
  have c111873 := (publicBatchWrapper86_copies3496 a h).2.1
  have c111874 := (publicBatchWrapper86_copies3496 a h).2.2.1
  have c111875 := (publicBatchWrapper86_copies3496 a h).2.2.2.1
  have c111876 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.1
  have c111877 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.1
  have c111878 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_16 := arithEq_of_rows h (row := 1832) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_16
  simp only [← c111873, ← c111874, ← c111875] at e_1832_16
  have e_1832_17 := arithEq_of_rows h (row := 1832) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_17
  simp only [← c111876, ← c111877, k2, ← c111878] at e_1832_17
  have hr := e_1832_17
  simp only [e_1832_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1832 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7791)) := by
  have c111879 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.1
  have c111880 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.1
  have c111881 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.1
  have c111882 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111883 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111884 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1832_18 := arithEq_of_rows h (row := 1832) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_18
  simp only [← c111879, ← c111880, ← c111881] at e_1832_18
  have e_1832_19 := arithEq_of_rows h (row := 1832) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_19
  simp only [← c111882, ← c111883, k2, ← c111884] at e_1832_19
  have hr := e_1832_19
  simp only [e_1832_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 7) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7792)) := by
  have c111885 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111886 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111887 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111888 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111889 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111890 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_0 := arithEq_of_rows h (row := 1833) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_0
  simp only [← c111885, ← c111886, ← c111887] at e_1833_0
  have e_1833_1 := arithEq_of_rows h (row := 1833) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_1
  simp only [← c111888, ← c111889, k2, ← c111890] at e_1833_1
  have hr := e_1833_1
  simp only [e_1833_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 15) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7793)) := by
  have c111891 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111892 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111893 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111894 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111895 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111896 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_2 := arithEq_of_rows h (row := 1833) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_2
  simp only [← c111891, ← c111892, ← c111893] at e_1833_2
  have e_1833_3 := arithEq_of_rows h (row := 1833) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_3
  simp only [← c111894, ← c111895, k2, ← c111896] at e_1833_3
  have hr := e_1833_3
  simp only [e_1833_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 23) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7794)) := by
  have c111897 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111898 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111899 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111900 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111901 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111902 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_4 := arithEq_of_rows h (row := 1833) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_4
  simp only [← c111897, ← c111898, ← c111899] at e_1833_4
  have e_1833_5 := arithEq_of_rows h (row := 1833) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_5
  simp only [← c111900, ← c111901, k2, ← c111902] at e_1833_5
  have hr := e_1833_5
  simp only [e_1833_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 31) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7795)) := by
  have c111903 := (publicBatchWrapper86_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111904 := (publicBatchWrapper86_copies3497 a h).1
  have c111905 := (publicBatchWrapper86_copies3497 a h).2.1
  have c111906 := (publicBatchWrapper86_copies3497 a h).2.2.1
  have c111907 := (publicBatchWrapper86_copies3497 a h).2.2.2.1
  have c111908 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_6 := arithEq_of_rows h (row := 1833) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_6
  simp only [← c111903, ← c111904, ← c111905] at e_1833_6
  have e_1833_7 := arithEq_of_rows h (row := 1833) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_7
  simp only [← c111906, ← c111907, k2, ← c111908] at e_1833_7
  have hr := e_1833_7
  simp only [e_1833_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 39) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7796)) := by
  have c111909 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.1
  have c111910 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.1
  have c111911 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.1
  have c111912 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.1
  have c111913 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.1
  have c111914 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_8 := arithEq_of_rows h (row := 1833) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_8
  simp only [← c111909, ← c111910, ← c111911] at e_1833_8
  have e_1833_9 := arithEq_of_rows h (row := 1833) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_9
  simp only [← c111912, ← c111913, k2, ← c111914] at e_1833_9
  have hr := e_1833_9
  simp only [e_1833_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 47) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7797)) := by
  have c111915 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111916 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111917 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111918 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111919 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111920 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_10 := arithEq_of_rows h (row := 1833) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_10
  simp only [← c111915, ← c111916, ← c111917] at e_1833_10
  have e_1833_11 := arithEq_of_rows h (row := 1833) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_11
  simp only [← c111918, ← c111919, k2, ← c111920] at e_1833_11
  have hr := e_1833_11
  simp only [e_1833_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 55) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7798)) := by
  have c111921 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111922 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111923 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111924 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111925 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111926 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_12 := arithEq_of_rows h (row := 1833) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_12
  simp only [← c111921, ← c111922, ← c111923] at e_1833_12
  have e_1833_13 := arithEq_of_rows h (row := 1833) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_13
  simp only [← c111924, ← c111925, k2, ← c111926] at e_1833_13
  have hr := e_1833_13
  simp only [e_1833_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 63) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7799)) := by
  have c111927 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111928 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111929 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111930 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111931 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111932 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_14 := arithEq_of_rows h (row := 1833) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_14
  simp only [← c111927, ← c111928, ← c111929] at e_1833_14
  have e_1833_15 := arithEq_of_rows h (row := 1833) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_15
  simp only [← c111930, ← c111931, k2, ← c111932] at e_1833_15
  have hr := e_1833_15
  simp only [e_1833_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 71) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7800)) := by
  have c111933 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111934 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111935 := (publicBatchWrapper86_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111936 := (publicBatchWrapper86_copies3498 a h).1
  have c111937 := (publicBatchWrapper86_copies3498 a h).2.1
  have c111938 := (publicBatchWrapper86_copies3498 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_16 := arithEq_of_rows h (row := 1833) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_16
  simp only [← c111933, ← c111934, ← c111935] at e_1833_16
  have e_1833_17 := arithEq_of_rows h (row := 1833) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_17
  simp only [← c111936, ← c111937, k2, ← c111938] at e_1833_17
  have hr := e_1833_17
  simp only [e_1833_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1833 79) = bselect (a (.wire 39 19)) (a (.virt 200)) (a (.virt 7801)) := by
  have c111939 := (publicBatchWrapper86_copies3498 a h).2.2.2.1
  have c111940 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.1
  have c111941 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.1
  have c111942 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.1
  have c111943 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.1
  have c111944 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1833_18 := arithEq_of_rows h (row := 1833) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_18
  simp only [← c111939, ← c111940, ← c111941] at e_1833_18
  have e_1833_19 := arithEq_of_rows h (row := 1833) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_19
  simp only [← c111942, ← c111943, k2, ← c111944] at e_1833_19
  have hr := e_1833_19
  simp only [e_1833_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7970)) := by
  have c111945 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.1
  have c111946 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111947 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111948 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111949 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111950 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_0 := arithEq_of_rows h (row := 1834) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_0
  simp only [← c111945, ← c111946, ← c111947] at e_1834_0
  have e_1834_1 := arithEq_of_rows h (row := 1834) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_1
  simp only [← c111948, ← c111949, k2, ← c111950] at e_1834_1
  have hr := e_1834_1
  simp only [e_1834_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7971)) := by
  have c111951 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111952 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111953 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111954 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111955 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111956 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_2 := arithEq_of_rows h (row := 1834) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_2
  simp only [← c111951, ← c111952, ← c111953] at e_1834_2
  have e_1834_3 := arithEq_of_rows h (row := 1834) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_3
  simp only [← c111954, ← c111955, k2, ← c111956] at e_1834_3
  have hr := e_1834_3
  simp only [e_1834_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7972)) := by
  have c111957 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111958 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111959 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111960 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111961 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111962 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_4 := arithEq_of_rows h (row := 1834) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_4
  simp only [← c111957, ← c111958, ← c111959] at e_1834_4
  have e_1834_5 := arithEq_of_rows h (row := 1834) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_5
  simp only [← c111960, ← c111961, k2, ← c111962] at e_1834_5
  have hr := e_1834_5
  simp only [e_1834_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7973)) := by
  have c111963 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111964 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111965 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111966 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111967 := (publicBatchWrapper86_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111968 := (publicBatchWrapper86_copies3499 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_6 := arithEq_of_rows h (row := 1834) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_6
  simp only [← c111963, ← c111964, ← c111965] at e_1834_6
  have e_1834_7 := arithEq_of_rows h (row := 1834) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_7
  simp only [← c111966, ← c111967, k2, ← c111968] at e_1834_7
  have hr := e_1834_7
  simp only [e_1834_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7974)) := by
  have c111969 := (publicBatchWrapper86_copies3499 a h).2.1
  have c111970 := (publicBatchWrapper86_copies3499 a h).2.2.1
  have c111971 := (publicBatchWrapper86_copies3499 a h).2.2.2.1
  have c111972 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.1
  have c111973 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.1
  have c111974 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_8 := arithEq_of_rows h (row := 1834) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_8
  simp only [← c111969, ← c111970, ← c111971] at e_1834_8
  have e_1834_9 := arithEq_of_rows h (row := 1834) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_9
  simp only [← c111972, ← c111973, k2, ← c111974] at e_1834_9
  have hr := e_1834_9
  simp only [e_1834_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 47) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7975)) := by
  have c111975 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.1
  have c111976 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.1
  have c111977 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.1
  have c111978 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111979 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111980 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_10 := arithEq_of_rows h (row := 1834) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_10
  simp only [← c111975, ← c111976, ← c111977] at e_1834_10
  have e_1834_11 := arithEq_of_rows h (row := 1834) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_11
  simp only [← c111978, ← c111979, k2, ← c111980] at e_1834_11
  have hr := e_1834_11
  simp only [e_1834_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 55) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7976)) := by
  have c111981 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111982 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111983 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111984 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111985 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111986 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_12 := arithEq_of_rows h (row := 1834) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_12
  simp only [← c111981, ← c111982, ← c111983] at e_1834_12
  have e_1834_13 := arithEq_of_rows h (row := 1834) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_13
  simp only [← c111984, ← c111985, k2, ← c111986] at e_1834_13
  have hr := e_1834_13
  simp only [e_1834_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 63) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7977)) := by
  have c111987 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111988 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111989 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111990 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111991 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111992 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_14 := arithEq_of_rows h (row := 1834) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_14
  simp only [← c111987, ← c111988, ← c111989] at e_1834_14
  have e_1834_15 := arithEq_of_rows h (row := 1834) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_15
  simp only [← c111990, ← c111991, k2, ← c111992] at e_1834_15
  have hr := e_1834_15
  simp only [e_1834_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 71) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7978)) := by
  have c111993 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111994 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111995 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111996 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111997 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111998 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_16 := arithEq_of_rows h (row := 1834) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_16
  simp only [← c111993, ← c111994, ← c111995] at e_1834_16
  have e_1834_17 := arithEq_of_rows h (row := 1834) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_17
  simp only [← c111996, ← c111997, k2, ← c111998] at e_1834_17
  have hr := e_1834_17
  simp only [e_1834_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1834 79) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7979)) := by
  have c111999 := (publicBatchWrapper86_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112000 := (publicBatchWrapper86_copies3500 a h).1
  have c112001 := (publicBatchWrapper86_copies3500 a h).2.1
  have c112002 := (publicBatchWrapper86_copies3500 a h).2.2.1
  have c112003 := (publicBatchWrapper86_copies3500 a h).2.2.2.1
  have c112004 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1834_18 := arithEq_of_rows h (row := 1834) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_18
  simp only [← c111999, ← c112000, ← c112001] at e_1834_18
  have e_1834_19 := arithEq_of_rows h (row := 1834) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_19
  simp only [← c112002, ← c112003, k2, ← c112004] at e_1834_19
  have hr := e_1834_19
  simp only [e_1834_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7980)) := by
  have c112005 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.1
  have c112006 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.1
  have c112007 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.1
  have c112008 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.1
  have c112009 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.1
  have c112010 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_0 := arithEq_of_rows h (row := 1835) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_0
  simp only [← c112005, ← c112006, ← c112007] at e_1835_0
  have e_1835_1 := arithEq_of_rows h (row := 1835) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_1
  simp only [← c112008, ← c112009, k2, ← c112010] at e_1835_1
  have hr := e_1835_1
  simp only [e_1835_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7981)) := by
  have c112011 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112012 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112013 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112014 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112015 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112016 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_2 := arithEq_of_rows h (row := 1835) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_2
  simp only [← c112011, ← c112012, ← c112013] at e_1835_2
  have e_1835_3 := arithEq_of_rows h (row := 1835) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_3
  simp only [← c112014, ← c112015, k2, ← c112016] at e_1835_3
  have hr := e_1835_3
  simp only [e_1835_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7982)) := by
  have c112017 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112018 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112019 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112020 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112021 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112022 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_4 := arithEq_of_rows h (row := 1835) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_4
  simp only [← c112017, ← c112018, ← c112019] at e_1835_4
  have e_1835_5 := arithEq_of_rows h (row := 1835) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_5
  simp only [← c112020, ← c112021, k2, ← c112022] at e_1835_5
  have hr := e_1835_5
  simp only [e_1835_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7983)) := by
  have c112023 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112024 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112025 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112026 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112027 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112028 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_6 := arithEq_of_rows h (row := 1835) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_6
  simp only [← c112023, ← c112024, ← c112025] at e_1835_6
  have e_1835_7 := arithEq_of_rows h (row := 1835) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_7
  simp only [← c112026, ← c112027, k2, ← c112028] at e_1835_7
  have hr := e_1835_7
  simp only [e_1835_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7984)) := by
  have c112029 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112030 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112031 := (publicBatchWrapper86_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112032 := (publicBatchWrapper86_copies3501 a h).1
  have c112033 := (publicBatchWrapper86_copies3501 a h).2.1
  have c112034 := (publicBatchWrapper86_copies3501 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_8 := arithEq_of_rows h (row := 1835) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_8
  simp only [← c112029, ← c112030, ← c112031] at e_1835_8
  have e_1835_9 := arithEq_of_rows h (row := 1835) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_9
  simp only [← c112032, ← c112033, k2, ← c112034] at e_1835_9
  have hr := e_1835_9
  simp only [e_1835_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 47) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7985)) := by
  have c112035 := (publicBatchWrapper86_copies3501 a h).2.2.2.1
  have c112036 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.1
  have c112037 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.1
  have c112038 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.1
  have c112039 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.1
  have c112040 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_10 := arithEq_of_rows h (row := 1835) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_10
  simp only [← c112035, ← c112036, ← c112037] at e_1835_10
  have e_1835_11 := arithEq_of_rows h (row := 1835) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_11
  simp only [← c112038, ← c112039, k2, ← c112040] at e_1835_11
  have hr := e_1835_11
  simp only [e_1835_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 55) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7986)) := by
  have c112041 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.1
  have c112042 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112043 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112044 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112045 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112046 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_12 := arithEq_of_rows h (row := 1835) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_12
  simp only [← c112041, ← c112042, ← c112043] at e_1835_12
  have e_1835_13 := arithEq_of_rows h (row := 1835) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_13
  simp only [← c112044, ← c112045, k2, ← c112046] at e_1835_13
  have hr := e_1835_13
  simp only [e_1835_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 63) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7987)) := by
  have c112047 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112048 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112049 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112050 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112051 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112052 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_14 := arithEq_of_rows h (row := 1835) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_14
  simp only [← c112047, ← c112048, ← c112049] at e_1835_14
  have e_1835_15 := arithEq_of_rows h (row := 1835) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_15
  simp only [← c112050, ← c112051, k2, ← c112052] at e_1835_15
  have hr := e_1835_15
  simp only [e_1835_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 71) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7988)) := by
  have c112053 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112054 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112055 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112056 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112057 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112058 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_16 := arithEq_of_rows h (row := 1835) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_16
  simp only [← c112053, ← c112054, ← c112055] at e_1835_16
  have e_1835_17 := arithEq_of_rows h (row := 1835) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_17
  simp only [← c112056, ← c112057, k2, ← c112058] at e_1835_17
  have hr := e_1835_17
  simp only [e_1835_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1835 79) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7989)) := by
  have c112059 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112060 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112061 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112062 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112063 := (publicBatchWrapper86_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112064 := (publicBatchWrapper86_copies3502 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1835_18 := arithEq_of_rows h (row := 1835) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_18
  simp only [← c112059, ← c112060, ← c112061] at e_1835_18
  have e_1835_19 := arithEq_of_rows h (row := 1835) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_19
  simp only [← c112062, ← c112063, k2, ← c112064] at e_1835_19
  have hr := e_1835_19
  simp only [e_1835_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7990)) := by
  have c112065 := (publicBatchWrapper86_copies3502 a h).2.1
  have c112066 := (publicBatchWrapper86_copies3502 a h).2.2.1
  have c112067 := (publicBatchWrapper86_copies3502 a h).2.2.2.1
  have c112068 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.1
  have c112069 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.1
  have c112070 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_0 := arithEq_of_rows h (row := 1836) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_0
  simp only [← c112065, ← c112066, ← c112067] at e_1836_0
  have e_1836_1 := arithEq_of_rows h (row := 1836) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_1
  simp only [← c112068, ← c112069, k2, ← c112070] at e_1836_1
  have hr := e_1836_1
  simp only [e_1836_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7991)) := by
  have c112071 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.1
  have c112072 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.1
  have c112073 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.1
  have c112074 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112075 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112076 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_2 := arithEq_of_rows h (row := 1836) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_2
  simp only [← c112071, ← c112072, ← c112073] at e_1836_2
  have e_1836_3 := arithEq_of_rows h (row := 1836) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_3
  simp only [← c112074, ← c112075, k2, ← c112076] at e_1836_3
  have hr := e_1836_3
  simp only [e_1836_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 23) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7992)) := by
  have c112077 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112078 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112079 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112080 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112081 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112082 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_4 := arithEq_of_rows h (row := 1836) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_4
  simp only [← c112077, ← c112078, ← c112079] at e_1836_4
  have e_1836_5 := arithEq_of_rows h (row := 1836) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_5
  simp only [← c112080, ← c112081, k2, ← c112082] at e_1836_5
  have hr := e_1836_5
  simp only [e_1836_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 31) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7993)) := by
  have c112083 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112084 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112085 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112086 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112087 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112088 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_6 := arithEq_of_rows h (row := 1836) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_6
  simp only [← c112083, ← c112084, ← c112085] at e_1836_6
  have e_1836_7 := arithEq_of_rows h (row := 1836) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_7
  simp only [← c112086, ← c112087, k2, ← c112088] at e_1836_7
  have hr := e_1836_7
  simp only [e_1836_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 39) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7994)) := by
  have c112089 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112090 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112091 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112092 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112093 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112094 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_8 := arithEq_of_rows h (row := 1836) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_8
  simp only [← c112089, ← c112090, ← c112091] at e_1836_8
  have e_1836_9 := arithEq_of_rows h (row := 1836) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_9
  simp only [← c112092, ← c112093, k2, ← c112094] at e_1836_9
  have hr := e_1836_9
  simp only [e_1836_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 47) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7995)) := by
  have c112095 := (publicBatchWrapper86_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112096 := (publicBatchWrapper86_copies3503 a h).1
  have c112097 := (publicBatchWrapper86_copies3503 a h).2.1
  have c112098 := (publicBatchWrapper86_copies3503 a h).2.2.1
  have c112099 := (publicBatchWrapper86_copies3503 a h).2.2.2.1
  have c112100 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_10 := arithEq_of_rows h (row := 1836) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_10
  simp only [← c112095, ← c112096, ← c112097] at e_1836_10
  have e_1836_11 := arithEq_of_rows h (row := 1836) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_11
  simp only [← c112098, ← c112099, k2, ← c112100] at e_1836_11
  have hr := e_1836_11
  simp only [e_1836_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 55) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7996)) := by
  have c112101 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.1
  have c112102 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.1
  have c112103 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.1
  have c112104 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.1
  have c112105 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.1
  have c112106 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_12 := arithEq_of_rows h (row := 1836) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_12
  simp only [← c112101, ← c112102, ← c112103] at e_1836_12
  have e_1836_13 := arithEq_of_rows h (row := 1836) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_13
  simp only [← c112104, ← c112105, k2, ← c112106] at e_1836_13
  have hr := e_1836_13
  simp only [e_1836_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 63) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7997)) := by
  have c112107 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112108 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112109 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112110 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112111 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112112 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_14 := arithEq_of_rows h (row := 1836) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_14
  simp only [← c112107, ← c112108, ← c112109] at e_1836_14
  have e_1836_15 := arithEq_of_rows h (row := 1836) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_15
  simp only [← c112110, ← c112111, k2, ← c112112] at e_1836_15
  have hr := e_1836_15
  simp only [e_1836_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 71) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7998)) := by
  have c112113 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112114 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112115 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112116 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112117 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112118 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_16 := arithEq_of_rows h (row := 1836) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_16
  simp only [← c112113, ← c112114, ← c112115] at e_1836_16
  have e_1836_17 := arithEq_of_rows h (row := 1836) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_17
  simp only [← c112116, ← c112117, k2, ← c112118] at e_1836_17
  have hr := e_1836_17
  simp only [e_1836_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1836 79) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 7999)) := by
  have c112119 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112120 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112121 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112122 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112123 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112124 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1836_18 := arithEq_of_rows h (row := 1836) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_18
  simp only [← c112119, ← c112120, ← c112121] at e_1836_18
  have e_1836_19 := arithEq_of_rows h (row := 1836) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_19
  simp only [← c112122, ← c112123, k2, ← c112124] at e_1836_19
  have hr := e_1836_19
  simp only [e_1836_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 7) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 8000)) := by
  have c112125 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112126 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112127 := (publicBatchWrapper86_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112128 := (publicBatchWrapper86_copies3504 a h).1
  have c112129 := (publicBatchWrapper86_copies3504 a h).2.1
  have c112130 := (publicBatchWrapper86_copies3504 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_0 := arithEq_of_rows h (row := 1837) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_0
  simp only [← c112125, ← c112126, ← c112127] at e_1837_0
  have e_1837_1 := arithEq_of_rows h (row := 1837) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_1
  simp only [← c112128, ← c112129, k2, ← c112130] at e_1837_1
  have hr := e_1837_1
  simp only [e_1837_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 15) = bselect (a (.wire 39 63)) (a (.virt 200)) (a (.virt 8001)) := by
  have c112131 := (publicBatchWrapper86_copies3504 a h).2.2.2.1
  have c112132 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.1
  have c112133 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.1
  have c112134 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.1
  have c112135 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.1
  have c112136 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_2 := arithEq_of_rows h (row := 1837) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_2
  simp only [← c112131, ← c112132, ← c112133] at e_1837_2
  have e_1837_3 := arithEq_of_rows h (row := 1837) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_3
  simp only [← c112134, ← c112135, k2, ← c112136] at e_1837_3
  have hr := e_1837_3
  simp only [e_1837_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 23) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8170)) := by
  have c112137 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.1
  have c112138 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112139 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112140 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112141 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112142 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_4 := arithEq_of_rows h (row := 1837) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_4
  simp only [← c112137, ← c112138, ← c112139] at e_1837_4
  have e_1837_5 := arithEq_of_rows h (row := 1837) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_5
  simp only [← c112140, ← c112141, k2, ← c112142] at e_1837_5
  have hr := e_1837_5
  simp only [e_1837_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 31) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8171)) := by
  have c112143 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112144 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112145 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112146 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112147 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112148 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_6 := arithEq_of_rows h (row := 1837) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_6
  simp only [← c112143, ← c112144, ← c112145] at e_1837_6
  have e_1837_7 := arithEq_of_rows h (row := 1837) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_7
  simp only [← c112146, ← c112147, k2, ← c112148] at e_1837_7
  have hr := e_1837_7
  simp only [e_1837_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 39) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8172)) := by
  have c112149 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112150 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112151 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112152 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112153 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112154 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_8 := arithEq_of_rows h (row := 1837) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_8
  simp only [← c112149, ← c112150, ← c112151] at e_1837_8
  have e_1837_9 := arithEq_of_rows h (row := 1837) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_9
  simp only [← c112152, ← c112153, k2, ← c112154] at e_1837_9
  have hr := e_1837_9
  simp only [e_1837_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 47) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8173)) := by
  have c112155 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112156 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112157 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112158 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112159 := (publicBatchWrapper86_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112160 := (publicBatchWrapper86_copies3505 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_10 := arithEq_of_rows h (row := 1837) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_10
  simp only [← c112155, ← c112156, ← c112157] at e_1837_10
  have e_1837_11 := arithEq_of_rows h (row := 1837) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_11
  simp only [← c112158, ← c112159, k2, ← c112160] at e_1837_11
  have hr := e_1837_11
  simp only [e_1837_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 55) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8174)) := by
  have c112161 := (publicBatchWrapper86_copies3505 a h).2.1
  have c112162 := (publicBatchWrapper86_copies3505 a h).2.2.1
  have c112163 := (publicBatchWrapper86_copies3505 a h).2.2.2.1
  have c112164 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.1
  have c112165 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.1
  have c112166 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_12 := arithEq_of_rows h (row := 1837) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_12
  simp only [← c112161, ← c112162, ← c112163] at e_1837_12
  have e_1837_13 := arithEq_of_rows h (row := 1837) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_13
  simp only [← c112164, ← c112165, k2, ← c112166] at e_1837_13
  have hr := e_1837_13
  simp only [e_1837_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 63) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8175)) := by
  have c112167 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.1
  have c112168 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.1
  have c112169 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.1
  have c112170 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112171 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112172 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_14 := arithEq_of_rows h (row := 1837) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_14
  simp only [← c112167, ← c112168, ← c112169] at e_1837_14
  have e_1837_15 := arithEq_of_rows h (row := 1837) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_15
  simp only [← c112170, ← c112171, k2, ← c112172] at e_1837_15
  have hr := e_1837_15
  simp only [e_1837_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 71) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8176)) := by
  have c112173 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112174 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112175 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112176 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112177 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112178 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_16 := arithEq_of_rows h (row := 1837) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_16
  simp only [← c112173, ← c112174, ← c112175] at e_1837_16
  have e_1837_17 := arithEq_of_rows h (row := 1837) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_17
  simp only [← c112176, ← c112177, k2, ← c112178] at e_1837_17
  have hr := e_1837_17
  simp only [e_1837_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1837 79) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8177)) := by
  have c112179 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112180 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112181 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112182 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112183 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112184 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1837_18 := arithEq_of_rows h (row := 1837) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_18
  simp only [← c112179, ← c112180, ← c112181] at e_1837_18
  have e_1837_19 := arithEq_of_rows h (row := 1837) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_19
  simp only [← c112182, ← c112183, k2, ← c112184] at e_1837_19
  have hr := e_1837_19
  simp only [e_1837_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 7) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8178)) := by
  have c112185 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112186 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112187 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112188 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112189 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112190 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_0 := arithEq_of_rows h (row := 1838) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_0
  simp only [← c112185, ← c112186, ← c112187] at e_1838_0
  have e_1838_1 := arithEq_of_rows h (row := 1838) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_1
  simp only [← c112188, ← c112189, k2, ← c112190] at e_1838_1
  have hr := e_1838_1
  simp only [e_1838_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 15) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8179)) := by
  have c112191 := (publicBatchWrapper86_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112192 := (publicBatchWrapper86_copies3506 a h).1
  have c112193 := (publicBatchWrapper86_copies3506 a h).2.1
  have c112194 := (publicBatchWrapper86_copies3506 a h).2.2.1
  have c112195 := (publicBatchWrapper86_copies3506 a h).2.2.2.1
  have c112196 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_2 := arithEq_of_rows h (row := 1838) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_2
  simp only [← c112191, ← c112192, ← c112193] at e_1838_2
  have e_1838_3 := arithEq_of_rows h (row := 1838) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_3
  simp only [← c112194, ← c112195, k2, ← c112196] at e_1838_3
  have hr := e_1838_3
  simp only [e_1838_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 23) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8180)) := by
  have c112197 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.1
  have c112198 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.1
  have c112199 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.1
  have c112200 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.1
  have c112201 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.1
  have c112202 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_4 := arithEq_of_rows h (row := 1838) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_4
  simp only [← c112197, ← c112198, ← c112199] at e_1838_4
  have e_1838_5 := arithEq_of_rows h (row := 1838) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_5
  simp only [← c112200, ← c112201, k2, ← c112202] at e_1838_5
  have hr := e_1838_5
  simp only [e_1838_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 31) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8181)) := by
  have c112203 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112204 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112205 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112206 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112207 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112208 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_6 := arithEq_of_rows h (row := 1838) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_6
  simp only [← c112203, ← c112204, ← c112205] at e_1838_6
  have e_1838_7 := arithEq_of_rows h (row := 1838) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_7
  simp only [← c112206, ← c112207, k2, ← c112208] at e_1838_7
  have hr := e_1838_7
  simp only [e_1838_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 39) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8182)) := by
  have c112209 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112210 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112211 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112212 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112213 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112214 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_8 := arithEq_of_rows h (row := 1838) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_8
  simp only [← c112209, ← c112210, ← c112211] at e_1838_8
  have e_1838_9 := arithEq_of_rows h (row := 1838) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_9
  simp only [← c112212, ← c112213, k2, ← c112214] at e_1838_9
  have hr := e_1838_9
  simp only [e_1838_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 47) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8183)) := by
  have c112215 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112216 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112217 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112218 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112219 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112220 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_10 := arithEq_of_rows h (row := 1838) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_10
  simp only [← c112215, ← c112216, ← c112217] at e_1838_10
  have e_1838_11 := arithEq_of_rows h (row := 1838) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_11
  simp only [← c112218, ← c112219, k2, ← c112220] at e_1838_11
  have hr := e_1838_11
  simp only [e_1838_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 55) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8184)) := by
  have c112221 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112222 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112223 := (publicBatchWrapper86_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112224 := (publicBatchWrapper86_copies3507 a h).1
  have c112225 := (publicBatchWrapper86_copies3507 a h).2.1
  have c112226 := (publicBatchWrapper86_copies3507 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_12 := arithEq_of_rows h (row := 1838) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_12
  simp only [← c112221, ← c112222, ← c112223] at e_1838_12
  have e_1838_13 := arithEq_of_rows h (row := 1838) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_13
  simp only [← c112224, ← c112225, k2, ← c112226] at e_1838_13
  have hr := e_1838_13
  simp only [e_1838_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 63) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8185)) := by
  have c112227 := (publicBatchWrapper86_copies3507 a h).2.2.2.1
  have c112228 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.1
  have c112229 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.1
  have c112230 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.1
  have c112231 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.1
  have c112232 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_14 := arithEq_of_rows h (row := 1838) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_14
  simp only [← c112227, ← c112228, ← c112229] at e_1838_14
  have e_1838_15 := arithEq_of_rows h (row := 1838) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_15
  simp only [← c112230, ← c112231, k2, ← c112232] at e_1838_15
  have hr := e_1838_15
  simp only [e_1838_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 71) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8186)) := by
  have c112233 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.1
  have c112234 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112235 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112236 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112237 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112238 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_16 := arithEq_of_rows h (row := 1838) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_16
  simp only [← c112233, ← c112234, ← c112235] at e_1838_16
  have e_1838_17 := arithEq_of_rows h (row := 1838) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_17
  simp only [← c112236, ← c112237, k2, ← c112238] at e_1838_17
  have hr := e_1838_17
  simp only [e_1838_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1838 79) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8187)) := by
  have c112239 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112240 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112241 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112242 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112243 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112244 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1838_18 := arithEq_of_rows h (row := 1838) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_18
  simp only [← c112239, ← c112240, ← c112241] at e_1838_18
  have e_1838_19 := arithEq_of_rows h (row := 1838) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_19
  simp only [← c112242, ← c112243, k2, ← c112244] at e_1838_19
  have hr := e_1838_19
  simp only [e_1838_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 7) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8188)) := by
  have c112245 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112246 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112247 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112248 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112249 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112250 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_0 := arithEq_of_rows h (row := 1839) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_0
  simp only [← c112245, ← c112246, ← c112247] at e_1839_0
  have e_1839_1 := arithEq_of_rows h (row := 1839) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_1
  simp only [← c112248, ← c112249, k2, ← c112250] at e_1839_1
  have hr := e_1839_1
  simp only [e_1839_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 15) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8189)) := by
  have c112251 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112252 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112253 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112254 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112255 := (publicBatchWrapper86_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112256 := (publicBatchWrapper86_copies3508 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_2 := arithEq_of_rows h (row := 1839) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_2
  simp only [← c112251, ← c112252, ← c112253] at e_1839_2
  have e_1839_3 := arithEq_of_rows h (row := 1839) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_3
  simp only [← c112254, ← c112255, k2, ← c112256] at e_1839_3
  have hr := e_1839_3
  simp only [e_1839_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 23) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8190)) := by
  have c112257 := (publicBatchWrapper86_copies3508 a h).2.1
  have c112258 := (publicBatchWrapper86_copies3508 a h).2.2.1
  have c112259 := (publicBatchWrapper86_copies3508 a h).2.2.2.1
  have c112260 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.1
  have c112261 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.1
  have c112262 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_4 := arithEq_of_rows h (row := 1839) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_4
  simp only [← c112257, ← c112258, ← c112259] at e_1839_4
  have e_1839_5 := arithEq_of_rows h (row := 1839) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_5
  simp only [← c112260, ← c112261, k2, ← c112262] at e_1839_5
  have hr := e_1839_5
  simp only [e_1839_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 31) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8191)) := by
  have c112263 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.1
  have c112264 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.1
  have c112265 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.1
  have c112266 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112267 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112268 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_6 := arithEq_of_rows h (row := 1839) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_6
  simp only [← c112263, ← c112264, ← c112265] at e_1839_6
  have e_1839_7 := arithEq_of_rows h (row := 1839) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_7
  simp only [← c112266, ← c112267, k2, ← c112268] at e_1839_7
  have hr := e_1839_7
  simp only [e_1839_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 39) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8192)) := by
  have c112269 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112270 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112271 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112272 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112273 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112274 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_8 := arithEq_of_rows h (row := 1839) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_8
  simp only [← c112269, ← c112270, ← c112271] at e_1839_8
  have e_1839_9 := arithEq_of_rows h (row := 1839) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_9
  simp only [← c112272, ← c112273, k2, ← c112274] at e_1839_9
  have hr := e_1839_9
  simp only [e_1839_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 47) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8193)) := by
  have c112275 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112276 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112277 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112278 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112279 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112280 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_10 := arithEq_of_rows h (row := 1839) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_10
  simp only [← c112275, ← c112276, ← c112277] at e_1839_10
  have e_1839_11 := arithEq_of_rows h (row := 1839) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_11
  simp only [← c112278, ← c112279, k2, ← c112280] at e_1839_11
  have hr := e_1839_11
  simp only [e_1839_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 55) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8194)) := by
  have c112281 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112282 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112283 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112284 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112285 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112286 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_12 := arithEq_of_rows h (row := 1839) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_12
  simp only [← c112281, ← c112282, ← c112283] at e_1839_12
  have e_1839_13 := arithEq_of_rows h (row := 1839) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_13
  simp only [← c112284, ← c112285, k2, ← c112286] at e_1839_13
  have hr := e_1839_13
  simp only [e_1839_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 63) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8195)) := by
  have c112287 := (publicBatchWrapper86_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112288 := (publicBatchWrapper86_copies3509 a h).1
  have c112289 := (publicBatchWrapper86_copies3509 a h).2.1
  have c112290 := (publicBatchWrapper86_copies3509 a h).2.2.1
  have c112291 := (publicBatchWrapper86_copies3509 a h).2.2.2.1
  have c112292 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_14 := arithEq_of_rows h (row := 1839) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_14
  simp only [← c112287, ← c112288, ← c112289] at e_1839_14
  have e_1839_15 := arithEq_of_rows h (row := 1839) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_15
  simp only [← c112290, ← c112291, k2, ← c112292] at e_1839_15
  have hr := e_1839_15
  simp only [e_1839_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 71) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8196)) := by
  have c112293 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.1
  have c112294 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.1
  have c112295 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.1
  have c112296 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.1
  have c112297 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.1
  have c112298 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_16 := arithEq_of_rows h (row := 1839) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_16
  simp only [← c112293, ← c112294, ← c112295] at e_1839_16
  have e_1839_17 := arithEq_of_rows h (row := 1839) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_17
  simp only [← c112296, ← c112297, k2, ← c112298] at e_1839_17
  have hr := e_1839_17
  simp only [e_1839_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1839 79) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8197)) := by
  have c112299 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112300 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112301 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112302 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112303 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112304 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1839_18 := arithEq_of_rows h (row := 1839) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_18
  simp only [← c112299, ← c112300, ← c112301] at e_1839_18
  have e_1839_19 := arithEq_of_rows h (row := 1839) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_19
  simp only [← c112302, ← c112303, k2, ← c112304] at e_1839_19
  have hr := e_1839_19
  simp only [e_1839_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 7) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8198)) := by
  have c112305 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112306 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112307 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112308 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112309 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112310 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_0 := arithEq_of_rows h (row := 1840) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_0
  simp only [← c112305, ← c112306, ← c112307] at e_1840_0
  have e_1840_1 := arithEq_of_rows h (row := 1840) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_1
  simp only [← c112308, ← c112309, k2, ← c112310] at e_1840_1
  have hr := e_1840_1
  simp only [e_1840_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 15) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8199)) := by
  have c112311 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112312 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112313 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112314 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112315 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112316 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_2 := arithEq_of_rows h (row := 1840) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_2
  simp only [← c112311, ← c112312, ← c112313] at e_1840_2
  have e_1840_3 := arithEq_of_rows h (row := 1840) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_3
  simp only [← c112314, ← c112315, k2, ← c112316] at e_1840_3
  have hr := e_1840_3
  simp only [e_1840_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 23) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8200)) := by
  have c112317 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112318 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112319 := (publicBatchWrapper86_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112320 := (publicBatchWrapper86_copies3510 a h).1
  have c112321 := (publicBatchWrapper86_copies3510 a h).2.1
  have c112322 := (publicBatchWrapper86_copies3510 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_4 := arithEq_of_rows h (row := 1840) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_4
  simp only [← c112317, ← c112318, ← c112319] at e_1840_4
  have e_1840_5 := arithEq_of_rows h (row := 1840) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_5
  simp only [← c112320, ← c112321, k2, ← c112322] at e_1840_5
  have hr := e_1840_5
  simp only [e_1840_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 31) = bselect (a (.wire 41 27)) (a (.virt 200)) (a (.virt 8201)) := by
  have c112323 := (publicBatchWrapper86_copies3510 a h).2.2.2.1
  have c112324 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.1
  have c112325 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.1
  have c112326 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.1
  have c112327 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.1
  have c112328 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_6 := arithEq_of_rows h (row := 1840) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_6
  simp only [← c112323, ← c112324, ← c112325] at e_1840_6
  have e_1840_7 := arithEq_of_rows h (row := 1840) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_7
  simp only [← c112326, ← c112327, k2, ← c112328] at e_1840_7
  have hr := e_1840_7
  simp only [e_1840_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 39) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8370)) := by
  have c112329 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.1
  have c112330 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112331 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112332 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112333 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112334 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_8 := arithEq_of_rows h (row := 1840) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_8
  simp only [← c112329, ← c112330, ← c112331] at e_1840_8
  have e_1840_9 := arithEq_of_rows h (row := 1840) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_9
  simp only [← c112332, ← c112333, k2, ← c112334] at e_1840_9
  have hr := e_1840_9
  simp only [e_1840_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 47) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8371)) := by
  have c112335 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112336 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112337 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112338 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112339 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112340 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_10 := arithEq_of_rows h (row := 1840) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_10
  simp only [← c112335, ← c112336, ← c112337] at e_1840_10
  have e_1840_11 := arithEq_of_rows h (row := 1840) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_11
  simp only [← c112338, ← c112339, k2, ← c112340] at e_1840_11
  have hr := e_1840_11
  simp only [e_1840_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 55) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8372)) := by
  have c112341 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112342 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112343 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112344 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112345 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112346 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_12 := arithEq_of_rows h (row := 1840) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_12
  simp only [← c112341, ← c112342, ← c112343] at e_1840_12
  have e_1840_13 := arithEq_of_rows h (row := 1840) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_13
  simp only [← c112344, ← c112345, k2, ← c112346] at e_1840_13
  have hr := e_1840_13
  simp only [e_1840_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 63) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8373)) := by
  have c112347 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112348 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112349 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112350 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112351 := (publicBatchWrapper86_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112352 := (publicBatchWrapper86_copies3511 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_14 := arithEq_of_rows h (row := 1840) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_14
  simp only [← c112347, ← c112348, ← c112349] at e_1840_14
  have e_1840_15 := arithEq_of_rows h (row := 1840) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_15
  simp only [← c112350, ← c112351, k2, ← c112352] at e_1840_15
  have hr := e_1840_15
  simp only [e_1840_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 71) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8374)) := by
  have c112353 := (publicBatchWrapper86_copies3511 a h).2.1
  have c112354 := (publicBatchWrapper86_copies3511 a h).2.2.1
  have c112355 := (publicBatchWrapper86_copies3511 a h).2.2.2.1
  have c112356 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.1
  have c112357 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.1
  have c112358 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_16 := arithEq_of_rows h (row := 1840) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_16
  simp only [← c112353, ← c112354, ← c112355] at e_1840_16
  have e_1840_17 := arithEq_of_rows h (row := 1840) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_17
  simp only [← c112356, ← c112357, k2, ← c112358] at e_1840_17
  have hr := e_1840_17
  simp only [e_1840_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1840 79) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8375)) := by
  have c112359 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.1
  have c112360 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.1
  have c112361 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.1
  have c112362 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112363 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112364 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1840_18 := arithEq_of_rows h (row := 1840) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_18
  simp only [← c112359, ← c112360, ← c112361] at e_1840_18
  have e_1840_19 := arithEq_of_rows h (row := 1840) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_19
  simp only [← c112362, ← c112363, k2, ← c112364] at e_1840_19
  have hr := e_1840_19
  simp only [e_1840_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 7) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8376)) := by
  have c112365 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112366 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112367 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112368 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112369 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112370 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_0 := arithEq_of_rows h (row := 1841) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_0
  simp only [← c112365, ← c112366, ← c112367] at e_1841_0
  have e_1841_1 := arithEq_of_rows h (row := 1841) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_1
  simp only [← c112368, ← c112369, k2, ← c112370] at e_1841_1
  have hr := e_1841_1
  simp only [e_1841_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 15) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8377)) := by
  have c112371 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112372 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112373 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112374 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112375 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112376 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_2 := arithEq_of_rows h (row := 1841) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_2
  simp only [← c112371, ← c112372, ← c112373] at e_1841_2
  have e_1841_3 := arithEq_of_rows h (row := 1841) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_3
  simp only [← c112374, ← c112375, k2, ← c112376] at e_1841_3
  have hr := e_1841_3
  simp only [e_1841_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 23) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8378)) := by
  have c112377 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112378 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112379 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112380 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112381 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112382 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_4 := arithEq_of_rows h (row := 1841) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_4
  simp only [← c112377, ← c112378, ← c112379] at e_1841_4
  have e_1841_5 := arithEq_of_rows h (row := 1841) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_5
  simp only [← c112380, ← c112381, k2, ← c112382] at e_1841_5
  have hr := e_1841_5
  simp only [e_1841_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 31) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8379)) := by
  have c112383 := (publicBatchWrapper86_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112384 := (publicBatchWrapper86_copies3512 a h).1
  have c112385 := (publicBatchWrapper86_copies3512 a h).2.1
  have c112386 := (publicBatchWrapper86_copies3512 a h).2.2.1
  have c112387 := (publicBatchWrapper86_copies3512 a h).2.2.2.1
  have c112388 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_6 := arithEq_of_rows h (row := 1841) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_6
  simp only [← c112383, ← c112384, ← c112385] at e_1841_6
  have e_1841_7 := arithEq_of_rows h (row := 1841) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_7
  simp only [← c112386, ← c112387, k2, ← c112388] at e_1841_7
  have hr := e_1841_7
  simp only [e_1841_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 39) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8380)) := by
  have c112389 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.1
  have c112390 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.1
  have c112391 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.1
  have c112392 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.1
  have c112393 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.1
  have c112394 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_8 := arithEq_of_rows h (row := 1841) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_8
  simp only [← c112389, ← c112390, ← c112391] at e_1841_8
  have e_1841_9 := arithEq_of_rows h (row := 1841) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_9
  simp only [← c112392, ← c112393, k2, ← c112394] at e_1841_9
  have hr := e_1841_9
  simp only [e_1841_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 47) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8381)) := by
  have c112395 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112396 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112397 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112398 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112399 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112400 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_10 := arithEq_of_rows h (row := 1841) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_10
  simp only [← c112395, ← c112396, ← c112397] at e_1841_10
  have e_1841_11 := arithEq_of_rows h (row := 1841) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_11
  simp only [← c112398, ← c112399, k2, ← c112400] at e_1841_11
  have hr := e_1841_11
  simp only [e_1841_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 55) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8382)) := by
  have c112401 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112402 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112403 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112404 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112405 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112406 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_12 := arithEq_of_rows h (row := 1841) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_12
  simp only [← c112401, ← c112402, ← c112403] at e_1841_12
  have e_1841_13 := arithEq_of_rows h (row := 1841) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_13
  simp only [← c112404, ← c112405, k2, ← c112406] at e_1841_13
  have hr := e_1841_13
  simp only [e_1841_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 63) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8383)) := by
  have c112407 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112408 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112409 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112410 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112411 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112412 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_14 := arithEq_of_rows h (row := 1841) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_14
  simp only [← c112407, ← c112408, ← c112409] at e_1841_14
  have e_1841_15 := arithEq_of_rows h (row := 1841) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_15
  simp only [← c112410, ← c112411, k2, ← c112412] at e_1841_15
  have hr := e_1841_15
  simp only [e_1841_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 71) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8384)) := by
  have c112413 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112414 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112415 := (publicBatchWrapper86_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112416 := (publicBatchWrapper86_copies3513 a h).1
  have c112417 := (publicBatchWrapper86_copies3513 a h).2.1
  have c112418 := (publicBatchWrapper86_copies3513 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_16 := arithEq_of_rows h (row := 1841) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_16
  simp only [← c112413, ← c112414, ← c112415] at e_1841_16
  have e_1841_17 := arithEq_of_rows h (row := 1841) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_17
  simp only [← c112416, ← c112417, k2, ← c112418] at e_1841_17
  have hr := e_1841_17
  simp only [e_1841_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1841 79) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8385)) := by
  have c112419 := (publicBatchWrapper86_copies3513 a h).2.2.2.1
  have c112420 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.1
  have c112421 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.1
  have c112422 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.1
  have c112423 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.1
  have c112424 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1841_18 := arithEq_of_rows h (row := 1841) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_18
  simp only [← c112419, ← c112420, ← c112421] at e_1841_18
  have e_1841_19 := arithEq_of_rows h (row := 1841) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_19
  simp only [← c112422, ← c112423, k2, ← c112424] at e_1841_19
  have hr := e_1841_19
  simp only [e_1841_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 7) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8386)) := by
  have c112425 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.1
  have c112426 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112427 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112428 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112429 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112430 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_0 := arithEq_of_rows h (row := 1842) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_0
  simp only [← c112425, ← c112426, ← c112427] at e_1842_0
  have e_1842_1 := arithEq_of_rows h (row := 1842) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_1
  simp only [← c112428, ← c112429, k2, ← c112430] at e_1842_1
  have hr := e_1842_1
  simp only [e_1842_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 15) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8387)) := by
  have c112431 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112432 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112433 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112434 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112435 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112436 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_2 := arithEq_of_rows h (row := 1842) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_2
  simp only [← c112431, ← c112432, ← c112433] at e_1842_2
  have e_1842_3 := arithEq_of_rows h (row := 1842) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_3
  simp only [← c112434, ← c112435, k2, ← c112436] at e_1842_3
  have hr := e_1842_3
  simp only [e_1842_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 23) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8388)) := by
  have c112437 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112438 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112439 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112440 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112441 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112442 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_4 := arithEq_of_rows h (row := 1842) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_4
  simp only [← c112437, ← c112438, ← c112439] at e_1842_4
  have e_1842_5 := arithEq_of_rows h (row := 1842) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_5
  simp only [← c112440, ← c112441, k2, ← c112442] at e_1842_5
  have hr := e_1842_5
  simp only [e_1842_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 31) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8389)) := by
  have c112443 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112444 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112445 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112446 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112447 := (publicBatchWrapper86_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112448 := (publicBatchWrapper86_copies3514 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_6 := arithEq_of_rows h (row := 1842) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_6
  simp only [← c112443, ← c112444, ← c112445] at e_1842_6
  have e_1842_7 := arithEq_of_rows h (row := 1842) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_7
  simp only [← c112446, ← c112447, k2, ← c112448] at e_1842_7
  have hr := e_1842_7
  simp only [e_1842_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 39) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8390)) := by
  have c112449 := (publicBatchWrapper86_copies3514 a h).2.1
  have c112450 := (publicBatchWrapper86_copies3514 a h).2.2.1
  have c112451 := (publicBatchWrapper86_copies3514 a h).2.2.2.1
  have c112452 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.1
  have c112453 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.1
  have c112454 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_8 := arithEq_of_rows h (row := 1842) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_8
  simp only [← c112449, ← c112450, ← c112451] at e_1842_8
  have e_1842_9 := arithEq_of_rows h (row := 1842) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_9
  simp only [← c112452, ← c112453, k2, ← c112454] at e_1842_9
  have hr := e_1842_9
  simp only [e_1842_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 47) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8391)) := by
  have c112455 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.1
  have c112456 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.1
  have c112457 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.1
  have c112458 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112459 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112460 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_10 := arithEq_of_rows h (row := 1842) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_10
  simp only [← c112455, ← c112456, ← c112457] at e_1842_10
  have e_1842_11 := arithEq_of_rows h (row := 1842) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_11
  simp only [← c112458, ← c112459, k2, ← c112460] at e_1842_11
  have hr := e_1842_11
  simp only [e_1842_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 55) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8392)) := by
  have c112461 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112462 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112463 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112464 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112465 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112466 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_12 := arithEq_of_rows h (row := 1842) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_12
  simp only [← c112461, ← c112462, ← c112463] at e_1842_12
  have e_1842_13 := arithEq_of_rows h (row := 1842) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_13
  simp only [← c112464, ← c112465, k2, ← c112466] at e_1842_13
  have hr := e_1842_13
  simp only [e_1842_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 63) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8393)) := by
  have c112467 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112468 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112469 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112470 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112471 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112472 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_14 := arithEq_of_rows h (row := 1842) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_14
  simp only [← c112467, ← c112468, ← c112469] at e_1842_14
  have e_1842_15 := arithEq_of_rows h (row := 1842) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_15
  simp only [← c112470, ← c112471, k2, ← c112472] at e_1842_15
  have hr := e_1842_15
  simp only [e_1842_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 71) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8394)) := by
  have c112473 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112474 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112475 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112476 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112477 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112478 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_16 := arithEq_of_rows h (row := 1842) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_16
  simp only [← c112473, ← c112474, ← c112475] at e_1842_16
  have e_1842_17 := arithEq_of_rows h (row := 1842) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_17
  simp only [← c112476, ← c112477, k2, ← c112478] at e_1842_17
  have hr := e_1842_17
  simp only [e_1842_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1842 79) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8395)) := by
  have c112479 := (publicBatchWrapper86_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112480 := (publicBatchWrapper86_copies3515 a h).1
  have c112481 := (publicBatchWrapper86_copies3515 a h).2.1
  have c112482 := (publicBatchWrapper86_copies3515 a h).2.2.1
  have c112483 := (publicBatchWrapper86_copies3515 a h).2.2.2.1
  have c112484 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1842_18 := arithEq_of_rows h (row := 1842) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_18
  simp only [← c112479, ← c112480, ← c112481] at e_1842_18
  have e_1842_19 := arithEq_of_rows h (row := 1842) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_19
  simp only [← c112482, ← c112483, k2, ← c112484] at e_1842_19
  have hr := e_1842_19
  simp only [e_1842_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 7) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8396)) := by
  have c112485 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.1
  have c112486 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.1
  have c112487 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.1
  have c112488 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.1
  have c112489 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.1
  have c112490 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_0 := arithEq_of_rows h (row := 1843) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_0
  simp only [← c112485, ← c112486, ← c112487] at e_1843_0
  have e_1843_1 := arithEq_of_rows h (row := 1843) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_1
  simp only [← c112488, ← c112489, k2, ← c112490] at e_1843_1
  have hr := e_1843_1
  simp only [e_1843_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 15) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8397)) := by
  have c112491 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112492 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112493 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112494 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112495 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112496 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_2 := arithEq_of_rows h (row := 1843) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_2
  simp only [← c112491, ← c112492, ← c112493] at e_1843_2
  have e_1843_3 := arithEq_of_rows h (row := 1843) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_3
  simp only [← c112494, ← c112495, k2, ← c112496] at e_1843_3
  have hr := e_1843_3
  simp only [e_1843_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 23) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8398)) := by
  have c112497 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112498 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112499 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112500 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112501 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112502 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_4 := arithEq_of_rows h (row := 1843) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_4
  simp only [← c112497, ← c112498, ← c112499] at e_1843_4
  have e_1843_5 := arithEq_of_rows h (row := 1843) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_5
  simp only [← c112500, ← c112501, k2, ← c112502] at e_1843_5
  have hr := e_1843_5
  simp only [e_1843_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 31) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8399)) := by
  have c112503 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112504 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112505 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112506 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112507 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112508 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_6 := arithEq_of_rows h (row := 1843) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_6
  simp only [← c112503, ← c112504, ← c112505] at e_1843_6
  have e_1843_7 := arithEq_of_rows h (row := 1843) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_7
  simp only [← c112506, ← c112507, k2, ← c112508] at e_1843_7
  have hr := e_1843_7
  simp only [e_1843_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 39) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8400)) := by
  have c112509 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112510 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112511 := (publicBatchWrapper86_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112512 := (publicBatchWrapper86_copies3516 a h).1
  have c112513 := (publicBatchWrapper86_copies3516 a h).2.1
  have c112514 := (publicBatchWrapper86_copies3516 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_8 := arithEq_of_rows h (row := 1843) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_8
  simp only [← c112509, ← c112510, ← c112511] at e_1843_8
  have e_1843_9 := arithEq_of_rows h (row := 1843) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_9
  simp only [← c112512, ← c112513, k2, ← c112514] at e_1843_9
  have hr := e_1843_9
  simp only [e_1843_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 47) = bselect (a (.wire 41 71)) (a (.virt 200)) (a (.virt 8401)) := by
  have c112515 := (publicBatchWrapper86_copies3516 a h).2.2.2.1
  have c112516 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.1
  have c112517 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.1
  have c112518 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.1
  have c112519 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.1
  have c112520 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_10 := arithEq_of_rows h (row := 1843) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_10
  simp only [← c112515, ← c112516, ← c112517] at e_1843_10
  have e_1843_11 := arithEq_of_rows h (row := 1843) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_11
  simp only [← c112518, ← c112519, k2, ← c112520] at e_1843_11
  have hr := e_1843_11
  simp only [e_1843_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 55) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8570)) := by
  have c112521 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.1
  have c112522 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112523 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112524 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112525 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112526 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_12 := arithEq_of_rows h (row := 1843) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_12
  simp only [← c112521, ← c112522, ← c112523] at e_1843_12
  have e_1843_13 := arithEq_of_rows h (row := 1843) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_13
  simp only [← c112524, ← c112525, k2, ← c112526] at e_1843_13
  have hr := e_1843_13
  simp only [e_1843_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 63) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8571)) := by
  have c112527 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112528 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112529 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112530 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112531 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112532 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_14 := arithEq_of_rows h (row := 1843) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_14
  simp only [← c112527, ← c112528, ← c112529] at e_1843_14
  have e_1843_15 := arithEq_of_rows h (row := 1843) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_15
  simp only [← c112530, ← c112531, k2, ← c112532] at e_1843_15
  have hr := e_1843_15
  simp only [e_1843_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 71) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8572)) := by
  have c112533 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112534 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112535 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112536 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112537 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112538 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_16 := arithEq_of_rows h (row := 1843) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_16
  simp only [← c112533, ← c112534, ← c112535] at e_1843_16
  have e_1843_17 := arithEq_of_rows h (row := 1843) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_17
  simp only [← c112536, ← c112537, k2, ← c112538] at e_1843_17
  have hr := e_1843_17
  simp only [e_1843_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1843 79) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8573)) := by
  have c112539 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112540 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112541 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112542 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112543 := (publicBatchWrapper86_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112544 := (publicBatchWrapper86_copies3517 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1843_18 := arithEq_of_rows h (row := 1843) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_18
  simp only [← c112539, ← c112540, ← c112541] at e_1843_18
  have e_1843_19 := arithEq_of_rows h (row := 1843) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_19
  simp only [← c112542, ← c112543, k2, ← c112544] at e_1843_19
  have hr := e_1843_19
  simp only [e_1843_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 7) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8574)) := by
  have c112545 := (publicBatchWrapper86_copies3517 a h).2.1
  have c112546 := (publicBatchWrapper86_copies3517 a h).2.2.1
  have c112547 := (publicBatchWrapper86_copies3517 a h).2.2.2.1
  have c112548 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.1
  have c112549 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.1
  have c112550 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_0 := arithEq_of_rows h (row := 1844) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_0
  simp only [← c112545, ← c112546, ← c112547] at e_1844_0
  have e_1844_1 := arithEq_of_rows h (row := 1844) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_1
  simp only [← c112548, ← c112549, k2, ← c112550] at e_1844_1
  have hr := e_1844_1
  simp only [e_1844_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 15) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8575)) := by
  have c112551 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.1
  have c112552 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.1
  have c112553 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.1
  have c112554 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112555 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112556 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_2 := arithEq_of_rows h (row := 1844) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_2
  simp only [← c112551, ← c112552, ← c112553] at e_1844_2
  have e_1844_3 := arithEq_of_rows h (row := 1844) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_3
  simp only [← c112554, ← c112555, k2, ← c112556] at e_1844_3
  have hr := e_1844_3
  simp only [e_1844_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 23) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8576)) := by
  have c112557 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112558 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112559 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112560 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112561 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112562 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_4 := arithEq_of_rows h (row := 1844) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_4
  simp only [← c112557, ← c112558, ← c112559] at e_1844_4
  have e_1844_5 := arithEq_of_rows h (row := 1844) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_5
  simp only [← c112560, ← c112561, k2, ← c112562] at e_1844_5
  have hr := e_1844_5
  simp only [e_1844_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 31) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8577)) := by
  have c112563 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112564 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112565 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112566 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112567 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112568 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_6 := arithEq_of_rows h (row := 1844) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_6
  simp only [← c112563, ← c112564, ← c112565] at e_1844_6
  have e_1844_7 := arithEq_of_rows h (row := 1844) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_7
  simp only [← c112566, ← c112567, k2, ← c112568] at e_1844_7
  have hr := e_1844_7
  simp only [e_1844_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 39) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8578)) := by
  have c112569 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112570 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112571 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112572 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112573 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112574 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_8 := arithEq_of_rows h (row := 1844) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_8
  simp only [← c112569, ← c112570, ← c112571] at e_1844_8
  have e_1844_9 := arithEq_of_rows h (row := 1844) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_9
  simp only [← c112572, ← c112573, k2, ← c112574] at e_1844_9
  have hr := e_1844_9
  simp only [e_1844_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 47) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8579)) := by
  have c112575 := (publicBatchWrapper86_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112576 := (publicBatchWrapper86_copies3518 a h).1
  have c112577 := (publicBatchWrapper86_copies3518 a h).2.1
  have c112578 := (publicBatchWrapper86_copies3518 a h).2.2.1
  have c112579 := (publicBatchWrapper86_copies3518 a h).2.2.2.1
  have c112580 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_10 := arithEq_of_rows h (row := 1844) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_10
  simp only [← c112575, ← c112576, ← c112577] at e_1844_10
  have e_1844_11 := arithEq_of_rows h (row := 1844) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_11
  simp only [← c112578, ← c112579, k2, ← c112580] at e_1844_11
  have hr := e_1844_11
  simp only [e_1844_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 55) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8580)) := by
  have c112581 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.1
  have c112582 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.1
  have c112583 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.1
  have c112584 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.1
  have c112585 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.1
  have c112586 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_12 := arithEq_of_rows h (row := 1844) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_12
  simp only [← c112581, ← c112582, ← c112583] at e_1844_12
  have e_1844_13 := arithEq_of_rows h (row := 1844) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_13
  simp only [← c112584, ← c112585, k2, ← c112586] at e_1844_13
  have hr := e_1844_13
  simp only [e_1844_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 63) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8581)) := by
  have c112587 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112588 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112589 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112590 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112591 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112592 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_14 := arithEq_of_rows h (row := 1844) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_14
  simp only [← c112587, ← c112588, ← c112589] at e_1844_14
  have e_1844_15 := arithEq_of_rows h (row := 1844) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_15
  simp only [← c112590, ← c112591, k2, ← c112592] at e_1844_15
  have hr := e_1844_15
  simp only [e_1844_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 71) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8582)) := by
  have c112593 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112594 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112595 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112596 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112597 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112598 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_16 := arithEq_of_rows h (row := 1844) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_16
  simp only [← c112593, ← c112594, ← c112595] at e_1844_16
  have e_1844_17 := arithEq_of_rows h (row := 1844) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_17
  simp only [← c112596, ← c112597, k2, ← c112598] at e_1844_17
  have hr := e_1844_17
  simp only [e_1844_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1844 79) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8583)) := by
  have c112599 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112600 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112601 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112602 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112603 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112604 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1844_18 := arithEq_of_rows h (row := 1844) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_18
  simp only [← c112599, ← c112600, ← c112601] at e_1844_18
  have e_1844_19 := arithEq_of_rows h (row := 1844) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_19
  simp only [← c112602, ← c112603, k2, ← c112604] at e_1844_19
  have hr := e_1844_19
  simp only [e_1844_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 7) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8584)) := by
  have c112605 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112606 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112607 := (publicBatchWrapper86_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112608 := (publicBatchWrapper86_copies3519 a h).1
  have c112609 := (publicBatchWrapper86_copies3519 a h).2.1
  have c112610 := (publicBatchWrapper86_copies3519 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_0 := arithEq_of_rows h (row := 1845) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_0
  simp only [← c112605, ← c112606, ← c112607] at e_1845_0
  have e_1845_1 := arithEq_of_rows h (row := 1845) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_1
  simp only [← c112608, ← c112609, k2, ← c112610] at e_1845_1
  have hr := e_1845_1
  simp only [e_1845_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 15) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8585)) := by
  have c112611 := (publicBatchWrapper86_copies3519 a h).2.2.2.1
  have c112612 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.1
  have c112613 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.1
  have c112614 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.1
  have c112615 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.1
  have c112616 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_2 := arithEq_of_rows h (row := 1845) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_2
  simp only [← c112611, ← c112612, ← c112613] at e_1845_2
  have e_1845_3 := arithEq_of_rows h (row := 1845) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_3
  simp only [← c112614, ← c112615, k2, ← c112616] at e_1845_3
  have hr := e_1845_3
  simp only [e_1845_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 23) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8586)) := by
  have c112617 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.1
  have c112618 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112619 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112620 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112621 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112622 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_4 := arithEq_of_rows h (row := 1845) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_4
  simp only [← c112617, ← c112618, ← c112619] at e_1845_4
  have e_1845_5 := arithEq_of_rows h (row := 1845) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_5
  simp only [← c112620, ← c112621, k2, ← c112622] at e_1845_5
  have hr := e_1845_5
  simp only [e_1845_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 31) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8587)) := by
  have c112623 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112624 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112625 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112626 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112627 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112628 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_6 := arithEq_of_rows h (row := 1845) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_6
  simp only [← c112623, ← c112624, ← c112625] at e_1845_6
  have e_1845_7 := arithEq_of_rows h (row := 1845) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_7
  simp only [← c112626, ← c112627, k2, ← c112628] at e_1845_7
  have hr := e_1845_7
  simp only [e_1845_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 39) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8588)) := by
  have c112629 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112630 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112631 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112632 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112633 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112634 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_8 := arithEq_of_rows h (row := 1845) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_8
  simp only [← c112629, ← c112630, ← c112631] at e_1845_8
  have e_1845_9 := arithEq_of_rows h (row := 1845) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_9
  simp only [← c112632, ← c112633, k2, ← c112634] at e_1845_9
  have hr := e_1845_9
  simp only [e_1845_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 47) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8589)) := by
  have c112635 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112636 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112637 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112638 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112639 := (publicBatchWrapper86_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112640 := (publicBatchWrapper86_copies3520 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_10 := arithEq_of_rows h (row := 1845) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_10
  simp only [← c112635, ← c112636, ← c112637] at e_1845_10
  have e_1845_11 := arithEq_of_rows h (row := 1845) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_11
  simp only [← c112638, ← c112639, k2, ← c112640] at e_1845_11
  have hr := e_1845_11
  simp only [e_1845_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 55) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8590)) := by
  have c112641 := (publicBatchWrapper86_copies3520 a h).2.1
  have c112642 := (publicBatchWrapper86_copies3520 a h).2.2.1
  have c112643 := (publicBatchWrapper86_copies3520 a h).2.2.2.1
  have c112644 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.1
  have c112645 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.1
  have c112646 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_12 := arithEq_of_rows h (row := 1845) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_12
  simp only [← c112641, ← c112642, ← c112643] at e_1845_12
  have e_1845_13 := arithEq_of_rows h (row := 1845) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_13
  simp only [← c112644, ← c112645, k2, ← c112646] at e_1845_13
  have hr := e_1845_13
  simp only [e_1845_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 63) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8591)) := by
  have c112647 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.1
  have c112648 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.1
  have c112649 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.1
  have c112650 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112651 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112652 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_14 := arithEq_of_rows h (row := 1845) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_14
  simp only [← c112647, ← c112648, ← c112649] at e_1845_14
  have e_1845_15 := arithEq_of_rows h (row := 1845) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_15
  simp only [← c112650, ← c112651, k2, ← c112652] at e_1845_15
  have hr := e_1845_15
  simp only [e_1845_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 71) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8592)) := by
  have c112653 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112654 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112655 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112656 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112657 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112658 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_16 := arithEq_of_rows h (row := 1845) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_16
  simp only [← c112653, ← c112654, ← c112655] at e_1845_16
  have e_1845_17 := arithEq_of_rows h (row := 1845) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_17
  simp only [← c112656, ← c112657, k2, ← c112658] at e_1845_17
  have hr := e_1845_17
  simp only [e_1845_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1845 79) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8593)) := by
  have c112659 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112660 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112661 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112662 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112663 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112664 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1845_18 := arithEq_of_rows h (row := 1845) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_18
  simp only [← c112659, ← c112660, ← c112661] at e_1845_18
  have e_1845_19 := arithEq_of_rows h (row := 1845) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_19
  simp only [← c112662, ← c112663, k2, ← c112664] at e_1845_19
  have hr := e_1845_19
  simp only [e_1845_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 7) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8594)) := by
  have c112665 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112666 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112667 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112668 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112669 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112670 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_0 := arithEq_of_rows h (row := 1846) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_0
  simp only [← c112665, ← c112666, ← c112667] at e_1846_0
  have e_1846_1 := arithEq_of_rows h (row := 1846) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_1
  simp only [← c112668, ← c112669, k2, ← c112670] at e_1846_1
  have hr := e_1846_1
  simp only [e_1846_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 15) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8595)) := by
  have c112671 := (publicBatchWrapper86_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112672 := (publicBatchWrapper86_copies3521 a h).1
  have c112673 := (publicBatchWrapper86_copies3521 a h).2.1
  have c112674 := (publicBatchWrapper86_copies3521 a h).2.2.1
  have c112675 := (publicBatchWrapper86_copies3521 a h).2.2.2.1
  have c112676 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_2 := arithEq_of_rows h (row := 1846) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_2
  simp only [← c112671, ← c112672, ← c112673] at e_1846_2
  have e_1846_3 := arithEq_of_rows h (row := 1846) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_3
  simp only [← c112674, ← c112675, k2, ← c112676] at e_1846_3
  have hr := e_1846_3
  simp only [e_1846_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 23) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8596)) := by
  have c112677 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.1
  have c112678 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.1
  have c112679 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.1
  have c112680 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.1
  have c112681 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.1
  have c112682 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_4 := arithEq_of_rows h (row := 1846) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_4
  simp only [← c112677, ← c112678, ← c112679] at e_1846_4
  have e_1846_5 := arithEq_of_rows h (row := 1846) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_5
  simp only [← c112680, ← c112681, k2, ← c112682] at e_1846_5
  have hr := e_1846_5
  simp only [e_1846_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 31) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8597)) := by
  have c112683 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112684 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112685 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112686 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112687 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112688 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_6 := arithEq_of_rows h (row := 1846) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_6
  simp only [← c112683, ← c112684, ← c112685] at e_1846_6
  have e_1846_7 := arithEq_of_rows h (row := 1846) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_7
  simp only [← c112686, ← c112687, k2, ← c112688] at e_1846_7
  have hr := e_1846_7
  simp only [e_1846_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 39) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8598)) := by
  have c112689 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112690 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112691 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112692 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112693 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112694 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_8 := arithEq_of_rows h (row := 1846) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_8
  simp only [← c112689, ← c112690, ← c112691] at e_1846_8
  have e_1846_9 := arithEq_of_rows h (row := 1846) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_9
  simp only [← c112692, ← c112693, k2, ← c112694] at e_1846_9
  have hr := e_1846_9
  simp only [e_1846_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 47) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8599)) := by
  have c112695 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112696 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112697 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112698 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112699 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112700 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_10 := arithEq_of_rows h (row := 1846) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_10
  simp only [← c112695, ← c112696, ← c112697] at e_1846_10
  have e_1846_11 := arithEq_of_rows h (row := 1846) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_11
  simp only [← c112698, ← c112699, k2, ← c112700] at e_1846_11
  have hr := e_1846_11
  simp only [e_1846_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 55) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8600)) := by
  have c112701 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112702 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112703 := (publicBatchWrapper86_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112704 := (publicBatchWrapper86_copies3522 a h).1
  have c112705 := (publicBatchWrapper86_copies3522 a h).2.1
  have c112706 := (publicBatchWrapper86_copies3522 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_12 := arithEq_of_rows h (row := 1846) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_12
  simp only [← c112701, ← c112702, ← c112703] at e_1846_12
  have e_1846_13 := arithEq_of_rows h (row := 1846) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_13
  simp only [← c112704, ← c112705, k2, ← c112706] at e_1846_13
  have hr := e_1846_13
  simp only [e_1846_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 63) = bselect (a (.wire 42 35)) (a (.virt 200)) (a (.virt 8601)) := by
  have c112707 := (publicBatchWrapper86_copies3522 a h).2.2.2.1
  have c112708 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.1
  have c112709 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.1
  have c112710 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.1
  have c112711 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.1
  have c112712 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_14 := arithEq_of_rows h (row := 1846) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_14
  simp only [← c112707, ← c112708, ← c112709] at e_1846_14
  have e_1846_15 := arithEq_of_rows h (row := 1846) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_15
  simp only [← c112710, ← c112711, k2, ← c112712] at e_1846_15
  have hr := e_1846_15
  simp only [e_1846_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8770)) := by
  have c112713 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.1
  have c112714 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112715 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112716 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112717 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112718 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_16 := arithEq_of_rows h (row := 1846) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_16
  simp only [← c112713, ← c112714, ← c112715] at e_1846_16
  have e_1846_17 := arithEq_of_rows h (row := 1846) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_17
  simp only [← c112716, ← c112717, k2, ← c112718] at e_1846_17
  have hr := e_1846_17
  simp only [e_1846_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1846 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8771)) := by
  have c112719 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112720 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112721 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112722 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112723 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112724 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1846_18 := arithEq_of_rows h (row := 1846) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_18
  simp only [← c112719, ← c112720, ← c112721] at e_1846_18
  have e_1846_19 := arithEq_of_rows h (row := 1846) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_19
  simp only [← c112722, ← c112723, k2, ← c112724] at e_1846_19
  have hr := e_1846_19
  simp only [e_1846_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8772)) := by
  have c112725 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112726 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112727 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112728 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112729 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112730 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_0 := arithEq_of_rows h (row := 1847) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_0
  simp only [← c112725, ← c112726, ← c112727] at e_1847_0
  have e_1847_1 := arithEq_of_rows h (row := 1847) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_1
  simp only [← c112728, ← c112729, k2, ← c112730] at e_1847_1
  have hr := e_1847_1
  simp only [e_1847_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8773)) := by
  have c112731 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112732 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112733 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112734 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112735 := (publicBatchWrapper86_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112736 := (publicBatchWrapper86_copies3523 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_2 := arithEq_of_rows h (row := 1847) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_2
  simp only [← c112731, ← c112732, ← c112733] at e_1847_2
  have e_1847_3 := arithEq_of_rows h (row := 1847) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_3
  simp only [← c112734, ← c112735, k2, ← c112736] at e_1847_3
  have hr := e_1847_3
  simp only [e_1847_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8774)) := by
  have c112737 := (publicBatchWrapper86_copies3523 a h).2.1
  have c112738 := (publicBatchWrapper86_copies3523 a h).2.2.1
  have c112739 := (publicBatchWrapper86_copies3523 a h).2.2.2.1
  have c112740 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.1
  have c112741 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.1
  have c112742 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_4 := arithEq_of_rows h (row := 1847) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_4
  simp only [← c112737, ← c112738, ← c112739] at e_1847_4
  have e_1847_5 := arithEq_of_rows h (row := 1847) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_5
  simp only [← c112740, ← c112741, k2, ← c112742] at e_1847_5
  have hr := e_1847_5
  simp only [e_1847_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8775)) := by
  have c112743 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.1
  have c112744 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.1
  have c112745 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.1
  have c112746 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112747 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112748 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_6 := arithEq_of_rows h (row := 1847) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_6
  simp only [← c112743, ← c112744, ← c112745] at e_1847_6
  have e_1847_7 := arithEq_of_rows h (row := 1847) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_7
  simp only [← c112746, ← c112747, k2, ← c112748] at e_1847_7
  have hr := e_1847_7
  simp only [e_1847_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8776)) := by
  have c112749 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112750 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112751 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112752 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112753 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112754 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_8 := arithEq_of_rows h (row := 1847) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_8
  simp only [← c112749, ← c112750, ← c112751] at e_1847_8
  have e_1847_9 := arithEq_of_rows h (row := 1847) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_9
  simp only [← c112752, ← c112753, k2, ← c112754] at e_1847_9
  have hr := e_1847_9
  simp only [e_1847_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8777)) := by
  have c112755 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112756 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112757 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112758 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112759 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112760 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_10 := arithEq_of_rows h (row := 1847) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_10
  simp only [← c112755, ← c112756, ← c112757] at e_1847_10
  have e_1847_11 := arithEq_of_rows h (row := 1847) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_11
  simp only [← c112758, ← c112759, k2, ← c112760] at e_1847_11
  have hr := e_1847_11
  simp only [e_1847_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8778)) := by
  have c112761 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112762 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112763 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112764 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112765 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112766 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_12 := arithEq_of_rows h (row := 1847) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_12
  simp only [← c112761, ← c112762, ← c112763] at e_1847_12
  have e_1847_13 := arithEq_of_rows h (row := 1847) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_13
  simp only [← c112764, ← c112765, k2, ← c112766] at e_1847_13
  have hr := e_1847_13
  simp only [e_1847_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8779)) := by
  have c112767 := (publicBatchWrapper86_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112768 := (publicBatchWrapper86_copies3524 a h).1
  have c112769 := (publicBatchWrapper86_copies3524 a h).2.1
  have c112770 := (publicBatchWrapper86_copies3524 a h).2.2.1
  have c112771 := (publicBatchWrapper86_copies3524 a h).2.2.2.1
  have c112772 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_14 := arithEq_of_rows h (row := 1847) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_14
  simp only [← c112767, ← c112768, ← c112769] at e_1847_14
  have e_1847_15 := arithEq_of_rows h (row := 1847) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_15
  simp only [← c112770, ← c112771, k2, ← c112772] at e_1847_15
  have hr := e_1847_15
  simp only [e_1847_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8780)) := by
  have c112773 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.1
  have c112774 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.1
  have c112775 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.1
  have c112776 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.1
  have c112777 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.1
  have c112778 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_16 := arithEq_of_rows h (row := 1847) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_16
  simp only [← c112773, ← c112774, ← c112775] at e_1847_16
  have e_1847_17 := arithEq_of_rows h (row := 1847) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_17
  simp only [← c112776, ← c112777, k2, ← c112778] at e_1847_17
  have hr := e_1847_17
  simp only [e_1847_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1847 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8781)) := by
  have c112779 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112780 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112781 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112782 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112783 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112784 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1847_18 := arithEq_of_rows h (row := 1847) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_18
  simp only [← c112779, ← c112780, ← c112781] at e_1847_18
  have e_1847_19 := arithEq_of_rows h (row := 1847) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_19
  simp only [← c112782, ← c112783, k2, ← c112784] at e_1847_19
  have hr := e_1847_19
  simp only [e_1847_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8782)) := by
  have c112785 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112786 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112787 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112788 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112789 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112790 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_0 := arithEq_of_rows h (row := 1848) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_0
  simp only [← c112785, ← c112786, ← c112787] at e_1848_0
  have e_1848_1 := arithEq_of_rows h (row := 1848) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_1
  simp only [← c112788, ← c112789, k2, ← c112790] at e_1848_1
  have hr := e_1848_1
  simp only [e_1848_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8783)) := by
  have c112791 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112792 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112793 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112794 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112795 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112796 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_2 := arithEq_of_rows h (row := 1848) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_2
  simp only [← c112791, ← c112792, ← c112793] at e_1848_2
  have e_1848_3 := arithEq_of_rows h (row := 1848) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_3
  simp only [← c112794, ← c112795, k2, ← c112796] at e_1848_3
  have hr := e_1848_3
  simp only [e_1848_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8784)) := by
  have c112797 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112798 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112799 := (publicBatchWrapper86_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112800 := (publicBatchWrapper86_copies3525 a h).1
  have c112801 := (publicBatchWrapper86_copies3525 a h).2.1
  have c112802 := (publicBatchWrapper86_copies3525 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_4 := arithEq_of_rows h (row := 1848) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_4
  simp only [← c112797, ← c112798, ← c112799] at e_1848_4
  have e_1848_5 := arithEq_of_rows h (row := 1848) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_5
  simp only [← c112800, ← c112801, k2, ← c112802] at e_1848_5
  have hr := e_1848_5
  simp only [e_1848_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8785)) := by
  have c112803 := (publicBatchWrapper86_copies3525 a h).2.2.2.1
  have c112804 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.1
  have c112805 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.1
  have c112806 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.1
  have c112807 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.1
  have c112808 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_6 := arithEq_of_rows h (row := 1848) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_6
  simp only [← c112803, ← c112804, ← c112805] at e_1848_6
  have e_1848_7 := arithEq_of_rows h (row := 1848) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_7
  simp only [← c112806, ← c112807, k2, ← c112808] at e_1848_7
  have hr := e_1848_7
  simp only [e_1848_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8786)) := by
  have c112809 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.1
  have c112810 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112811 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112812 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112813 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112814 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_8 := arithEq_of_rows h (row := 1848) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_8
  simp only [← c112809, ← c112810, ← c112811] at e_1848_8
  have e_1848_9 := arithEq_of_rows h (row := 1848) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_9
  simp only [← c112812, ← c112813, k2, ← c112814] at e_1848_9
  have hr := e_1848_9
  simp only [e_1848_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8787)) := by
  have c112815 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112816 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112817 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112818 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112819 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112820 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_10 := arithEq_of_rows h (row := 1848) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_10
  simp only [← c112815, ← c112816, ← c112817] at e_1848_10
  have e_1848_11 := arithEq_of_rows h (row := 1848) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_11
  simp only [← c112818, ← c112819, k2, ← c112820] at e_1848_11
  have hr := e_1848_11
  simp only [e_1848_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8788)) := by
  have c112821 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112822 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112823 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112824 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112825 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112826 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_12 := arithEq_of_rows h (row := 1848) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_12
  simp only [← c112821, ← c112822, ← c112823] at e_1848_12
  have e_1848_13 := arithEq_of_rows h (row := 1848) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_13
  simp only [← c112824, ← c112825, k2, ← c112826] at e_1848_13
  have hr := e_1848_13
  simp only [e_1848_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8789)) := by
  have c112827 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112828 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112829 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112830 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112831 := (publicBatchWrapper86_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112832 := (publicBatchWrapper86_copies3526 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_14 := arithEq_of_rows h (row := 1848) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_14
  simp only [← c112827, ← c112828, ← c112829] at e_1848_14
  have e_1848_15 := arithEq_of_rows h (row := 1848) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_15
  simp only [← c112830, ← c112831, k2, ← c112832] at e_1848_15
  have hr := e_1848_15
  simp only [e_1848_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8790)) := by
  have c112833 := (publicBatchWrapper86_copies3526 a h).2.1
  have c112834 := (publicBatchWrapper86_copies3526 a h).2.2.1
  have c112835 := (publicBatchWrapper86_copies3526 a h).2.2.2.1
  have c112836 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.1
  have c112837 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.1
  have c112838 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_16 := arithEq_of_rows h (row := 1848) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_16
  simp only [← c112833, ← c112834, ← c112835] at e_1848_16
  have e_1848_17 := arithEq_of_rows h (row := 1848) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_17
  simp only [← c112836, ← c112837, k2, ← c112838] at e_1848_17
  have hr := e_1848_17
  simp only [e_1848_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1848 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8791)) := by
  have c112839 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.1
  have c112840 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.1
  have c112841 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.1
  have c112842 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112843 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112844 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1848_18 := arithEq_of_rows h (row := 1848) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_18
  simp only [← c112839, ← c112840, ← c112841] at e_1848_18
  have e_1848_19 := arithEq_of_rows h (row := 1848) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_19
  simp only [← c112842, ← c112843, k2, ← c112844] at e_1848_19
  have hr := e_1848_19
  simp only [e_1848_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 7) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8792)) := by
  have c112845 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112846 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112847 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112848 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112849 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112850 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_0 := arithEq_of_rows h (row := 1849) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_0
  simp only [← c112845, ← c112846, ← c112847] at e_1849_0
  have e_1849_1 := arithEq_of_rows h (row := 1849) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_1
  simp only [← c112848, ← c112849, k2, ← c112850] at e_1849_1
  have hr := e_1849_1
  simp only [e_1849_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 15) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8793)) := by
  have c112851 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112852 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112853 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112854 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112855 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112856 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_2 := arithEq_of_rows h (row := 1849) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_2
  simp only [← c112851, ← c112852, ← c112853] at e_1849_2
  have e_1849_3 := arithEq_of_rows h (row := 1849) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_3
  simp only [← c112854, ← c112855, k2, ← c112856] at e_1849_3
  have hr := e_1849_3
  simp only [e_1849_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 23) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8794)) := by
  have c112857 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112858 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112859 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112860 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112861 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112862 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_4 := arithEq_of_rows h (row := 1849) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_4
  simp only [← c112857, ← c112858, ← c112859] at e_1849_4
  have e_1849_5 := arithEq_of_rows h (row := 1849) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_5
  simp only [← c112860, ← c112861, k2, ← c112862] at e_1849_5
  have hr := e_1849_5
  simp only [e_1849_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 31) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8795)) := by
  have c112863 := (publicBatchWrapper86_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112864 := (publicBatchWrapper86_copies3527 a h).1
  have c112865 := (publicBatchWrapper86_copies3527 a h).2.1
  have c112866 := (publicBatchWrapper86_copies3527 a h).2.2.1
  have c112867 := (publicBatchWrapper86_copies3527 a h).2.2.2.1
  have c112868 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_6 := arithEq_of_rows h (row := 1849) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_6
  simp only [← c112863, ← c112864, ← c112865] at e_1849_6
  have e_1849_7 := arithEq_of_rows h (row := 1849) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_7
  simp only [← c112866, ← c112867, k2, ← c112868] at e_1849_7
  have hr := e_1849_7
  simp only [e_1849_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 39) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8796)) := by
  have c112869 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.1
  have c112870 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.1
  have c112871 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.1
  have c112872 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.1
  have c112873 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.1
  have c112874 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_8 := arithEq_of_rows h (row := 1849) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_8
  simp only [← c112869, ← c112870, ← c112871] at e_1849_8
  have e_1849_9 := arithEq_of_rows h (row := 1849) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_9
  simp only [← c112872, ← c112873, k2, ← c112874] at e_1849_9
  have hr := e_1849_9
  simp only [e_1849_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 47) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8797)) := by
  have c112875 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112876 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112877 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112878 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112879 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112880 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_10 := arithEq_of_rows h (row := 1849) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_10
  simp only [← c112875, ← c112876, ← c112877] at e_1849_10
  have e_1849_11 := arithEq_of_rows h (row := 1849) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_11
  simp only [← c112878, ← c112879, k2, ← c112880] at e_1849_11
  have hr := e_1849_11
  simp only [e_1849_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 55) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8798)) := by
  have c112881 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112882 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112883 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112884 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112885 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112886 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_12 := arithEq_of_rows h (row := 1849) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_12
  simp only [← c112881, ← c112882, ← c112883] at e_1849_12
  have e_1849_13 := arithEq_of_rows h (row := 1849) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_13
  simp only [← c112884, ← c112885, k2, ← c112886] at e_1849_13
  have hr := e_1849_13
  simp only [e_1849_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 63) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8799)) := by
  have c112887 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112888 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112889 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112890 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112891 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112892 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_14 := arithEq_of_rows h (row := 1849) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_14
  simp only [← c112887, ← c112888, ← c112889] at e_1849_14
  have e_1849_15 := arithEq_of_rows h (row := 1849) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_15
  simp only [← c112890, ← c112891, k2, ← c112892] at e_1849_15
  have hr := e_1849_15
  simp only [e_1849_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 71) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8800)) := by
  have c112893 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112894 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112895 := (publicBatchWrapper86_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112896 := (publicBatchWrapper86_copies3528 a h).1
  have c112897 := (publicBatchWrapper86_copies3528 a h).2.1
  have c112898 := (publicBatchWrapper86_copies3528 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_16 := arithEq_of_rows h (row := 1849) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_16
  simp only [← c112893, ← c112894, ← c112895] at e_1849_16
  have e_1849_17 := arithEq_of_rows h (row := 1849) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_17
  simp only [← c112896, ← c112897, k2, ← c112898] at e_1849_17
  have hr := e_1849_17
  simp only [e_1849_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1849 79) = bselect (a (.wire 42 79)) (a (.virt 200)) (a (.virt 8801)) := by
  have c112899 := (publicBatchWrapper86_copies3528 a h).2.2.2.1
  have c112900 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.1
  have c112901 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.1
  have c112902 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.1
  have c112903 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.1
  have c112904 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1849_18 := arithEq_of_rows h (row := 1849) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_18
  simp only [← c112899, ← c112900, ← c112901] at e_1849_18
  have e_1849_19 := arithEq_of_rows h (row := 1849) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_19
  simp only [← c112902, ← c112903, k2, ← c112904] at e_1849_19
  have hr := e_1849_19
  simp only [e_1849_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8970)) := by
  have c112905 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.1
  have c112906 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112907 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112908 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112909 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112910 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_0 := arithEq_of_rows h (row := 1850) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_0
  simp only [← c112905, ← c112906, ← c112907] at e_1850_0
  have e_1850_1 := arithEq_of_rows h (row := 1850) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_1
  simp only [← c112908, ← c112909, k2, ← c112910] at e_1850_1
  have hr := e_1850_1
  simp only [e_1850_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8971)) := by
  have c112911 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112912 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112913 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112914 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112915 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112916 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_2 := arithEq_of_rows h (row := 1850) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_2
  simp only [← c112911, ← c112912, ← c112913] at e_1850_2
  have e_1850_3 := arithEq_of_rows h (row := 1850) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_3
  simp only [← c112914, ← c112915, k2, ← c112916] at e_1850_3
  have hr := e_1850_3
  simp only [e_1850_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8972)) := by
  have c112917 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112918 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112919 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112920 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112921 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112922 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_4 := arithEq_of_rows h (row := 1850) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_4
  simp only [← c112917, ← c112918, ← c112919] at e_1850_4
  have e_1850_5 := arithEq_of_rows h (row := 1850) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_5
  simp only [← c112920, ← c112921, k2, ← c112922] at e_1850_5
  have hr := e_1850_5
  simp only [e_1850_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8973)) := by
  have c112923 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112924 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112925 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112926 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112927 := (publicBatchWrapper86_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112928 := (publicBatchWrapper86_copies3529 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_6 := arithEq_of_rows h (row := 1850) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_6
  simp only [← c112923, ← c112924, ← c112925] at e_1850_6
  have e_1850_7 := arithEq_of_rows h (row := 1850) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_7
  simp only [← c112926, ← c112927, k2, ← c112928] at e_1850_7
  have hr := e_1850_7
  simp only [e_1850_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8974)) := by
  have c112929 := (publicBatchWrapper86_copies3529 a h).2.1
  have c112930 := (publicBatchWrapper86_copies3529 a h).2.2.1
  have c112931 := (publicBatchWrapper86_copies3529 a h).2.2.2.1
  have c112932 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.1
  have c112933 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.1
  have c112934 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_8 := arithEq_of_rows h (row := 1850) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_8
  simp only [← c112929, ← c112930, ← c112931] at e_1850_8
  have e_1850_9 := arithEq_of_rows h (row := 1850) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_9
  simp only [← c112932, ← c112933, k2, ← c112934] at e_1850_9
  have hr := e_1850_9
  simp only [e_1850_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8975)) := by
  have c112935 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.1
  have c112936 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.1
  have c112937 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.1
  have c112938 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112939 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112940 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_10 := arithEq_of_rows h (row := 1850) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_10
  simp only [← c112935, ← c112936, ← c112937] at e_1850_10
  have e_1850_11 := arithEq_of_rows h (row := 1850) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_11
  simp only [← c112938, ← c112939, k2, ← c112940] at e_1850_11
  have hr := e_1850_11
  simp only [e_1850_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8976)) := by
  have c112941 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112942 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112943 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112944 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112945 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112946 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_12 := arithEq_of_rows h (row := 1850) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_12
  simp only [← c112941, ← c112942, ← c112943] at e_1850_12
  have e_1850_13 := arithEq_of_rows h (row := 1850) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_13
  simp only [← c112944, ← c112945, k2, ← c112946] at e_1850_13
  have hr := e_1850_13
  simp only [e_1850_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8977)) := by
  have c112947 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112948 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112949 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112950 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112951 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112952 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_14 := arithEq_of_rows h (row := 1850) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_14
  simp only [← c112947, ← c112948, ← c112949] at e_1850_14
  have e_1850_15 := arithEq_of_rows h (row := 1850) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_15
  simp only [← c112950, ← c112951, k2, ← c112952] at e_1850_15
  have hr := e_1850_15
  simp only [e_1850_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8978)) := by
  have c112953 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112954 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112955 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112956 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112957 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112958 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_16 := arithEq_of_rows h (row := 1850) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_16
  simp only [← c112953, ← c112954, ← c112955] at e_1850_16
  have e_1850_17 := arithEq_of_rows h (row := 1850) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_17
  simp only [← c112956, ← c112957, k2, ← c112958] at e_1850_17
  have hr := e_1850_17
  simp only [e_1850_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1850 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8979)) := by
  have c112959 := (publicBatchWrapper86_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112960 := (publicBatchWrapper86_copies3530 a h).1
  have c112961 := (publicBatchWrapper86_copies3530 a h).2.1
  have c112962 := (publicBatchWrapper86_copies3530 a h).2.2.1
  have c112963 := (publicBatchWrapper86_copies3530 a h).2.2.2.1
  have c112964 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1850_18 := arithEq_of_rows h (row := 1850) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_18
  simp only [← c112959, ← c112960, ← c112961] at e_1850_18
  have e_1850_19 := arithEq_of_rows h (row := 1850) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_19
  simp only [← c112962, ← c112963, k2, ← c112964] at e_1850_19
  have hr := e_1850_19
  simp only [e_1850_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8980)) := by
  have c112965 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.1
  have c112966 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.1
  have c112967 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.1
  have c112968 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.1
  have c112969 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.1
  have c112970 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_0 := arithEq_of_rows h (row := 1851) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_0
  simp only [← c112965, ← c112966, ← c112967] at e_1851_0
  have e_1851_1 := arithEq_of_rows h (row := 1851) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_1
  simp only [← c112968, ← c112969, k2, ← c112970] at e_1851_1
  have hr := e_1851_1
  simp only [e_1851_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8981)) := by
  have c112971 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112972 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112973 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112974 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112975 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112976 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_2 := arithEq_of_rows h (row := 1851) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_2
  simp only [← c112971, ← c112972, ← c112973] at e_1851_2
  have e_1851_3 := arithEq_of_rows h (row := 1851) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_3
  simp only [← c112974, ← c112975, k2, ← c112976] at e_1851_3
  have hr := e_1851_3
  simp only [e_1851_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8982)) := by
  have c112977 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112978 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112979 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112980 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112981 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112982 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_4 := arithEq_of_rows h (row := 1851) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_4
  simp only [← c112977, ← c112978, ← c112979] at e_1851_4
  have e_1851_5 := arithEq_of_rows h (row := 1851) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_5
  simp only [← c112980, ← c112981, k2, ← c112982] at e_1851_5
  have hr := e_1851_5
  simp only [e_1851_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8983)) := by
  have c112983 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112984 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112985 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112986 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112987 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112988 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_6 := arithEq_of_rows h (row := 1851) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_6
  simp only [← c112983, ← c112984, ← c112985] at e_1851_6
  have e_1851_7 := arithEq_of_rows h (row := 1851) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_7
  simp only [← c112986, ← c112987, k2, ← c112988] at e_1851_7
  have hr := e_1851_7
  simp only [e_1851_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8984)) := by
  have c112989 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112990 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112991 := (publicBatchWrapper86_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112992 := (publicBatchWrapper86_copies3531 a h).1
  have c112993 := (publicBatchWrapper86_copies3531 a h).2.1
  have c112994 := (publicBatchWrapper86_copies3531 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_8 := arithEq_of_rows h (row := 1851) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_8
  simp only [← c112989, ← c112990, ← c112991] at e_1851_8
  have e_1851_9 := arithEq_of_rows h (row := 1851) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_9
  simp only [← c112992, ← c112993, k2, ← c112994] at e_1851_9
  have hr := e_1851_9
  simp only [e_1851_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8985)) := by
  have c112995 := (publicBatchWrapper86_copies3531 a h).2.2.2.1
  have c112996 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.1
  have c112997 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.1
  have c112998 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.1
  have c112999 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.1
  have c113000 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_10 := arithEq_of_rows h (row := 1851) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_10
  simp only [← c112995, ← c112996, ← c112997] at e_1851_10
  have e_1851_11 := arithEq_of_rows h (row := 1851) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_11
  simp only [← c112998, ← c112999, k2, ← c113000] at e_1851_11
  have hr := e_1851_11
  simp only [e_1851_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8986)) := by
  have c113001 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.1
  have c113002 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113003 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113004 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113005 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113006 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_12 := arithEq_of_rows h (row := 1851) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_12
  simp only [← c113001, ← c113002, ← c113003] at e_1851_12
  have e_1851_13 := arithEq_of_rows h (row := 1851) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_13
  simp only [← c113004, ← c113005, k2, ← c113006] at e_1851_13
  have hr := e_1851_13
  simp only [e_1851_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8987)) := by
  have c113007 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113008 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113009 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113010 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113011 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113012 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_14 := arithEq_of_rows h (row := 1851) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_14
  simp only [← c113007, ← c113008, ← c113009] at e_1851_14
  have e_1851_15 := arithEq_of_rows h (row := 1851) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_15
  simp only [← c113010, ← c113011, k2, ← c113012] at e_1851_15
  have hr := e_1851_15
  simp only [e_1851_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8988)) := by
  have c113013 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113014 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113015 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113016 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113017 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113018 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_16 := arithEq_of_rows h (row := 1851) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_16
  simp only [← c113013, ← c113014, ← c113015] at e_1851_16
  have e_1851_17 := arithEq_of_rows h (row := 1851) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_17
  simp only [← c113016, ← c113017, k2, ← c113018] at e_1851_17
  have hr := e_1851_17
  simp only [e_1851_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1851 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8989)) := by
  have c113019 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113020 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113021 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113022 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113023 := (publicBatchWrapper86_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113024 := (publicBatchWrapper86_copies3532 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1851_18 := arithEq_of_rows h (row := 1851) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_18
  simp only [← c113019, ← c113020, ← c113021] at e_1851_18
  have e_1851_19 := arithEq_of_rows h (row := 1851) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_19
  simp only [← c113022, ← c113023, k2, ← c113024] at e_1851_19
  have hr := e_1851_19
  simp only [e_1851_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8990)) := by
  have c113025 := (publicBatchWrapper86_copies3532 a h).2.1
  have c113026 := (publicBatchWrapper86_copies3532 a h).2.2.1
  have c113027 := (publicBatchWrapper86_copies3532 a h).2.2.2.1
  have c113028 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.1
  have c113029 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.1
  have c113030 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_0 := arithEq_of_rows h (row := 1852) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_0
  simp only [← c113025, ← c113026, ← c113027] at e_1852_0
  have e_1852_1 := arithEq_of_rows h (row := 1852) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_1
  simp only [← c113028, ← c113029, k2, ← c113030] at e_1852_1
  have hr := e_1852_1
  simp only [e_1852_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8991)) := by
  have c113031 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.1
  have c113032 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.1
  have c113033 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.1
  have c113034 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113035 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113036 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_2 := arithEq_of_rows h (row := 1852) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_2
  simp only [← c113031, ← c113032, ← c113033] at e_1852_2
  have e_1852_3 := arithEq_of_rows h (row := 1852) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_3
  simp only [← c113034, ← c113035, k2, ← c113036] at e_1852_3
  have hr := e_1852_3
  simp only [e_1852_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 23) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8992)) := by
  have c113037 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113038 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113039 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113040 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113041 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113042 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_4 := arithEq_of_rows h (row := 1852) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_4
  simp only [← c113037, ← c113038, ← c113039] at e_1852_4
  have e_1852_5 := arithEq_of_rows h (row := 1852) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_5
  simp only [← c113040, ← c113041, k2, ← c113042] at e_1852_5
  have hr := e_1852_5
  simp only [e_1852_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 31) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8993)) := by
  have c113043 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113044 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113045 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113046 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113047 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113048 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_6 := arithEq_of_rows h (row := 1852) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_6
  simp only [← c113043, ← c113044, ← c113045] at e_1852_6
  have e_1852_7 := arithEq_of_rows h (row := 1852) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_7
  simp only [← c113046, ← c113047, k2, ← c113048] at e_1852_7
  have hr := e_1852_7
  simp only [e_1852_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 39) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8994)) := by
  have c113049 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113050 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113051 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113052 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113053 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113054 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_8 := arithEq_of_rows h (row := 1852) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_8
  simp only [← c113049, ← c113050, ← c113051] at e_1852_8
  have e_1852_9 := arithEq_of_rows h (row := 1852) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_9
  simp only [← c113052, ← c113053, k2, ← c113054] at e_1852_9
  have hr := e_1852_9
  simp only [e_1852_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 47) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8995)) := by
  have c113055 := (publicBatchWrapper86_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113056 := (publicBatchWrapper86_copies3533 a h).1
  have c113057 := (publicBatchWrapper86_copies3533 a h).2.1
  have c113058 := (publicBatchWrapper86_copies3533 a h).2.2.1
  have c113059 := (publicBatchWrapper86_copies3533 a h).2.2.2.1
  have c113060 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_10 := arithEq_of_rows h (row := 1852) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_10
  simp only [← c113055, ← c113056, ← c113057] at e_1852_10
  have e_1852_11 := arithEq_of_rows h (row := 1852) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_11
  simp only [← c113058, ← c113059, k2, ← c113060] at e_1852_11
  have hr := e_1852_11
  simp only [e_1852_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 55) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8996)) := by
  have c113061 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.1
  have c113062 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.1
  have c113063 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.1
  have c113064 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.1
  have c113065 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.1
  have c113066 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_12 := arithEq_of_rows h (row := 1852) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_12
  simp only [← c113061, ← c113062, ← c113063] at e_1852_12
  have e_1852_13 := arithEq_of_rows h (row := 1852) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_13
  simp only [← c113064, ← c113065, k2, ← c113066] at e_1852_13
  have hr := e_1852_13
  simp only [e_1852_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 63) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8997)) := by
  have c113067 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113068 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113069 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113070 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113071 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113072 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_14 := arithEq_of_rows h (row := 1852) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_14
  simp only [← c113067, ← c113068, ← c113069] at e_1852_14
  have e_1852_15 := arithEq_of_rows h (row := 1852) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_15
  simp only [← c113070, ← c113071, k2, ← c113072] at e_1852_15
  have hr := e_1852_15
  simp only [e_1852_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 71) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8998)) := by
  have c113073 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113074 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113075 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113076 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113077 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113078 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_16 := arithEq_of_rows h (row := 1852) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_16
  simp only [← c113073, ← c113074, ← c113075] at e_1852_16
  have e_1852_17 := arithEq_of_rows h (row := 1852) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_17
  simp only [← c113076, ← c113077, k2, ← c113078] at e_1852_17
  have hr := e_1852_17
  simp only [e_1852_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1852 79) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 8999)) := by
  have c113079 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113080 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113081 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113082 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113083 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113084 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1852_18 := arithEq_of_rows h (row := 1852) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_18
  simp only [← c113079, ← c113080, ← c113081] at e_1852_18
  have e_1852_19 := arithEq_of_rows h (row := 1852) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_19
  simp only [← c113082, ← c113083, k2, ← c113084] at e_1852_19
  have hr := e_1852_19
  simp only [e_1852_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 7) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 9000)) := by
  have c113085 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113086 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113087 := (publicBatchWrapper86_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113088 := (publicBatchWrapper86_copies3534 a h).1
  have c113089 := (publicBatchWrapper86_copies3534 a h).2.1
  have c113090 := (publicBatchWrapper86_copies3534 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_0 := arithEq_of_rows h (row := 1853) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_0
  simp only [← c113085, ← c113086, ← c113087] at e_1853_0
  have e_1853_1 := arithEq_of_rows h (row := 1853) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_1
  simp only [← c113088, ← c113089, k2, ← c113090] at e_1853_1
  have hr := e_1853_1
  simp only [e_1853_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 15) = bselect (a (.wire 44 43)) (a (.virt 200)) (a (.virt 9001)) := by
  have c113091 := (publicBatchWrapper86_copies3534 a h).2.2.2.1
  have c113092 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.1
  have c113093 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.1
  have c113094 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.1
  have c113095 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.1
  have c113096 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_2 := arithEq_of_rows h (row := 1853) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_2
  simp only [← c113091, ← c113092, ← c113093] at e_1853_2
  have e_1853_3 := arithEq_of_rows h (row := 1853) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_3
  simp only [← c113094, ← c113095, k2, ← c113096] at e_1853_3
  have hr := e_1853_3
  simp only [e_1853_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9170)) := by
  have c113097 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.1
  have c113098 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113099 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113100 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113101 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113102 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_4 := arithEq_of_rows h (row := 1853) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_4
  simp only [← c113097, ← c113098, ← c113099] at e_1853_4
  have e_1853_5 := arithEq_of_rows h (row := 1853) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_5
  simp only [← c113100, ← c113101, k2, ← c113102] at e_1853_5
  have hr := e_1853_5
  simp only [e_1853_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9171)) := by
  have c113103 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113104 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113105 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113106 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113107 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113108 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_6 := arithEq_of_rows h (row := 1853) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_6
  simp only [← c113103, ← c113104, ← c113105] at e_1853_6
  have e_1853_7 := arithEq_of_rows h (row := 1853) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_7
  simp only [← c113106, ← c113107, k2, ← c113108] at e_1853_7
  have hr := e_1853_7
  simp only [e_1853_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9172)) := by
  have c113109 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113110 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113111 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113112 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113113 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113114 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_8 := arithEq_of_rows h (row := 1853) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_8
  simp only [← c113109, ← c113110, ← c113111] at e_1853_8
  have e_1853_9 := arithEq_of_rows h (row := 1853) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_9
  simp only [← c113112, ← c113113, k2, ← c113114] at e_1853_9
  have hr := e_1853_9
  simp only [e_1853_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9173)) := by
  have c113115 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113116 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113117 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113118 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113119 := (publicBatchWrapper86_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113120 := (publicBatchWrapper86_copies3535 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_10 := arithEq_of_rows h (row := 1853) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_10
  simp only [← c113115, ← c113116, ← c113117] at e_1853_10
  have e_1853_11 := arithEq_of_rows h (row := 1853) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_11
  simp only [← c113118, ← c113119, k2, ← c113120] at e_1853_11
  have hr := e_1853_11
  simp only [e_1853_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9174)) := by
  have c113121 := (publicBatchWrapper86_copies3535 a h).2.1
  have c113122 := (publicBatchWrapper86_copies3535 a h).2.2.1
  have c113123 := (publicBatchWrapper86_copies3535 a h).2.2.2.1
  have c113124 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.1
  have c113125 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.1
  have c113126 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_12 := arithEq_of_rows h (row := 1853) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_12
  simp only [← c113121, ← c113122, ← c113123] at e_1853_12
  have e_1853_13 := arithEq_of_rows h (row := 1853) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_13
  simp only [← c113124, ← c113125, k2, ← c113126] at e_1853_13
  have hr := e_1853_13
  simp only [e_1853_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9175)) := by
  have c113127 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.1
  have c113128 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.1
  have c113129 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.1
  have c113130 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113131 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113132 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_14 := arithEq_of_rows h (row := 1853) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_14
  simp only [← c113127, ← c113128, ← c113129] at e_1853_14
  have e_1853_15 := arithEq_of_rows h (row := 1853) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_15
  simp only [← c113130, ← c113131, k2, ← c113132] at e_1853_15
  have hr := e_1853_15
  simp only [e_1853_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9176)) := by
  have c113133 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113134 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113135 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113136 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113137 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113138 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_16 := arithEq_of_rows h (row := 1853) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_16
  simp only [← c113133, ← c113134, ← c113135] at e_1853_16
  have e_1853_17 := arithEq_of_rows h (row := 1853) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_17
  simp only [← c113136, ← c113137, k2, ← c113138] at e_1853_17
  have hr := e_1853_17
  simp only [e_1853_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1853 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9177)) := by
  have c113139 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113140 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113141 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113142 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113143 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113144 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1853_18 := arithEq_of_rows h (row := 1853) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_18
  simp only [← c113139, ← c113140, ← c113141] at e_1853_18
  have e_1853_19 := arithEq_of_rows h (row := 1853) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_19
  simp only [← c113142, ← c113143, k2, ← c113144] at e_1853_19
  have hr := e_1853_19
  simp only [e_1853_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9178)) := by
  have c113145 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113146 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113147 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113148 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113149 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113150 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_0 := arithEq_of_rows h (row := 1854) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_0
  simp only [← c113145, ← c113146, ← c113147] at e_1854_0
  have e_1854_1 := arithEq_of_rows h (row := 1854) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_1
  simp only [← c113148, ← c113149, k2, ← c113150] at e_1854_1
  have hr := e_1854_1
  simp only [e_1854_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9179)) := by
  have c113151 := (publicBatchWrapper86_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113152 := (publicBatchWrapper86_copies3536 a h).1
  have c113153 := (publicBatchWrapper86_copies3536 a h).2.1
  have c113154 := (publicBatchWrapper86_copies3536 a h).2.2.1
  have c113155 := (publicBatchWrapper86_copies3536 a h).2.2.2.1
  have c113156 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_2 := arithEq_of_rows h (row := 1854) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_2
  simp only [← c113151, ← c113152, ← c113153] at e_1854_2
  have e_1854_3 := arithEq_of_rows h (row := 1854) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_3
  simp only [← c113154, ← c113155, k2, ← c113156] at e_1854_3
  have hr := e_1854_3
  simp only [e_1854_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9180)) := by
  have c113157 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.1
  have c113158 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.1
  have c113159 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.1
  have c113160 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.1
  have c113161 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.1
  have c113162 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_4 := arithEq_of_rows h (row := 1854) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_4
  simp only [← c113157, ← c113158, ← c113159] at e_1854_4
  have e_1854_5 := arithEq_of_rows h (row := 1854) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_5
  simp only [← c113160, ← c113161, k2, ← c113162] at e_1854_5
  have hr := e_1854_5
  simp only [e_1854_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9181)) := by
  have c113163 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113164 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113165 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113166 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113167 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113168 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_6 := arithEq_of_rows h (row := 1854) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_6
  simp only [← c113163, ← c113164, ← c113165] at e_1854_6
  have e_1854_7 := arithEq_of_rows h (row := 1854) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_7
  simp only [← c113166, ← c113167, k2, ← c113168] at e_1854_7
  have hr := e_1854_7
  simp only [e_1854_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9182)) := by
  have c113169 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113170 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113171 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113172 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113173 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113174 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_8 := arithEq_of_rows h (row := 1854) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_8
  simp only [← c113169, ← c113170, ← c113171] at e_1854_8
  have e_1854_9 := arithEq_of_rows h (row := 1854) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_9
  simp only [← c113172, ← c113173, k2, ← c113174] at e_1854_9
  have hr := e_1854_9
  simp only [e_1854_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9183)) := by
  have c113175 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113176 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113177 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113178 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113179 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113180 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_10 := arithEq_of_rows h (row := 1854) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_10
  simp only [← c113175, ← c113176, ← c113177] at e_1854_10
  have e_1854_11 := arithEq_of_rows h (row := 1854) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_11
  simp only [← c113178, ← c113179, k2, ← c113180] at e_1854_11
  have hr := e_1854_11
  simp only [e_1854_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9184)) := by
  have c113181 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113182 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113183 := (publicBatchWrapper86_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113184 := (publicBatchWrapper86_copies3537 a h).1
  have c113185 := (publicBatchWrapper86_copies3537 a h).2.1
  have c113186 := (publicBatchWrapper86_copies3537 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_12 := arithEq_of_rows h (row := 1854) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_12
  simp only [← c113181, ← c113182, ← c113183] at e_1854_12
  have e_1854_13 := arithEq_of_rows h (row := 1854) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_13
  simp only [← c113184, ← c113185, k2, ← c113186] at e_1854_13
  have hr := e_1854_13
  simp only [e_1854_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9185)) := by
  have c113187 := (publicBatchWrapper86_copies3537 a h).2.2.2.1
  have c113188 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.1
  have c113189 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.1
  have c113190 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.1
  have c113191 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.1
  have c113192 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_14 := arithEq_of_rows h (row := 1854) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_14
  simp only [← c113187, ← c113188, ← c113189] at e_1854_14
  have e_1854_15 := arithEq_of_rows h (row := 1854) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_15
  simp only [← c113190, ← c113191, k2, ← c113192] at e_1854_15
  have hr := e_1854_15
  simp only [e_1854_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9186)) := by
  have c113193 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.1
  have c113194 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113195 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113196 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113197 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113198 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_16 := arithEq_of_rows h (row := 1854) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_16
  simp only [← c113193, ← c113194, ← c113195] at e_1854_16
  have e_1854_17 := arithEq_of_rows h (row := 1854) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_17
  simp only [← c113196, ← c113197, k2, ← c113198] at e_1854_17
  have hr := e_1854_17
  simp only [e_1854_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1854 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9187)) := by
  have c113199 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113200 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113201 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113202 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113203 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113204 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1854_18 := arithEq_of_rows h (row := 1854) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_18
  simp only [← c113199, ← c113200, ← c113201] at e_1854_18
  have e_1854_19 := arithEq_of_rows h (row := 1854) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_19
  simp only [← c113202, ← c113203, k2, ← c113204] at e_1854_19
  have hr := e_1854_19
  simp only [e_1854_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9188)) := by
  have c113205 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113206 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113207 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113208 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113209 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113210 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_0 := arithEq_of_rows h (row := 1855) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_0
  simp only [← c113205, ← c113206, ← c113207] at e_1855_0
  have e_1855_1 := arithEq_of_rows h (row := 1855) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_1
  simp only [← c113208, ← c113209, k2, ← c113210] at e_1855_1
  have hr := e_1855_1
  simp only [e_1855_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9189)) := by
  have c113211 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113212 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113213 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113214 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113215 := (publicBatchWrapper86_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113216 := (publicBatchWrapper86_copies3538 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_2 := arithEq_of_rows h (row := 1855) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_2
  simp only [← c113211, ← c113212, ← c113213] at e_1855_2
  have e_1855_3 := arithEq_of_rows h (row := 1855) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_3
  simp only [← c113214, ← c113215, k2, ← c113216] at e_1855_3
  have hr := e_1855_3
  simp only [e_1855_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9190)) := by
  have c113217 := (publicBatchWrapper86_copies3538 a h).2.1
  have c113218 := (publicBatchWrapper86_copies3538 a h).2.2.1
  have c113219 := (publicBatchWrapper86_copies3538 a h).2.2.2.1
  have c113220 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.1
  have c113221 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.1
  have c113222 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_4 := arithEq_of_rows h (row := 1855) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_4
  simp only [← c113217, ← c113218, ← c113219] at e_1855_4
  have e_1855_5 := arithEq_of_rows h (row := 1855) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_5
  simp only [← c113220, ← c113221, k2, ← c113222] at e_1855_5
  have hr := e_1855_5
  simp only [e_1855_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9191)) := by
  have c113223 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.1
  have c113224 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.1
  have c113225 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.1
  have c113226 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113227 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113228 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_6 := arithEq_of_rows h (row := 1855) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_6
  simp only [← c113223, ← c113224, ← c113225] at e_1855_6
  have e_1855_7 := arithEq_of_rows h (row := 1855) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_7
  simp only [← c113226, ← c113227, k2, ← c113228] at e_1855_7
  have hr := e_1855_7
  simp only [e_1855_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 39) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9192)) := by
  have c113229 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113230 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113231 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113232 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113233 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113234 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_8 := arithEq_of_rows h (row := 1855) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_8
  simp only [← c113229, ← c113230, ← c113231] at e_1855_8
  have e_1855_9 := arithEq_of_rows h (row := 1855) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_9
  simp only [← c113232, ← c113233, k2, ← c113234] at e_1855_9
  have hr := e_1855_9
  simp only [e_1855_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 47) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9193)) := by
  have c113235 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113236 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113237 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113238 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113239 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113240 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_10 := arithEq_of_rows h (row := 1855) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_10
  simp only [← c113235, ← c113236, ← c113237] at e_1855_10
  have e_1855_11 := arithEq_of_rows h (row := 1855) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_11
  simp only [← c113238, ← c113239, k2, ← c113240] at e_1855_11
  have hr := e_1855_11
  simp only [e_1855_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 55) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9194)) := by
  have c113241 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113242 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113243 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113244 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113245 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113246 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_12 := arithEq_of_rows h (row := 1855) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_12
  simp only [← c113241, ← c113242, ← c113243] at e_1855_12
  have e_1855_13 := arithEq_of_rows h (row := 1855) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_13
  simp only [← c113244, ← c113245, k2, ← c113246] at e_1855_13
  have hr := e_1855_13
  simp only [e_1855_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 63) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9195)) := by
  have c113247 := (publicBatchWrapper86_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113248 := (publicBatchWrapper86_copies3539 a h).1
  have c113249 := (publicBatchWrapper86_copies3539 a h).2.1
  have c113250 := (publicBatchWrapper86_copies3539 a h).2.2.1
  have c113251 := (publicBatchWrapper86_copies3539 a h).2.2.2.1
  have c113252 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_14 := arithEq_of_rows h (row := 1855) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_14
  simp only [← c113247, ← c113248, ← c113249] at e_1855_14
  have e_1855_15 := arithEq_of_rows h (row := 1855) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_15
  simp only [← c113250, ← c113251, k2, ← c113252] at e_1855_15
  have hr := e_1855_15
  simp only [e_1855_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 71) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9196)) := by
  have c113253 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.1
  have c113254 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.1
  have c113255 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.1
  have c113256 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.1
  have c113257 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.1
  have c113258 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_16 := arithEq_of_rows h (row := 1855) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_16
  simp only [← c113253, ← c113254, ← c113255] at e_1855_16
  have e_1855_17 := arithEq_of_rows h (row := 1855) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_17
  simp only [← c113256, ← c113257, k2, ← c113258] at e_1855_17
  have hr := e_1855_17
  simp only [e_1855_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1855 79) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9197)) := by
  have c113259 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113260 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113261 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113262 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113263 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113264 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1855_18 := arithEq_of_rows h (row := 1855) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_18
  simp only [← c113259, ← c113260, ← c113261] at e_1855_18
  have e_1855_19 := arithEq_of_rows h (row := 1855) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_19
  simp only [← c113262, ← c113263, k2, ← c113264] at e_1855_19
  have hr := e_1855_19
  simp only [e_1855_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 7) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9198)) := by
  have c113265 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113266 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113267 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113268 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113269 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113270 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_0 := arithEq_of_rows h (row := 1856) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_0
  simp only [← c113265, ← c113266, ← c113267] at e_1856_0
  have e_1856_1 := arithEq_of_rows h (row := 1856) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_1
  simp only [← c113268, ← c113269, k2, ← c113270] at e_1856_1
  have hr := e_1856_1
  simp only [e_1856_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 15) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9199)) := by
  have c113271 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113272 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113273 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113274 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113275 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113276 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_2 := arithEq_of_rows h (row := 1856) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_2
  simp only [← c113271, ← c113272, ← c113273] at e_1856_2
  have e_1856_3 := arithEq_of_rows h (row := 1856) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_3
  simp only [← c113274, ← c113275, k2, ← c113276] at e_1856_3
  have hr := e_1856_3
  simp only [e_1856_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 23) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9200)) := by
  have c113277 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113278 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113279 := (publicBatchWrapper86_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113280 := (publicBatchWrapper86_copies3540 a h).1
  have c113281 := (publicBatchWrapper86_copies3540 a h).2.1
  have c113282 := (publicBatchWrapper86_copies3540 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_4 := arithEq_of_rows h (row := 1856) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_4
  simp only [← c113277, ← c113278, ← c113279] at e_1856_4
  have e_1856_5 := arithEq_of_rows h (row := 1856) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_5
  simp only [← c113280, ← c113281, k2, ← c113282] at e_1856_5
  have hr := e_1856_5
  simp only [e_1856_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 31) = bselect (a (.wire 46 7)) (a (.virt 200)) (a (.virt 9201)) := by
  have c113283 := (publicBatchWrapper86_copies3540 a h).2.2.2.1
  have c113284 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.1
  have c113285 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.1
  have c113286 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.1
  have c113287 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.1
  have c113288 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_6 := arithEq_of_rows h (row := 1856) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_6
  simp only [← c113283, ← c113284, ← c113285] at e_1856_6
  have e_1856_7 := arithEq_of_rows h (row := 1856) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_7
  simp only [← c113286, ← c113287, k2, ← c113288] at e_1856_7
  have hr := e_1856_7
  simp only [e_1856_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9370)) := by
  have c113289 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.1
  have c113290 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113291 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113292 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113293 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113294 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_8 := arithEq_of_rows h (row := 1856) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_8
  simp only [← c113289, ← c113290, ← c113291] at e_1856_8
  have e_1856_9 := arithEq_of_rows h (row := 1856) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_9
  simp only [← c113292, ← c113293, k2, ← c113294] at e_1856_9
  have hr := e_1856_9
  simp only [e_1856_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9371)) := by
  have c113295 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113296 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113297 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113298 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113299 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113300 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_10 := arithEq_of_rows h (row := 1856) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_10
  simp only [← c113295, ← c113296, ← c113297] at e_1856_10
  have e_1856_11 := arithEq_of_rows h (row := 1856) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_11
  simp only [← c113298, ← c113299, k2, ← c113300] at e_1856_11
  have hr := e_1856_11
  simp only [e_1856_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9372)) := by
  have c113301 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113302 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113303 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113304 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113305 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113306 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_12 := arithEq_of_rows h (row := 1856) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_12
  simp only [← c113301, ← c113302, ← c113303] at e_1856_12
  have e_1856_13 := arithEq_of_rows h (row := 1856) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_13
  simp only [← c113304, ← c113305, k2, ← c113306] at e_1856_13
  have hr := e_1856_13
  simp only [e_1856_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9373)) := by
  have c113307 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113308 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113309 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113310 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113311 := (publicBatchWrapper86_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113312 := (publicBatchWrapper86_copies3541 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_14 := arithEq_of_rows h (row := 1856) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_14
  simp only [← c113307, ← c113308, ← c113309] at e_1856_14
  have e_1856_15 := arithEq_of_rows h (row := 1856) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_15
  simp only [← c113310, ← c113311, k2, ← c113312] at e_1856_15
  have hr := e_1856_15
  simp only [e_1856_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9374)) := by
  have c113313 := (publicBatchWrapper86_copies3541 a h).2.1
  have c113314 := (publicBatchWrapper86_copies3541 a h).2.2.1
  have c113315 := (publicBatchWrapper86_copies3541 a h).2.2.2.1
  have c113316 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.1
  have c113317 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.1
  have c113318 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_16 := arithEq_of_rows h (row := 1856) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_16
  simp only [← c113313, ← c113314, ← c113315] at e_1856_16
  have e_1856_17 := arithEq_of_rows h (row := 1856) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_17
  simp only [← c113316, ← c113317, k2, ← c113318] at e_1856_17
  have hr := e_1856_17
  simp only [e_1856_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1856 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9375)) := by
  have c113319 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.1
  have c113320 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.1
  have c113321 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.1
  have c113322 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113323 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113324 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1856_18 := arithEq_of_rows h (row := 1856) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_18
  simp only [← c113319, ← c113320, ← c113321] at e_1856_18
  have e_1856_19 := arithEq_of_rows h (row := 1856) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_19
  simp only [← c113322, ← c113323, k2, ← c113324] at e_1856_19
  have hr := e_1856_19
  simp only [e_1856_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9376)) := by
  have c113325 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113326 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113327 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113328 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113329 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113330 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_0 := arithEq_of_rows h (row := 1857) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_0
  simp only [← c113325, ← c113326, ← c113327] at e_1857_0
  have e_1857_1 := arithEq_of_rows h (row := 1857) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_1
  simp only [← c113328, ← c113329, k2, ← c113330] at e_1857_1
  have hr := e_1857_1
  simp only [e_1857_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9377)) := by
  have c113331 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113332 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113333 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113334 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113335 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113336 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_2 := arithEq_of_rows h (row := 1857) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_2
  simp only [← c113331, ← c113332, ← c113333] at e_1857_2
  have e_1857_3 := arithEq_of_rows h (row := 1857) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_3
  simp only [← c113334, ← c113335, k2, ← c113336] at e_1857_3
  have hr := e_1857_3
  simp only [e_1857_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9378)) := by
  have c113337 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113338 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113339 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113340 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113341 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113342 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_4 := arithEq_of_rows h (row := 1857) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_4
  simp only [← c113337, ← c113338, ← c113339] at e_1857_4
  have e_1857_5 := arithEq_of_rows h (row := 1857) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_5
  simp only [← c113340, ← c113341, k2, ← c113342] at e_1857_5
  have hr := e_1857_5
  simp only [e_1857_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9379)) := by
  have c113343 := (publicBatchWrapper86_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113344 := (publicBatchWrapper86_copies3542 a h).1
  have c113345 := (publicBatchWrapper86_copies3542 a h).2.1
  have c113346 := (publicBatchWrapper86_copies3542 a h).2.2.1
  have c113347 := (publicBatchWrapper86_copies3542 a h).2.2.2.1
  have c113348 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_6 := arithEq_of_rows h (row := 1857) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_6
  simp only [← c113343, ← c113344, ← c113345] at e_1857_6
  have e_1857_7 := arithEq_of_rows h (row := 1857) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_7
  simp only [← c113346, ← c113347, k2, ← c113348] at e_1857_7
  have hr := e_1857_7
  simp only [e_1857_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9380)) := by
  have c113349 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.1
  have c113350 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.1
  have c113351 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.1
  have c113352 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.1
  have c113353 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.1
  have c113354 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_8 := arithEq_of_rows h (row := 1857) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_8
  simp only [← c113349, ← c113350, ← c113351] at e_1857_8
  have e_1857_9 := arithEq_of_rows h (row := 1857) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_9
  simp only [← c113352, ← c113353, k2, ← c113354] at e_1857_9
  have hr := e_1857_9
  simp only [e_1857_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9381)) := by
  have c113355 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113356 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113357 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113358 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113359 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113360 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_10 := arithEq_of_rows h (row := 1857) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_10
  simp only [← c113355, ← c113356, ← c113357] at e_1857_10
  have e_1857_11 := arithEq_of_rows h (row := 1857) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_11
  simp only [← c113358, ← c113359, k2, ← c113360] at e_1857_11
  have hr := e_1857_11
  simp only [e_1857_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9382)) := by
  have c113361 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113362 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113363 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113364 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113365 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113366 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_12 := arithEq_of_rows h (row := 1857) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_12
  simp only [← c113361, ← c113362, ← c113363] at e_1857_12
  have e_1857_13 := arithEq_of_rows h (row := 1857) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_13
  simp only [← c113364, ← c113365, k2, ← c113366] at e_1857_13
  have hr := e_1857_13
  simp only [e_1857_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9383)) := by
  have c113367 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113368 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113369 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113370 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113371 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113372 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_14 := arithEq_of_rows h (row := 1857) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_14
  simp only [← c113367, ← c113368, ← c113369] at e_1857_14
  have e_1857_15 := arithEq_of_rows h (row := 1857) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_15
  simp only [← c113370, ← c113371, k2, ← c113372] at e_1857_15
  have hr := e_1857_15
  simp only [e_1857_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9384)) := by
  have c113373 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113374 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113375 := (publicBatchWrapper86_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113376 := (publicBatchWrapper86_copies3543 a h).1
  have c113377 := (publicBatchWrapper86_copies3543 a h).2.1
  have c113378 := (publicBatchWrapper86_copies3543 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_16 := arithEq_of_rows h (row := 1857) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_16
  simp only [← c113373, ← c113374, ← c113375] at e_1857_16
  have e_1857_17 := arithEq_of_rows h (row := 1857) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_17
  simp only [← c113376, ← c113377, k2, ← c113378] at e_1857_17
  have hr := e_1857_17
  simp only [e_1857_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1857 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9385)) := by
  have c113379 := (publicBatchWrapper86_copies3543 a h).2.2.2.1
  have c113380 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.1
  have c113381 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.1
  have c113382 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.1
  have c113383 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.1
  have c113384 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1857_18 := arithEq_of_rows h (row := 1857) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_18
  simp only [← c113379, ← c113380, ← c113381] at e_1857_18
  have e_1857_19 := arithEq_of_rows h (row := 1857) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_19
  simp only [← c113382, ← c113383, k2, ← c113384] at e_1857_19
  have hr := e_1857_19
  simp only [e_1857_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9386)) := by
  have c113385 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.1
  have c113386 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113387 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113388 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113389 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113390 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_0 := arithEq_of_rows h (row := 1858) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_0
  simp only [← c113385, ← c113386, ← c113387] at e_1858_0
  have e_1858_1 := arithEq_of_rows h (row := 1858) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_1
  simp only [← c113388, ← c113389, k2, ← c113390] at e_1858_1
  have hr := e_1858_1
  simp only [e_1858_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9387)) := by
  have c113391 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113392 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113393 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113394 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113395 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113396 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_2 := arithEq_of_rows h (row := 1858) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_2
  simp only [← c113391, ← c113392, ← c113393] at e_1858_2
  have e_1858_3 := arithEq_of_rows h (row := 1858) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_3
  simp only [← c113394, ← c113395, k2, ← c113396] at e_1858_3
  have hr := e_1858_3
  simp only [e_1858_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9388)) := by
  have c113397 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113398 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113399 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113400 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113401 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113402 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_4 := arithEq_of_rows h (row := 1858) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_4
  simp only [← c113397, ← c113398, ← c113399] at e_1858_4
  have e_1858_5 := arithEq_of_rows h (row := 1858) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_5
  simp only [← c113400, ← c113401, k2, ← c113402] at e_1858_5
  have hr := e_1858_5
  simp only [e_1858_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9389)) := by
  have c113403 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113404 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113405 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113406 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113407 := (publicBatchWrapper86_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113408 := (publicBatchWrapper86_copies3544 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_6 := arithEq_of_rows h (row := 1858) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_6
  simp only [← c113403, ← c113404, ← c113405] at e_1858_6
  have e_1858_7 := arithEq_of_rows h (row := 1858) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_7
  simp only [← c113406, ← c113407, k2, ← c113408] at e_1858_7
  have hr := e_1858_7
  simp only [e_1858_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9390)) := by
  have c113409 := (publicBatchWrapper86_copies3544 a h).2.1
  have c113410 := (publicBatchWrapper86_copies3544 a h).2.2.1
  have c113411 := (publicBatchWrapper86_copies3544 a h).2.2.2.1
  have c113412 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.1
  have c113413 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.1
  have c113414 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_8 := arithEq_of_rows h (row := 1858) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_8
  simp only [← c113409, ← c113410, ← c113411] at e_1858_8
  have e_1858_9 := arithEq_of_rows h (row := 1858) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_9
  simp only [← c113412, ← c113413, k2, ← c113414] at e_1858_9
  have hr := e_1858_9
  simp only [e_1858_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9391)) := by
  have c113415 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.1
  have c113416 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.1
  have c113417 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.1
  have c113418 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113419 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113420 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_10 := arithEq_of_rows h (row := 1858) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_10
  simp only [← c113415, ← c113416, ← c113417] at e_1858_10
  have e_1858_11 := arithEq_of_rows h (row := 1858) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_11
  simp only [← c113418, ← c113419, k2, ← c113420] at e_1858_11
  have hr := e_1858_11
  simp only [e_1858_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 55) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9392)) := by
  have c113421 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113422 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113423 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113424 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113425 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113426 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_12 := arithEq_of_rows h (row := 1858) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_12
  simp only [← c113421, ← c113422, ← c113423] at e_1858_12
  have e_1858_13 := arithEq_of_rows h (row := 1858) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_13
  simp only [← c113424, ← c113425, k2, ← c113426] at e_1858_13
  have hr := e_1858_13
  simp only [e_1858_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 63) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9393)) := by
  have c113427 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113428 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113429 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113430 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113431 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113432 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_14 := arithEq_of_rows h (row := 1858) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_14
  simp only [← c113427, ← c113428, ← c113429] at e_1858_14
  have e_1858_15 := arithEq_of_rows h (row := 1858) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_15
  simp only [← c113430, ← c113431, k2, ← c113432] at e_1858_15
  have hr := e_1858_15
  simp only [e_1858_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 71) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9394)) := by
  have c113433 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113434 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113435 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113436 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113437 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113438 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_16 := arithEq_of_rows h (row := 1858) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_16
  simp only [← c113433, ← c113434, ← c113435] at e_1858_16
  have e_1858_17 := arithEq_of_rows h (row := 1858) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_17
  simp only [← c113436, ← c113437, k2, ← c113438] at e_1858_17
  have hr := e_1858_17
  simp only [e_1858_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1858 79) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9395)) := by
  have c113439 := (publicBatchWrapper86_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113440 := (publicBatchWrapper86_copies3545 a h).1
  have c113441 := (publicBatchWrapper86_copies3545 a h).2.1
  have c113442 := (publicBatchWrapper86_copies3545 a h).2.2.1
  have c113443 := (publicBatchWrapper86_copies3545 a h).2.2.2.1
  have c113444 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1858_18 := arithEq_of_rows h (row := 1858) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_18
  simp only [← c113439, ← c113440, ← c113441] at e_1858_18
  have e_1858_19 := arithEq_of_rows h (row := 1858) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_19
  simp only [← c113442, ← c113443, k2, ← c113444] at e_1858_19
  have hr := e_1858_19
  simp only [e_1858_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 7) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9396)) := by
  have c113445 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.1
  have c113446 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.1
  have c113447 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.1
  have c113448 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.1
  have c113449 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.1
  have c113450 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_0 := arithEq_of_rows h (row := 1859) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_0
  simp only [← c113445, ← c113446, ← c113447] at e_1859_0
  have e_1859_1 := arithEq_of_rows h (row := 1859) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_1
  simp only [← c113448, ← c113449, k2, ← c113450] at e_1859_1
  have hr := e_1859_1
  simp only [e_1859_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 15) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9397)) := by
  have c113451 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113452 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113453 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113454 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113455 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113456 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_2 := arithEq_of_rows h (row := 1859) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_2
  simp only [← c113451, ← c113452, ← c113453] at e_1859_2
  have e_1859_3 := arithEq_of_rows h (row := 1859) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_3
  simp only [← c113454, ← c113455, k2, ← c113456] at e_1859_3
  have hr := e_1859_3
  simp only [e_1859_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 23) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9398)) := by
  have c113457 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113458 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113459 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113460 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113461 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113462 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_4 := arithEq_of_rows h (row := 1859) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_4
  simp only [← c113457, ← c113458, ← c113459] at e_1859_4
  have e_1859_5 := arithEq_of_rows h (row := 1859) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_5
  simp only [← c113460, ← c113461, k2, ← c113462] at e_1859_5
  have hr := e_1859_5
  simp only [e_1859_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 31) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9399)) := by
  have c113463 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113464 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113465 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113466 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113467 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113468 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_6 := arithEq_of_rows h (row := 1859) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_6
  simp only [← c113463, ← c113464, ← c113465] at e_1859_6
  have e_1859_7 := arithEq_of_rows h (row := 1859) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_7
  simp only [← c113466, ← c113467, k2, ← c113468] at e_1859_7
  have hr := e_1859_7
  simp only [e_1859_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 39) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9400)) := by
  have c113469 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113470 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113471 := (publicBatchWrapper86_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113472 := (publicBatchWrapper86_copies3546 a h).1
  have c113473 := (publicBatchWrapper86_copies3546 a h).2.1
  have c113474 := (publicBatchWrapper86_copies3546 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_8 := arithEq_of_rows h (row := 1859) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_8
  simp only [← c113469, ← c113470, ← c113471] at e_1859_8
  have e_1859_9 := arithEq_of_rows h (row := 1859) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_9
  simp only [← c113472, ← c113473, k2, ← c113474] at e_1859_9
  have hr := e_1859_9
  simp only [e_1859_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 47) = bselect (a (.wire 46 51)) (a (.virt 200)) (a (.virt 9401)) := by
  have c113475 := (publicBatchWrapper86_copies3546 a h).2.2.2.1
  have c113476 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.1
  have c113477 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.1
  have c113478 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.1
  have c113479 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.1
  have c113480 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_10 := arithEq_of_rows h (row := 1859) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_10
  simp only [← c113475, ← c113476, ← c113477] at e_1859_10
  have e_1859_11 := arithEq_of_rows h (row := 1859) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_11
  simp only [← c113478, ← c113479, k2, ← c113480] at e_1859_11
  have hr := e_1859_11
  simp only [e_1859_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9570)) := by
  have c113481 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.1
  have c113482 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113483 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113484 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113485 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113486 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_12 := arithEq_of_rows h (row := 1859) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_12
  simp only [← c113481, ← c113482, ← c113483] at e_1859_12
  have e_1859_13 := arithEq_of_rows h (row := 1859) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_13
  simp only [← c113484, ← c113485, k2, ← c113486] at e_1859_13
  have hr := e_1859_13
  simp only [e_1859_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9571)) := by
  have c113487 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113488 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113489 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113490 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113491 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113492 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_14 := arithEq_of_rows h (row := 1859) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_14
  simp only [← c113487, ← c113488, ← c113489] at e_1859_14
  have e_1859_15 := arithEq_of_rows h (row := 1859) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_15
  simp only [← c113490, ← c113491, k2, ← c113492] at e_1859_15
  have hr := e_1859_15
  simp only [e_1859_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9572)) := by
  have c113493 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113494 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113495 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113496 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113497 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113498 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_16 := arithEq_of_rows h (row := 1859) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_16
  simp only [← c113493, ← c113494, ← c113495] at e_1859_16
  have e_1859_17 := arithEq_of_rows h (row := 1859) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_17
  simp only [← c113496, ← c113497, k2, ← c113498] at e_1859_17
  have hr := e_1859_17
  simp only [e_1859_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1859 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9573)) := by
  have c113499 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113500 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113501 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113502 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113503 := (publicBatchWrapper86_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113504 := (publicBatchWrapper86_copies3547 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1859_18 := arithEq_of_rows h (row := 1859) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_18
  simp only [← c113499, ← c113500, ← c113501] at e_1859_18
  have e_1859_19 := arithEq_of_rows h (row := 1859) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_19
  simp only [← c113502, ← c113503, k2, ← c113504] at e_1859_19
  have hr := e_1859_19
  simp only [e_1859_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9574)) := by
  have c113505 := (publicBatchWrapper86_copies3547 a h).2.1
  have c113506 := (publicBatchWrapper86_copies3547 a h).2.2.1
  have c113507 := (publicBatchWrapper86_copies3547 a h).2.2.2.1
  have c113508 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.1
  have c113509 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.1
  have c113510 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_0 := arithEq_of_rows h (row := 1860) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_0
  simp only [← c113505, ← c113506, ← c113507] at e_1860_0
  have e_1860_1 := arithEq_of_rows h (row := 1860) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_1
  simp only [← c113508, ← c113509, k2, ← c113510] at e_1860_1
  have hr := e_1860_1
  simp only [e_1860_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9575)) := by
  have c113511 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.1
  have c113512 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.1
  have c113513 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.1
  have c113514 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113515 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113516 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_2 := arithEq_of_rows h (row := 1860) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_2
  simp only [← c113511, ← c113512, ← c113513] at e_1860_2
  have e_1860_3 := arithEq_of_rows h (row := 1860) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_3
  simp only [← c113514, ← c113515, k2, ← c113516] at e_1860_3
  have hr := e_1860_3
  simp only [e_1860_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9576)) := by
  have c113517 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113518 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113519 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113520 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113521 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113522 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_4 := arithEq_of_rows h (row := 1860) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_4
  simp only [← c113517, ← c113518, ← c113519] at e_1860_4
  have e_1860_5 := arithEq_of_rows h (row := 1860) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_5
  simp only [← c113520, ← c113521, k2, ← c113522] at e_1860_5
  have hr := e_1860_5
  simp only [e_1860_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9577)) := by
  have c113523 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113524 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113525 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113526 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113527 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113528 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_6 := arithEq_of_rows h (row := 1860) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_6
  simp only [← c113523, ← c113524, ← c113525] at e_1860_6
  have e_1860_7 := arithEq_of_rows h (row := 1860) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_7
  simp only [← c113526, ← c113527, k2, ← c113528] at e_1860_7
  have hr := e_1860_7
  simp only [e_1860_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9578)) := by
  have c113529 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113530 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113531 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113532 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113533 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113534 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_8 := arithEq_of_rows h (row := 1860) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_8
  simp only [← c113529, ← c113530, ← c113531] at e_1860_8
  have e_1860_9 := arithEq_of_rows h (row := 1860) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_9
  simp only [← c113532, ← c113533, k2, ← c113534] at e_1860_9
  have hr := e_1860_9
  simp only [e_1860_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9579)) := by
  have c113535 := (publicBatchWrapper86_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113536 := (publicBatchWrapper86_copies3548 a h).1
  have c113537 := (publicBatchWrapper86_copies3548 a h).2.1
  have c113538 := (publicBatchWrapper86_copies3548 a h).2.2.1
  have c113539 := (publicBatchWrapper86_copies3548 a h).2.2.2.1
  have c113540 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_10 := arithEq_of_rows h (row := 1860) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_10
  simp only [← c113535, ← c113536, ← c113537] at e_1860_10
  have e_1860_11 := arithEq_of_rows h (row := 1860) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_11
  simp only [← c113538, ← c113539, k2, ← c113540] at e_1860_11
  have hr := e_1860_11
  simp only [e_1860_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9580)) := by
  have c113541 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.1
  have c113542 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.1
  have c113543 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.1
  have c113544 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.1
  have c113545 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.1
  have c113546 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_12 := arithEq_of_rows h (row := 1860) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_12
  simp only [← c113541, ← c113542, ← c113543] at e_1860_12
  have e_1860_13 := arithEq_of_rows h (row := 1860) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_13
  simp only [← c113544, ← c113545, k2, ← c113546] at e_1860_13
  have hr := e_1860_13
  simp only [e_1860_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9581)) := by
  have c113547 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113548 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113549 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113550 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113551 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113552 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_14 := arithEq_of_rows h (row := 1860) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_14
  simp only [← c113547, ← c113548, ← c113549] at e_1860_14
  have e_1860_15 := arithEq_of_rows h (row := 1860) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_15
  simp only [← c113550, ← c113551, k2, ← c113552] at e_1860_15
  have hr := e_1860_15
  simp only [e_1860_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9582)) := by
  have c113553 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113554 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113555 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113556 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113557 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113558 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_16 := arithEq_of_rows h (row := 1860) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_16
  simp only [← c113553, ← c113554, ← c113555] at e_1860_16
  have e_1860_17 := arithEq_of_rows h (row := 1860) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_17
  simp only [← c113556, ← c113557, k2, ← c113558] at e_1860_17
  have hr := e_1860_17
  simp only [e_1860_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1860 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9583)) := by
  have c113559 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113560 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113561 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113562 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113563 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113564 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1860_18 := arithEq_of_rows h (row := 1860) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_18
  simp only [← c113559, ← c113560, ← c113561] at e_1860_18
  have e_1860_19 := arithEq_of_rows h (row := 1860) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_19
  simp only [← c113562, ← c113563, k2, ← c113564] at e_1860_19
  have hr := e_1860_19
  simp only [e_1860_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9584)) := by
  have c113565 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113566 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113567 := (publicBatchWrapper86_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113568 := (publicBatchWrapper86_copies3549 a h).1
  have c113569 := (publicBatchWrapper86_copies3549 a h).2.1
  have c113570 := (publicBatchWrapper86_copies3549 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_0 := arithEq_of_rows h (row := 1861) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_0
  simp only [← c113565, ← c113566, ← c113567] at e_1861_0
  have e_1861_1 := arithEq_of_rows h (row := 1861) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_1
  simp only [← c113568, ← c113569, k2, ← c113570] at e_1861_1
  have hr := e_1861_1
  simp only [e_1861_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9585)) := by
  have c113571 := (publicBatchWrapper86_copies3549 a h).2.2.2.1
  have c113572 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.1
  have c113573 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.1
  have c113574 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.1
  have c113575 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.1
  have c113576 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_2 := arithEq_of_rows h (row := 1861) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_2
  simp only [← c113571, ← c113572, ← c113573] at e_1861_2
  have e_1861_3 := arithEq_of_rows h (row := 1861) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_3
  simp only [← c113574, ← c113575, k2, ← c113576] at e_1861_3
  have hr := e_1861_3
  simp only [e_1861_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9586)) := by
  have c113577 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.1
  have c113578 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113579 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113580 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113581 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113582 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_4 := arithEq_of_rows h (row := 1861) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_4
  simp only [← c113577, ← c113578, ← c113579] at e_1861_4
  have e_1861_5 := arithEq_of_rows h (row := 1861) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_5
  simp only [← c113580, ← c113581, k2, ← c113582] at e_1861_5
  have hr := e_1861_5
  simp only [e_1861_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9587)) := by
  have c113583 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113584 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113585 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113586 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113587 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113588 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_6 := arithEq_of_rows h (row := 1861) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_6
  simp only [← c113583, ← c113584, ← c113585] at e_1861_6
  have e_1861_7 := arithEq_of_rows h (row := 1861) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_7
  simp only [← c113586, ← c113587, k2, ← c113588] at e_1861_7
  have hr := e_1861_7
  simp only [e_1861_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9588)) := by
  have c113589 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113590 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113591 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113592 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113593 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113594 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_8 := arithEq_of_rows h (row := 1861) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_8
  simp only [← c113589, ← c113590, ← c113591] at e_1861_8
  have e_1861_9 := arithEq_of_rows h (row := 1861) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_9
  simp only [← c113592, ← c113593, k2, ← c113594] at e_1861_9
  have hr := e_1861_9
  simp only [e_1861_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9589)) := by
  have c113595 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113596 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113597 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113598 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113599 := (publicBatchWrapper86_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113600 := (publicBatchWrapper86_copies3550 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_10 := arithEq_of_rows h (row := 1861) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_10
  simp only [← c113595, ← c113596, ← c113597] at e_1861_10
  have e_1861_11 := arithEq_of_rows h (row := 1861) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_11
  simp only [← c113598, ← c113599, k2, ← c113600] at e_1861_11
  have hr := e_1861_11
  simp only [e_1861_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9590)) := by
  have c113601 := (publicBatchWrapper86_copies3550 a h).2.1
  have c113602 := (publicBatchWrapper86_copies3550 a h).2.2.1
  have c113603 := (publicBatchWrapper86_copies3550 a h).2.2.2.1
  have c113604 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.1
  have c113605 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.1
  have c113606 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_12 := arithEq_of_rows h (row := 1861) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_12
  simp only [← c113601, ← c113602, ← c113603] at e_1861_12
  have e_1861_13 := arithEq_of_rows h (row := 1861) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_13
  simp only [← c113604, ← c113605, k2, ← c113606] at e_1861_13
  have hr := e_1861_13
  simp only [e_1861_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9591)) := by
  have c113607 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.1
  have c113608 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.1
  have c113609 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.1
  have c113610 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113611 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113612 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_14 := arithEq_of_rows h (row := 1861) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_14
  simp only [← c113607, ← c113608, ← c113609] at e_1861_14
  have e_1861_15 := arithEq_of_rows h (row := 1861) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_15
  simp only [← c113610, ← c113611, k2, ← c113612] at e_1861_15
  have hr := e_1861_15
  simp only [e_1861_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 71) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9592)) := by
  have c113613 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113614 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113615 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113616 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113617 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113618 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_16 := arithEq_of_rows h (row := 1861) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_16
  simp only [← c113613, ← c113614, ← c113615] at e_1861_16
  have e_1861_17 := arithEq_of_rows h (row := 1861) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_17
  simp only [← c113616, ← c113617, k2, ← c113618] at e_1861_17
  have hr := e_1861_17
  simp only [e_1861_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1861 79) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9593)) := by
  have c113619 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113620 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113621 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113622 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113623 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113624 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1861_18 := arithEq_of_rows h (row := 1861) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_18
  simp only [← c113619, ← c113620, ← c113621] at e_1861_18
  have e_1861_19 := arithEq_of_rows h (row := 1861) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_19
  simp only [← c113622, ← c113623, k2, ← c113624] at e_1861_19
  have hr := e_1861_19
  simp only [e_1861_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 7) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9594)) := by
  have c113625 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113626 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113627 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113628 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113629 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113630 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_0 := arithEq_of_rows h (row := 1862) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_0
  simp only [← c113625, ← c113626, ← c113627] at e_1862_0
  have e_1862_1 := arithEq_of_rows h (row := 1862) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_1
  simp only [← c113628, ← c113629, k2, ← c113630] at e_1862_1
  have hr := e_1862_1
  simp only [e_1862_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 15) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9595)) := by
  have c113631 := (publicBatchWrapper86_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113632 := (publicBatchWrapper86_copies3551 a h).1
  have c113633 := (publicBatchWrapper86_copies3551 a h).2.1
  have c113634 := (publicBatchWrapper86_copies3551 a h).2.2.1
  have c113635 := (publicBatchWrapper86_copies3551 a h).2.2.2.1
  have c113636 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_2 := arithEq_of_rows h (row := 1862) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_2
  simp only [← c113631, ← c113632, ← c113633] at e_1862_2
  have e_1862_3 := arithEq_of_rows h (row := 1862) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_3
  simp only [← c113634, ← c113635, k2, ← c113636] at e_1862_3
  have hr := e_1862_3
  simp only [e_1862_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 23) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9596)) := by
  have c113637 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.1
  have c113638 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.1
  have c113639 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.1
  have c113640 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.1
  have c113641 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.1
  have c113642 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_4 := arithEq_of_rows h (row := 1862) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_4
  simp only [← c113637, ← c113638, ← c113639] at e_1862_4
  have e_1862_5 := arithEq_of_rows h (row := 1862) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_5
  simp only [← c113640, ← c113641, k2, ← c113642] at e_1862_5
  have hr := e_1862_5
  simp only [e_1862_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 31) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9597)) := by
  have c113643 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113644 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113645 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113646 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113647 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113648 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_6 := arithEq_of_rows h (row := 1862) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_6
  simp only [← c113643, ← c113644, ← c113645] at e_1862_6
  have e_1862_7 := arithEq_of_rows h (row := 1862) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_7
  simp only [← c113646, ← c113647, k2, ← c113648] at e_1862_7
  have hr := e_1862_7
  simp only [e_1862_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 39) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9598)) := by
  have c113649 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113650 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113651 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113652 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113653 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113654 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_8 := arithEq_of_rows h (row := 1862) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_8
  simp only [← c113649, ← c113650, ← c113651] at e_1862_8
  have e_1862_9 := arithEq_of_rows h (row := 1862) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_9
  simp only [← c113652, ← c113653, k2, ← c113654] at e_1862_9
  have hr := e_1862_9
  simp only [e_1862_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 47) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9599)) := by
  have c113655 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113656 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113657 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113658 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113659 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113660 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_10 := arithEq_of_rows h (row := 1862) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_10
  simp only [← c113655, ← c113656, ← c113657] at e_1862_10
  have e_1862_11 := arithEq_of_rows h (row := 1862) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_11
  simp only [← c113658, ← c113659, k2, ← c113660] at e_1862_11
  have hr := e_1862_11
  simp only [e_1862_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 55) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9600)) := by
  have c113661 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113662 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113663 := (publicBatchWrapper86_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113664 := (publicBatchWrapper86_copies3552 a h).1
  have c113665 := (publicBatchWrapper86_copies3552 a h).2.1
  have c113666 := (publicBatchWrapper86_copies3552 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_12 := arithEq_of_rows h (row := 1862) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_12
  simp only [← c113661, ← c113662, ← c113663] at e_1862_12
  have e_1862_13 := arithEq_of_rows h (row := 1862) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_13
  simp only [← c113664, ← c113665, k2, ← c113666] at e_1862_13
  have hr := e_1862_13
  simp only [e_1862_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 63) = bselect (a (.wire 48 15)) (a (.virt 200)) (a (.virt 9601)) := by
  have c113667 := (publicBatchWrapper86_copies3552 a h).2.2.2.1
  have c113668 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.1
  have c113669 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.1
  have c113670 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.1
  have c113671 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.1
  have c113672 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_14 := arithEq_of_rows h (row := 1862) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_14
  simp only [← c113667, ← c113668, ← c113669] at e_1862_14
  have e_1862_15 := arithEq_of_rows h (row := 1862) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_15
  simp only [← c113670, ← c113671, k2, ← c113672] at e_1862_15
  have hr := e_1862_15
  simp only [e_1862_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9770)) := by
  have c113673 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.1
  have c113674 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113675 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113676 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113677 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113678 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_16 := arithEq_of_rows h (row := 1862) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_16
  simp only [← c113673, ← c113674, ← c113675] at e_1862_16
  have e_1862_17 := arithEq_of_rows h (row := 1862) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_17
  simp only [← c113676, ← c113677, k2, ← c113678] at e_1862_17
  have hr := e_1862_17
  simp only [e_1862_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1862 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9771)) := by
  have c113679 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113680 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113681 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113682 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113683 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113684 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1862_18 := arithEq_of_rows h (row := 1862) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_18
  simp only [← c113679, ← c113680, ← c113681] at e_1862_18
  have e_1862_19 := arithEq_of_rows h (row := 1862) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_19
  simp only [← c113682, ← c113683, k2, ← c113684] at e_1862_19
  have hr := e_1862_19
  simp only [e_1862_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9772)) := by
  have c113685 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113686 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113687 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113688 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113689 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113690 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_0 := arithEq_of_rows h (row := 1863) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_0
  simp only [← c113685, ← c113686, ← c113687] at e_1863_0
  have e_1863_1 := arithEq_of_rows h (row := 1863) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_1
  simp only [← c113688, ← c113689, k2, ← c113690] at e_1863_1
  have hr := e_1863_1
  simp only [e_1863_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9773)) := by
  have c113691 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113692 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113693 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113694 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113695 := (publicBatchWrapper86_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113696 := (publicBatchWrapper86_copies3553 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_2 := arithEq_of_rows h (row := 1863) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_2
  simp only [← c113691, ← c113692, ← c113693] at e_1863_2
  have e_1863_3 := arithEq_of_rows h (row := 1863) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_3
  simp only [← c113694, ← c113695, k2, ← c113696] at e_1863_3
  have hr := e_1863_3
  simp only [e_1863_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9774)) := by
  have c113697 := (publicBatchWrapper86_copies3553 a h).2.1
  have c113698 := (publicBatchWrapper86_copies3553 a h).2.2.1
  have c113699 := (publicBatchWrapper86_copies3553 a h).2.2.2.1
  have c113700 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.1
  have c113701 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.1
  have c113702 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_4 := arithEq_of_rows h (row := 1863) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_4
  simp only [← c113697, ← c113698, ← c113699] at e_1863_4
  have e_1863_5 := arithEq_of_rows h (row := 1863) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_5
  simp only [← c113700, ← c113701, k2, ← c113702] at e_1863_5
  have hr := e_1863_5
  simp only [e_1863_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9775)) := by
  have c113703 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.1
  have c113704 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.1
  have c113705 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.1
  have c113706 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113707 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113708 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_6 := arithEq_of_rows h (row := 1863) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_6
  simp only [← c113703, ← c113704, ← c113705] at e_1863_6
  have e_1863_7 := arithEq_of_rows h (row := 1863) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_7
  simp only [← c113706, ← c113707, k2, ← c113708] at e_1863_7
  have hr := e_1863_7
  simp only [e_1863_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9776)) := by
  have c113709 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113710 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113711 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113712 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113713 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113714 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_8 := arithEq_of_rows h (row := 1863) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_8
  simp only [← c113709, ← c113710, ← c113711] at e_1863_8
  have e_1863_9 := arithEq_of_rows h (row := 1863) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_9
  simp only [← c113712, ← c113713, k2, ← c113714] at e_1863_9
  have hr := e_1863_9
  simp only [e_1863_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9777)) := by
  have c113715 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113716 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113717 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113718 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113719 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113720 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_10 := arithEq_of_rows h (row := 1863) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_10
  simp only [← c113715, ← c113716, ← c113717] at e_1863_10
  have e_1863_11 := arithEq_of_rows h (row := 1863) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_11
  simp only [← c113718, ← c113719, k2, ← c113720] at e_1863_11
  have hr := e_1863_11
  simp only [e_1863_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9778)) := by
  have c113721 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113722 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113723 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113724 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113725 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113726 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_12 := arithEq_of_rows h (row := 1863) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_12
  simp only [← c113721, ← c113722, ← c113723] at e_1863_12
  have e_1863_13 := arithEq_of_rows h (row := 1863) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_13
  simp only [← c113724, ← c113725, k2, ← c113726] at e_1863_13
  have hr := e_1863_13
  simp only [e_1863_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9779)) := by
  have c113727 := (publicBatchWrapper86_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113728 := (publicBatchWrapper86_copies3554 a h).1
  have c113729 := (publicBatchWrapper86_copies3554 a h).2.1
  have c113730 := (publicBatchWrapper86_copies3554 a h).2.2.1
  have c113731 := (publicBatchWrapper86_copies3554 a h).2.2.2.1
  have c113732 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_14 := arithEq_of_rows h (row := 1863) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_14
  simp only [← c113727, ← c113728, ← c113729] at e_1863_14
  have e_1863_15 := arithEq_of_rows h (row := 1863) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_15
  simp only [← c113730, ← c113731, k2, ← c113732] at e_1863_15
  have hr := e_1863_15
  simp only [e_1863_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9780)) := by
  have c113733 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.1
  have c113734 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.1
  have c113735 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.1
  have c113736 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.1
  have c113737 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.1
  have c113738 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_16 := arithEq_of_rows h (row := 1863) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_16
  simp only [← c113733, ← c113734, ← c113735] at e_1863_16
  have e_1863_17 := arithEq_of_rows h (row := 1863) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_17
  simp only [← c113736, ← c113737, k2, ← c113738] at e_1863_17
  have hr := e_1863_17
  simp only [e_1863_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1863 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9781)) := by
  have c113739 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113740 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113741 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113742 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113743 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113744 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1863_18 := arithEq_of_rows h (row := 1863) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_18
  simp only [← c113739, ← c113740, ← c113741] at e_1863_18
  have e_1863_19 := arithEq_of_rows h (row := 1863) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_19
  simp only [← c113742, ← c113743, k2, ← c113744] at e_1863_19
  have hr := e_1863_19
  simp only [e_1863_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 7) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9782)) := by
  have c113745 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113746 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113747 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113748 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113749 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113750 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_0 := arithEq_of_rows h (row := 1864) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_0
  simp only [← c113745, ← c113746, ← c113747] at e_1864_0
  have e_1864_1 := arithEq_of_rows h (row := 1864) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_1
  simp only [← c113748, ← c113749, k2, ← c113750] at e_1864_1
  have hr := e_1864_1
  simp only [e_1864_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 15) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9783)) := by
  have c113751 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113752 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113753 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113754 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113755 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113756 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_2 := arithEq_of_rows h (row := 1864) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_2
  simp only [← c113751, ← c113752, ← c113753] at e_1864_2
  have e_1864_3 := arithEq_of_rows h (row := 1864) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_3
  simp only [← c113754, ← c113755, k2, ← c113756] at e_1864_3
  have hr := e_1864_3
  simp only [e_1864_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 23) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9784)) := by
  have c113757 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113758 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113759 := (publicBatchWrapper86_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113760 := (publicBatchWrapper86_copies3555 a h).1
  have c113761 := (publicBatchWrapper86_copies3555 a h).2.1
  have c113762 := (publicBatchWrapper86_copies3555 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_4 := arithEq_of_rows h (row := 1864) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_4
  simp only [← c113757, ← c113758, ← c113759] at e_1864_4
  have e_1864_5 := arithEq_of_rows h (row := 1864) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_5
  simp only [← c113760, ← c113761, k2, ← c113762] at e_1864_5
  have hr := e_1864_5
  simp only [e_1864_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 31) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9785)) := by
  have c113763 := (publicBatchWrapper86_copies3555 a h).2.2.2.1
  have c113764 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.1
  have c113765 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.1
  have c113766 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.1
  have c113767 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.1
  have c113768 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_6 := arithEq_of_rows h (row := 1864) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_6
  simp only [← c113763, ← c113764, ← c113765] at e_1864_6
  have e_1864_7 := arithEq_of_rows h (row := 1864) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_7
  simp only [← c113766, ← c113767, k2, ← c113768] at e_1864_7
  have hr := e_1864_7
  simp only [e_1864_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 39) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9786)) := by
  have c113769 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.1
  have c113770 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113771 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113772 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113773 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113774 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_8 := arithEq_of_rows h (row := 1864) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_8
  simp only [← c113769, ← c113770, ← c113771] at e_1864_8
  have e_1864_9 := arithEq_of_rows h (row := 1864) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_9
  simp only [← c113772, ← c113773, k2, ← c113774] at e_1864_9
  have hr := e_1864_9
  simp only [e_1864_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 47) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9787)) := by
  have c113775 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113776 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113777 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113778 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113779 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113780 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_10 := arithEq_of_rows h (row := 1864) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_10
  simp only [← c113775, ← c113776, ← c113777] at e_1864_10
  have e_1864_11 := arithEq_of_rows h (row := 1864) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_11
  simp only [← c113778, ← c113779, k2, ← c113780] at e_1864_11
  have hr := e_1864_11
  simp only [e_1864_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 55) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9788)) := by
  have c113781 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113782 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113783 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113784 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113785 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113786 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_12 := arithEq_of_rows h (row := 1864) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_12
  simp only [← c113781, ← c113782, ← c113783] at e_1864_12
  have e_1864_13 := arithEq_of_rows h (row := 1864) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_13
  simp only [← c113784, ← c113785, k2, ← c113786] at e_1864_13
  have hr := e_1864_13
  simp only [e_1864_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 63) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9789)) := by
  have c113787 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113788 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113789 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113790 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113791 := (publicBatchWrapper86_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113792 := (publicBatchWrapper86_copies3556 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_14 := arithEq_of_rows h (row := 1864) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_14
  simp only [← c113787, ← c113788, ← c113789] at e_1864_14
  have e_1864_15 := arithEq_of_rows h (row := 1864) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_15
  simp only [← c113790, ← c113791, k2, ← c113792] at e_1864_15
  have hr := e_1864_15
  simp only [e_1864_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 71) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9790)) := by
  have c113793 := (publicBatchWrapper86_copies3556 a h).2.1
  have c113794 := (publicBatchWrapper86_copies3556 a h).2.2.1
  have c113795 := (publicBatchWrapper86_copies3556 a h).2.2.2.1
  have c113796 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.1
  have c113797 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.1
  have c113798 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_16 := arithEq_of_rows h (row := 1864) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_16
  simp only [← c113793, ← c113794, ← c113795] at e_1864_16
  have e_1864_17 := arithEq_of_rows h (row := 1864) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_17
  simp only [← c113796, ← c113797, k2, ← c113798] at e_1864_17
  have hr := e_1864_17
  simp only [e_1864_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f18241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1864 79) = bselect (a (.wire 48 59)) (a (.virt 200)) (a (.virt 9791)) := by
  have c113799 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.1
  have c113800 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.1
  have c113801 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.1
  have c113802 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113803 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113804 := (publicBatchWrapper86_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1864_18 := arithEq_of_rows h (row := 1864) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_18
  simp only [← c113799, ← c113800, ← c113801] at e_1864_18
  have e_1864_19 := arithEq_of_rows h (row := 1864) (i := 19) rfl (by decide)
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
  have e_1865_0 := arithEq_of_rows h (row := 1865) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_0
  simp only [← c113805, ← c113806, ← c113807] at e_1865_0
  have e_1865_1 := arithEq_of_rows h (row := 1865) (i := 1) rfl (by decide)
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
  have e_1865_2 := arithEq_of_rows h (row := 1865) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_2
  simp only [← c113811, ← c113812, ← c113813] at e_1865_2
  have e_1865_3 := arithEq_of_rows h (row := 1865) (i := 3) rfl (by decide)
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
  have e_1865_4 := arithEq_of_rows h (row := 1865) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_4
  simp only [← c113817, ← c113818, ← c113819] at e_1865_4
  have e_1865_5 := arithEq_of_rows h (row := 1865) (i := 5) rfl (by decide)
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
  have e_1865_6 := arithEq_of_rows h (row := 1865) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_6
  simp only [← c113823, ← c113824, ← c113825] at e_1865_6
  have e_1865_7 := arithEq_of_rows h (row := 1865) (i := 7) rfl (by decide)
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
  have e_1865_8 := arithEq_of_rows h (row := 1865) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_8
  simp only [← c113829, ← c113830, ← c113831] at e_1865_8
  have e_1865_9 := arithEq_of_rows h (row := 1865) (i := 9) rfl (by decide)
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
  have e_1865_10 := arithEq_of_rows h (row := 1865) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_10
  simp only [← c113835, ← c113836, ← c113837] at e_1865_10
  have e_1865_11 := arithEq_of_rows h (row := 1865) (i := 11) rfl (by decide)
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
  have e_1865_12 := arithEq_of_rows h (row := 1865) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_12
  simp only [← c113841, ← c113842, ← c113843] at e_1865_12
  have e_1865_13 := arithEq_of_rows h (row := 1865) (i := 13) rfl (by decide)
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
  have e_1865_14 := arithEq_of_rows h (row := 1865) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_14
  simp only [← c113847, ← c113848, ← c113849] at e_1865_14
  have e_1865_15 := arithEq_of_rows h (row := 1865) (i := 15) rfl (by decide)
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
  have e_1865_16 := arithEq_of_rows h (row := 1865) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_16
  simp only [← c113853, ← c113854, ← c113855] at e_1865_16
  have e_1865_17 := arithEq_of_rows h (row := 1865) (i := 17) rfl (by decide)
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
  have e_1865_18 := arithEq_of_rows h (row := 1865) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_18
  simp only [← c113859, ← c113860, ← c113861] at e_1865_18
  have e_1865_19 := arithEq_of_rows h (row := 1865) (i := 19) rfl (by decide)
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
  have e_1866_0 := arithEq_of_rows h (row := 1866) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_0
  simp only [← c113865, ← c113866, ← c113867] at e_1866_0
  have e_1866_1 := arithEq_of_rows h (row := 1866) (i := 1) rfl (by decide)
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
  have e_1866_2 := arithEq_of_rows h (row := 1866) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_2
  simp only [← c113871, ← c113872, ← c113873] at e_1866_2
  have e_1866_3 := arithEq_of_rows h (row := 1866) (i := 3) rfl (by decide)
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
  have e_1866_4 := arithEq_of_rows h (row := 1866) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_4
  simp only [← c113877, ← c113878, ← c113879] at e_1866_4
  have e_1866_5 := arithEq_of_rows h (row := 1866) (i := 5) rfl (by decide)
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
  have e_1866_6 := arithEq_of_rows h (row := 1866) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_6
  simp only [← c113883, ← c113884, ← c113885] at e_1866_6
  have e_1866_7 := arithEq_of_rows h (row := 1866) (i := 7) rfl (by decide)
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
  have e_1866_8 := arithEq_of_rows h (row := 1866) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_8
  simp only [← c113889, ← c113890, ← c113891] at e_1866_8
  have e_1866_9 := arithEq_of_rows h (row := 1866) (i := 9) rfl (by decide)
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
  have e_1866_10 := arithEq_of_rows h (row := 1866) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_10
  simp only [← c113895, ← c113896, ← c113897] at e_1866_10
  have e_1866_11 := arithEq_of_rows h (row := 1866) (i := 11) rfl (by decide)
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
  have e_1866_12 := arithEq_of_rows h (row := 1866) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_12
  simp only [← c113901, ← c113902, ← c113903] at e_1866_12
  have e_1866_13 := arithEq_of_rows h (row := 1866) (i := 13) rfl (by decide)
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
  have e_1866_14 := arithEq_of_rows h (row := 1866) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_14
  simp only [← c113907, ← c113908, ← c113909] at e_1866_14
  have e_1866_15 := arithEq_of_rows h (row := 1866) (i := 15) rfl (by decide)
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
  have e_1866_16 := arithEq_of_rows h (row := 1866) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_16
  simp only [← c113913, ← c113914, ← c113915] at e_1866_16
  have e_1866_17 := arithEq_of_rows h (row := 1866) (i := 17) rfl (by decide)
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
  have e_1866_18 := arithEq_of_rows h (row := 1866) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_18
  simp only [← c113919, ← c113920, ← c113921] at e_1866_18
  have e_1866_19 := arithEq_of_rows h (row := 1866) (i := 19) rfl (by decide)
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
  have e_1867_0 := arithEq_of_rows h (row := 1867) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_0
  simp only [← c113925, ← c113926, ← c113927] at e_1867_0
  have e_1867_1 := arithEq_of_rows h (row := 1867) (i := 1) rfl (by decide)
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
  have e_1867_2 := arithEq_of_rows h (row := 1867) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_2
  simp only [← c113931, ← c113932, ← c113933] at e_1867_2
  have e_1867_3 := arithEq_of_rows h (row := 1867) (i := 3) rfl (by decide)
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
  have e_1867_4 := arithEq_of_rows h (row := 1867) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_4
  simp only [← c113937, ← c113938, ← c113939] at e_1867_4
  have e_1867_5 := arithEq_of_rows h (row := 1867) (i := 5) rfl (by decide)
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
  have e_1867_6 := arithEq_of_rows h (row := 1867) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_6
  simp only [← c113943, ← c113944, ← c113945] at e_1867_6
  have e_1867_7 := arithEq_of_rows h (row := 1867) (i := 7) rfl (by decide)
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
  have e_1867_8 := arithEq_of_rows h (row := 1867) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_8
  simp only [← c113949, ← c113950, ← c113951] at e_1867_8
  have e_1867_9 := arithEq_of_rows h (row := 1867) (i := 9) rfl (by decide)
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
  have e_1867_10 := arithEq_of_rows h (row := 1867) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_10
  simp only [← c113955, ← c113956, ← c113957] at e_1867_10
  have e_1867_11 := arithEq_of_rows h (row := 1867) (i := 11) rfl (by decide)
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
  have e_1867_12 := arithEq_of_rows h (row := 1867) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_12
  simp only [← c113961, ← c113962, ← c113963] at e_1867_12
  have e_1867_13 := arithEq_of_rows h (row := 1867) (i := 13) rfl (by decide)
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
  have e_1867_14 := arithEq_of_rows h (row := 1867) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_14
  simp only [← c113967, ← c113968, ← c113969] at e_1867_14
  have e_1867_15 := arithEq_of_rows h (row := 1867) (i := 15) rfl (by decide)
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
  have e_1867_16 := arithEq_of_rows h (row := 1867) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_16
  simp only [← c113973, ← c113974, ← c113975] at e_1867_16
  have e_1867_17 := arithEq_of_rows h (row := 1867) (i := 17) rfl (by decide)
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
  have e_1867_18 := arithEq_of_rows h (row := 1867) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_18
  simp only [← c113979, ← c113980, ← c113981] at e_1867_18
  have e_1867_19 := arithEq_of_rows h (row := 1867) (i := 19) rfl (by decide)
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
  have e_1868_0 := arithEq_of_rows h (row := 1868) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_0
  simp only [← c113985, ← c113986, ← c113987] at e_1868_0
  have e_1868_1 := arithEq_of_rows h (row := 1868) (i := 1) rfl (by decide)
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
  have e_1868_2 := arithEq_of_rows h (row := 1868) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_2
  simp only [← c113991, ← c113992, ← c113993] at e_1868_2
  have e_1868_3 := arithEq_of_rows h (row := 1868) (i := 3) rfl (by decide)
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
  have e_1868_4 := arithEq_of_rows h (row := 1868) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_4
  simp only [← c113997, ← c113998, ← c113999] at e_1868_4
  have e_1868_5 := arithEq_of_rows h (row := 1868) (i := 5) rfl (by decide)
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
  have e_1868_6 := arithEq_of_rows h (row := 1868) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_6
  simp only [← c114003, ← c114004, ← c114005] at e_1868_6
  have e_1868_7 := arithEq_of_rows h (row := 1868) (i := 7) rfl (by decide)
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
  have e_1868_8 := arithEq_of_rows h (row := 1868) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_8
  simp only [← c114009, ← c114010, ← c114011] at e_1868_8
  have e_1868_9 := arithEq_of_rows h (row := 1868) (i := 9) rfl (by decide)
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
  have e_1868_10 := arithEq_of_rows h (row := 1868) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_10
  simp only [← c114015, ← c114016, ← c114017] at e_1868_10
  have e_1868_11 := arithEq_of_rows h (row := 1868) (i := 11) rfl (by decide)
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
  have e_1868_12 := arithEq_of_rows h (row := 1868) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_12
  simp only [← c114021, ← c114022, ← c114023] at e_1868_12
  have e_1868_13 := arithEq_of_rows h (row := 1868) (i := 13) rfl (by decide)
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
  have e_1868_14 := arithEq_of_rows h (row := 1868) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_14
  simp only [← c114027, ← c114028, ← c114029] at e_1868_14
  have e_1868_15 := arithEq_of_rows h (row := 1868) (i := 15) rfl (by decide)
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
  have e_1868_16 := arithEq_of_rows h (row := 1868) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_16
  simp only [← c114033, ← c114034, ← c114035] at e_1868_16
  have e_1868_17 := arithEq_of_rows h (row := 1868) (i := 17) rfl (by decide)
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
  have e_1868_18 := arithEq_of_rows h (row := 1868) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_18
  simp only [← c114039, ← c114040, ← c114041] at e_1868_18
  have e_1868_19 := arithEq_of_rows h (row := 1868) (i := 19) rfl (by decide)
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
  have e_1869_0 := arithEq_of_rows h (row := 1869) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_0
  simp only [← c114045, ← c114046, ← c114047] at e_1869_0
  have e_1869_1 := arithEq_of_rows h (row := 1869) (i := 1) rfl (by decide)
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
  have e_1869_2 := arithEq_of_rows h (row := 1869) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_2
  simp only [← c114051, ← c114052, ← c114053] at e_1869_2
  have e_1869_3 := arithEq_of_rows h (row := 1869) (i := 3) rfl (by decide)
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
  have e_1869_4 := arithEq_of_rows h (row := 1869) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_4
  simp only [← c114057, ← c114058, ← c114059] at e_1869_4
  have e_1869_5 := arithEq_of_rows h (row := 1869) (i := 5) rfl (by decide)
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
  have e_1869_6 := arithEq_of_rows h (row := 1869) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_6
  simp only [← c114063, ← c114064, ← c114065] at e_1869_6
  have e_1869_7 := arithEq_of_rows h (row := 1869) (i := 7) rfl (by decide)
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
  have e_1869_8 := arithEq_of_rows h (row := 1869) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_8
  simp only [← c114069, ← c114070, ← c114071] at e_1869_8
  have e_1869_9 := arithEq_of_rows h (row := 1869) (i := 9) rfl (by decide)
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
  have e_1869_10 := arithEq_of_rows h (row := 1869) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_10
  simp only [← c114075, ← c114076, ← c114077] at e_1869_10
  have e_1869_11 := arithEq_of_rows h (row := 1869) (i := 11) rfl (by decide)
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
  have e_1869_12 := arithEq_of_rows h (row := 1869) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_12
  simp only [← c114081, ← c114082, ← c114083] at e_1869_12
  have e_1869_13 := arithEq_of_rows h (row := 1869) (i := 13) rfl (by decide)
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
  have e_1869_14 := arithEq_of_rows h (row := 1869) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_14
  simp only [← c114087, ← c114088, ← c114089] at e_1869_14
  have e_1869_15 := arithEq_of_rows h (row := 1869) (i := 15) rfl (by decide)
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
  have e_1869_16 := arithEq_of_rows h (row := 1869) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_16
  simp only [← c114093, ← c114094, ← c114095] at e_1869_16
  have e_1869_17 := arithEq_of_rows h (row := 1869) (i := 17) rfl (by decide)
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
  have e_1869_18 := arithEq_of_rows h (row := 1869) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_18
  simp only [← c114099, ← c114100, ← c114101] at e_1869_18
  have e_1869_19 := arithEq_of_rows h (row := 1869) (i := 19) rfl (by decide)
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
  have e_1870_0 := arithEq_of_rows h (row := 1870) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_0
  simp only [← c114105, ← c114106, ← c114107] at e_1870_0
  have e_1870_1 := arithEq_of_rows h (row := 1870) (i := 1) rfl (by decide)
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
  have e_1870_2 := arithEq_of_rows h (row := 1870) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_2
  simp only [← c114111, ← c114112, ← c114113] at e_1870_2
  have e_1870_3 := arithEq_of_rows h (row := 1870) (i := 3) rfl (by decide)
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
  have e_1870_4 := arithEq_of_rows h (row := 1870) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_4
  simp only [← c114117, ← c114118, ← c114119] at e_1870_4
  have e_1870_5 := arithEq_of_rows h (row := 1870) (i := 5) rfl (by decide)
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
  have e_1870_6 := arithEq_of_rows h (row := 1870) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_6
  simp only [← c114123, ← c114124, ← c114125] at e_1870_6
  have e_1870_7 := arithEq_of_rows h (row := 1870) (i := 7) rfl (by decide)
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
  have e_1870_8 := arithEq_of_rows h (row := 1870) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_8
  simp only [← c114129, ← c114130, ← c114131] at e_1870_8
  have e_1870_9 := arithEq_of_rows h (row := 1870) (i := 9) rfl (by decide)
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
  have e_1870_10 := arithEq_of_rows h (row := 1870) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_10
  simp only [← c114135, ← c114136, ← c114137] at e_1870_10
  have e_1870_11 := arithEq_of_rows h (row := 1870) (i := 11) rfl (by decide)
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
  have e_1870_12 := arithEq_of_rows h (row := 1870) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_12
  simp only [← c114141, ← c114142, ← c114143] at e_1870_12
  have e_1870_13 := arithEq_of_rows h (row := 1870) (i := 13) rfl (by decide)
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
  have e_1870_14 := arithEq_of_rows h (row := 1870) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_14
  simp only [← c114147, ← c114148, ← c114149] at e_1870_14
  have e_1870_15 := arithEq_of_rows h (row := 1870) (i := 15) rfl (by decide)
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
  have e_1870_16 := arithEq_of_rows h (row := 1870) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_16
  simp only [← c114153, ← c114154, ← c114155] at e_1870_16
  have e_1870_17 := arithEq_of_rows h (row := 1870) (i := 17) rfl (by decide)
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
  have e_1870_18 := arithEq_of_rows h (row := 1870) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_18
  simp only [← c114159, ← c114160, ← c114161] at e_1870_18
  have e_1870_19 := arithEq_of_rows h (row := 1870) (i := 19) rfl (by decide)
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
  have e_1871_0 := arithEq_of_rows h (row := 1871) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_0
  simp only [← c114165, ← c114166, ← c114167] at e_1871_0
  have e_1871_1 := arithEq_of_rows h (row := 1871) (i := 1) rfl (by decide)
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
  have e_1871_2 := arithEq_of_rows h (row := 1871) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_2
  simp only [← c114171, ← c114172, ← c114173] at e_1871_2
  have e_1871_3 := arithEq_of_rows h (row := 1871) (i := 3) rfl (by decide)
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
  have e_1871_4 := arithEq_of_rows h (row := 1871) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_4
  simp only [← c114177, ← c114178, ← c114179] at e_1871_4
  have e_1871_5 := arithEq_of_rows h (row := 1871) (i := 5) rfl (by decide)
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
  have e_1871_6 := arithEq_of_rows h (row := 1871) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_6
  simp only [← c114183, ← c114184, ← c114185] at e_1871_6
  have e_1871_7 := arithEq_of_rows h (row := 1871) (i := 7) rfl (by decide)
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
  have e_1871_8 := arithEq_of_rows h (row := 1871) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_8
  simp only [← c114189, ← c114190, ← c114191] at e_1871_8
  have e_1871_9 := arithEq_of_rows h (row := 1871) (i := 9) rfl (by decide)
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
  have e_1871_10 := arithEq_of_rows h (row := 1871) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_10
  simp only [← c114195, ← c114196, ← c114197] at e_1871_10
  have e_1871_11 := arithEq_of_rows h (row := 1871) (i := 11) rfl (by decide)
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
  have e_1871_12 := arithEq_of_rows h (row := 1871) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_12
  simp only [← c114201, ← c114202, ← c114203] at e_1871_12
  have e_1871_13 := arithEq_of_rows h (row := 1871) (i := 13) rfl (by decide)
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
  have e_1871_14 := arithEq_of_rows h (row := 1871) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_14
  simp only [← c114207, ← c114208, ← c114209] at e_1871_14
  have e_1871_15 := arithEq_of_rows h (row := 1871) (i := 15) rfl (by decide)
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
  have e_1871_16 := arithEq_of_rows h (row := 1871) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_16
  simp only [← c114213, ← c114214, ← c114215] at e_1871_16
  have e_1871_17 := arithEq_of_rows h (row := 1871) (i := 17) rfl (by decide)
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
  have e_1871_18 := arithEq_of_rows h (row := 1871) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_18
  simp only [← c114219, ← c114220, ← c114221] at e_1871_18
  have e_1871_19 := arithEq_of_rows h (row := 1871) (i := 19) rfl (by decide)
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
  have e_1872_0 := arithEq_of_rows h (row := 1872) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_0
  simp only [← c114225, ← c114226, ← c114227] at e_1872_0
  have e_1872_1 := arithEq_of_rows h (row := 1872) (i := 1) rfl (by decide)
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
  have e_1872_2 := arithEq_of_rows h (row := 1872) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_2
  simp only [← c114231, ← c114232, ← c114233] at e_1872_2
  have e_1872_3 := arithEq_of_rows h (row := 1872) (i := 3) rfl (by decide)
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
  have e_1872_4 := arithEq_of_rows h (row := 1872) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_4
  simp only [← c114237, ← c114238, ← c114239] at e_1872_4
  have e_1872_5 := arithEq_of_rows h (row := 1872) (i := 5) rfl (by decide)
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
  have e_1872_6 := arithEq_of_rows h (row := 1872) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_6
  simp only [← c114243, ← c114244, ← c114245] at e_1872_6
  have e_1872_7 := arithEq_of_rows h (row := 1872) (i := 7) rfl (by decide)
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
  have e_1872_8 := arithEq_of_rows h (row := 1872) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_8
  simp only [← c114249, ← c114250, ← c114251] at e_1872_8
  have e_1872_9 := arithEq_of_rows h (row := 1872) (i := 9) rfl (by decide)
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
  have e_1872_10 := arithEq_of_rows h (row := 1872) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_10
  simp only [← c114255, ← c114256, ← c114257] at e_1872_10
  have e_1872_11 := arithEq_of_rows h (row := 1872) (i := 11) rfl (by decide)
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
  have e_1872_12 := arithEq_of_rows h (row := 1872) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_12
  simp only [← c114261, ← c114262, ← c114263] at e_1872_12
  have e_1872_13 := arithEq_of_rows h (row := 1872) (i := 13) rfl (by decide)
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
  have e_1872_14 := arithEq_of_rows h (row := 1872) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_14
  simp only [← c114267, ← c114268, ← c114269] at e_1872_14
  have e_1872_15 := arithEq_of_rows h (row := 1872) (i := 15) rfl (by decide)
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
  have e_1872_16 := arithEq_of_rows h (row := 1872) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_16
  simp only [← c114273, ← c114274, ← c114275] at e_1872_16
  have e_1872_17 := arithEq_of_rows h (row := 1872) (i := 17) rfl (by decide)
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
  have e_1872_18 := arithEq_of_rows h (row := 1872) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_18
  simp only [← c114279, ← c114280, ← c114281] at e_1872_18
  have e_1872_19 := arithEq_of_rows h (row := 1872) (i := 19) rfl (by decide)
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
  have e_1873_0 := arithEq_of_rows h (row := 1873) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_0
  simp only [← c114285, ← c114286, ← c114287] at e_1873_0
  have e_1873_1 := arithEq_of_rows h (row := 1873) (i := 1) rfl (by decide)
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
  have e_1873_2 := arithEq_of_rows h (row := 1873) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_2
  simp only [← c114291, ← c114292, ← c114293] at e_1873_2
  have e_1873_3 := arithEq_of_rows h (row := 1873) (i := 3) rfl (by decide)
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
  have e_1873_4 := arithEq_of_rows h (row := 1873) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_4
  simp only [← c114297, ← c114298, ← c114299] at e_1873_4
  have e_1873_5 := arithEq_of_rows h (row := 1873) (i := 5) rfl (by decide)
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
  have e_1873_6 := arithEq_of_rows h (row := 1873) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_6
  simp only [← c114303, ← c114304, ← c114305] at e_1873_6
  have e_1873_7 := arithEq_of_rows h (row := 1873) (i := 7) rfl (by decide)
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
  have e_1873_8 := arithEq_of_rows h (row := 1873) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_8
  simp only [← c114309, ← c114310, ← c114311] at e_1873_8
  have e_1873_9 := arithEq_of_rows h (row := 1873) (i := 9) rfl (by decide)
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
  have e_1873_10 := arithEq_of_rows h (row := 1873) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_10
  simp only [← c114315, ← c114316, ← c114317] at e_1873_10
  have e_1873_11 := arithEq_of_rows h (row := 1873) (i := 11) rfl (by decide)
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
  have e_1873_12 := arithEq_of_rows h (row := 1873) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_12
  simp only [← c114321, ← c114322, ← c114323] at e_1873_12
  have e_1873_13 := arithEq_of_rows h (row := 1873) (i := 13) rfl (by decide)
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
  have e_1873_14 := arithEq_of_rows h (row := 1873) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_14
  simp only [← c114327, ← c114328, ← c114329] at e_1873_14
  have e_1873_15 := arithEq_of_rows h (row := 1873) (i := 15) rfl (by decide)
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
  have e_1873_16 := arithEq_of_rows h (row := 1873) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_16
  simp only [← c114333, ← c114334, ← c114335] at e_1873_16
  have e_1873_17 := arithEq_of_rows h (row := 1873) (i := 17) rfl (by decide)
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
  have e_1873_18 := arithEq_of_rows h (row := 1873) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_18
  simp only [← c114339, ← c114340, ← c114341] at e_1873_18
  have e_1873_19 := arithEq_of_rows h (row := 1873) (i := 19) rfl (by decide)
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
  have e_1874_0 := arithEq_of_rows h (row := 1874) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_0
  simp only [← c114345, ← c114346, ← c114347] at e_1874_0
  have e_1874_1 := arithEq_of_rows h (row := 1874) (i := 1) rfl (by decide)
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
  have e_1874_2 := arithEq_of_rows h (row := 1874) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_2
  simp only [← c114351, ← c114352, ← c114353] at e_1874_2
  have e_1874_3 := arithEq_of_rows h (row := 1874) (i := 3) rfl (by decide)
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
  have e_1874_4 := arithEq_of_rows h (row := 1874) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_4
  simp only [← c114357, ← c114358, ← c114359] at e_1874_4
  have e_1874_5 := arithEq_of_rows h (row := 1874) (i := 5) rfl (by decide)
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
  have e_1874_6 := arithEq_of_rows h (row := 1874) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_6
  simp only [← c114363, ← c114364, ← c114365] at e_1874_6
  have e_1874_7 := arithEq_of_rows h (row := 1874) (i := 7) rfl (by decide)
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
  have e_1874_8 := arithEq_of_rows h (row := 1874) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_8
  simp only [← c114369, ← c114370, ← c114371] at e_1874_8
  have e_1874_9 := arithEq_of_rows h (row := 1874) (i := 9) rfl (by decide)
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
  have e_1874_10 := arithEq_of_rows h (row := 1874) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_10
  simp only [← c114375, ← c114376, ← c114377] at e_1874_10
  have e_1874_11 := arithEq_of_rows h (row := 1874) (i := 11) rfl (by decide)
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
  have e_1874_12 := arithEq_of_rows h (row := 1874) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_12
  simp only [← c114381, ← c114382, ← c114383] at e_1874_12
  have e_1874_13 := arithEq_of_rows h (row := 1874) (i := 13) rfl (by decide)
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
  have e_1874_14 := arithEq_of_rows h (row := 1874) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_14
  simp only [← c114387, ← c114388, ← c114389] at e_1874_14
  have e_1874_15 := arithEq_of_rows h (row := 1874) (i := 15) rfl (by decide)
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
  have e_1874_16 := arithEq_of_rows h (row := 1874) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_16
  simp only [← c114393, ← c114394, ← c114395] at e_1874_16
  have e_1874_17 := arithEq_of_rows h (row := 1874) (i := 17) rfl (by decide)
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
  have e_1874_18 := arithEq_of_rows h (row := 1874) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_18
  simp only [← c114399, ← c114400, ← c114401] at e_1874_18
  have e_1874_19 := arithEq_of_rows h (row := 1874) (i := 19) rfl (by decide)
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
  have e_1875_0 := arithEq_of_rows h (row := 1875) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_0
  simp only [← c114405, ← c114406, ← c114407] at e_1875_0
  have e_1875_1 := arithEq_of_rows h (row := 1875) (i := 1) rfl (by decide)
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
  have e_1875_2 := arithEq_of_rows h (row := 1875) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_2
  simp only [← c114411, ← c114412, ← c114413] at e_1875_2
  have e_1875_3 := arithEq_of_rows h (row := 1875) (i := 3) rfl (by decide)
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
  have e_1875_4 := arithEq_of_rows h (row := 1875) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_4
  simp only [← c114417, ← c114418, ← c114419] at e_1875_4
  have e_1875_5 := arithEq_of_rows h (row := 1875) (i := 5) rfl (by decide)
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
  have e_1875_6 := arithEq_of_rows h (row := 1875) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_6
  simp only [← c114423, ← c114424, ← c114425] at e_1875_6
  have e_1875_7 := arithEq_of_rows h (row := 1875) (i := 7) rfl (by decide)
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
  have e_1875_8 := arithEq_of_rows h (row := 1875) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_8
  simp only [← c114429, ← c114430, ← c114431] at e_1875_8
  have e_1875_9 := arithEq_of_rows h (row := 1875) (i := 9) rfl (by decide)
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
  have e_1875_10 := arithEq_of_rows h (row := 1875) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_10
  simp only [← c114435, ← c114436, ← c114437] at e_1875_10
  have e_1875_11 := arithEq_of_rows h (row := 1875) (i := 11) rfl (by decide)
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
  have e_1875_12 := arithEq_of_rows h (row := 1875) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_12
  simp only [← c114441, ← c114442, ← c114443] at e_1875_12
  have e_1875_13 := arithEq_of_rows h (row := 1875) (i := 13) rfl (by decide)
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
  have e_1875_14 := arithEq_of_rows h (row := 1875) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_14
  simp only [← c114447, ← c114448, ← c114449] at e_1875_14
  have e_1875_15 := arithEq_of_rows h (row := 1875) (i := 15) rfl (by decide)
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
  have e_1875_16 := arithEq_of_rows h (row := 1875) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_16
  simp only [← c114453, ← c114454, ← c114455] at e_1875_16
  have e_1875_17 := arithEq_of_rows h (row := 1875) (i := 17) rfl (by decide)
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
  have e_1875_18 := arithEq_of_rows h (row := 1875) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_18
  simp only [← c114459, ← c114460, ← c114461] at e_1875_18
  have e_1875_19 := arithEq_of_rows h (row := 1875) (i := 19) rfl (by decide)
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
  have e_1876_0 := arithEq_of_rows h (row := 1876) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_0
  simp only [← c114465, ← c114466, ← c114467] at e_1876_0
  have e_1876_1 := arithEq_of_rows h (row := 1876) (i := 1) rfl (by decide)
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
  have e_1876_2 := arithEq_of_rows h (row := 1876) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_2
  simp only [← c114471, ← c114472, ← c114473] at e_1876_2
  have e_1876_3 := arithEq_of_rows h (row := 1876) (i := 3) rfl (by decide)
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
  have e_1876_4 := arithEq_of_rows h (row := 1876) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_4
  simp only [← c114477, ← c114478, ← c114479] at e_1876_4
  have e_1876_5 := arithEq_of_rows h (row := 1876) (i := 5) rfl (by decide)
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
  have e_1876_6 := arithEq_of_rows h (row := 1876) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_6
  simp only [← c114483, ← c114484, ← c114485] at e_1876_6
  have e_1876_7 := arithEq_of_rows h (row := 1876) (i := 7) rfl (by decide)
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
  have e_1876_8 := arithEq_of_rows h (row := 1876) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_8
  simp only [← c114489, ← c114490, ← c114491] at e_1876_8
  have e_1876_9 := arithEq_of_rows h (row := 1876) (i := 9) rfl (by decide)
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
  have e_1876_10 := arithEq_of_rows h (row := 1876) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_10
  simp only [← c114495, ← c114496, ← c114497] at e_1876_10
  have e_1876_11 := arithEq_of_rows h (row := 1876) (i := 11) rfl (by decide)
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
  have e_1876_12 := arithEq_of_rows h (row := 1876) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_12
  simp only [← c114501, ← c114502, ← c114503] at e_1876_12
  have e_1876_13 := arithEq_of_rows h (row := 1876) (i := 13) rfl (by decide)
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
  have e_1876_14 := arithEq_of_rows h (row := 1876) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_14
  simp only [← c114507, ← c114508, ← c114509] at e_1876_14
  have e_1876_15 := arithEq_of_rows h (row := 1876) (i := 15) rfl (by decide)
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
  have e_1876_16 := arithEq_of_rows h (row := 1876) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_16
  simp only [← c114513, ← c114514, ← c114515] at e_1876_16
  have e_1876_17 := arithEq_of_rows h (row := 1876) (i := 17) rfl (by decide)
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
  have e_1876_18 := arithEq_of_rows h (row := 1876) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_18
  simp only [← c114519, ← c114520, ← c114521] at e_1876_18
  have e_1876_19 := arithEq_of_rows h (row := 1876) (i := 19) rfl (by decide)
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
  have e_1877_0 := arithEq_of_rows h (row := 1877) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_0
  simp only [← c114525, ← c114526, ← c114527] at e_1877_0
  have e_1877_1 := arithEq_of_rows h (row := 1877) (i := 1) rfl (by decide)
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
  have e_1877_2 := arithEq_of_rows h (row := 1877) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_2
  simp only [← c114531, ← c114532, ← c114533] at e_1877_2
  have e_1877_3 := arithEq_of_rows h (row := 1877) (i := 3) rfl (by decide)
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
  have e_1877_4 := arithEq_of_rows h (row := 1877) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_4
  simp only [← c114537, ← c114538, ← c114539] at e_1877_4
  have e_1877_5 := arithEq_of_rows h (row := 1877) (i := 5) rfl (by decide)
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
  have e_1877_6 := arithEq_of_rows h (row := 1877) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_6
  simp only [← c114543, ← c114544, ← c114545] at e_1877_6
  have e_1877_7 := arithEq_of_rows h (row := 1877) (i := 7) rfl (by decide)
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
  have e_1877_8 := arithEq_of_rows h (row := 1877) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_8
  simp only [← c114549, ← c114550, ← c114551] at e_1877_8
  have e_1877_9 := arithEq_of_rows h (row := 1877) (i := 9) rfl (by decide)
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
  have e_1877_10 := arithEq_of_rows h (row := 1877) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_10
  simp only [← c114555, ← c114556, ← c114557] at e_1877_10
  have e_1877_11 := arithEq_of_rows h (row := 1877) (i := 11) rfl (by decide)
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
  have e_1877_12 := arithEq_of_rows h (row := 1877) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_12
  simp only [← c114561, ← c114562, ← c114563] at e_1877_12
  have e_1877_13 := arithEq_of_rows h (row := 1877) (i := 13) rfl (by decide)
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
  have e_1877_14 := arithEq_of_rows h (row := 1877) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_14
  simp only [← c114567, ← c114568, ← c114569] at e_1877_14
  have e_1877_15 := arithEq_of_rows h (row := 1877) (i := 15) rfl (by decide)
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
  have e_1877_16 := arithEq_of_rows h (row := 1877) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_16
  simp only [← c114573, ← c114574, ← c114575] at e_1877_16
  have e_1877_17 := arithEq_of_rows h (row := 1877) (i := 17) rfl (by decide)
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
  have e_1877_18 := arithEq_of_rows h (row := 1877) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_18
  simp only [← c114579, ← c114580, ← c114581] at e_1877_18
  have e_1877_19 := arithEq_of_rows h (row := 1877) (i := 19) rfl (by decide)
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
  have e_1878_0 := arithEq_of_rows h (row := 1878) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_0
  simp only [← c114585, ← c114586, ← c114587] at e_1878_0
  have e_1878_1 := arithEq_of_rows h (row := 1878) (i := 1) rfl (by decide)
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
  have e_1878_2 := arithEq_of_rows h (row := 1878) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_2
  simp only [← c114591, ← c114592, ← c114593] at e_1878_2
  have e_1878_3 := arithEq_of_rows h (row := 1878) (i := 3) rfl (by decide)
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
  have e_1878_4 := arithEq_of_rows h (row := 1878) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_4
  simp only [← c114597, ← c114598, ← c114599] at e_1878_4
  have e_1878_5 := arithEq_of_rows h (row := 1878) (i := 5) rfl (by decide)
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
  have e_1878_6 := arithEq_of_rows h (row := 1878) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_6
  simp only [← c114603, ← c114604, ← c114605] at e_1878_6
  have e_1878_7 := arithEq_of_rows h (row := 1878) (i := 7) rfl (by decide)
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
  have e_1878_8 := arithEq_of_rows h (row := 1878) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_8
  simp only [← c114609, ← c114610, ← c114611] at e_1878_8
  have e_1878_9 := arithEq_of_rows h (row := 1878) (i := 9) rfl (by decide)
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
  have e_1878_10 := arithEq_of_rows h (row := 1878) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_10
  simp only [← c114615, ← c114616, ← c114617] at e_1878_10
  have e_1878_11 := arithEq_of_rows h (row := 1878) (i := 11) rfl (by decide)
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
  have e_1878_12 := arithEq_of_rows h (row := 1878) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_12
  simp only [← c114621, ← c114622, ← c114623] at e_1878_12
  have e_1878_13 := arithEq_of_rows h (row := 1878) (i := 13) rfl (by decide)
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
  have e_1878_14 := arithEq_of_rows h (row := 1878) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_14
  simp only [← c114627, ← c114628, ← c114629] at e_1878_14
  have e_1878_15 := arithEq_of_rows h (row := 1878) (i := 15) rfl (by decide)
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
  have e_1878_16 := arithEq_of_rows h (row := 1878) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_16
  simp only [← c114633, ← c114634, ← c114635] at e_1878_16
  have e_1878_17 := arithEq_of_rows h (row := 1878) (i := 17) rfl (by decide)
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
  have e_1878_18 := arithEq_of_rows h (row := 1878) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_18
  simp only [← c114639, ← c114640, ← c114641] at e_1878_18
  have e_1878_19 := arithEq_of_rows h (row := 1878) (i := 19) rfl (by decide)
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
  have e_1879_0 := arithEq_of_rows h (row := 1879) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_0
  simp only [← c114645, ← c114646, ← c114647] at e_1879_0
  have e_1879_1 := arithEq_of_rows h (row := 1879) (i := 1) rfl (by decide)
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
  have e_1879_2 := arithEq_of_rows h (row := 1879) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_2
  simp only [← c114651, ← c114652, ← c114653] at e_1879_2
  have e_1879_3 := arithEq_of_rows h (row := 1879) (i := 3) rfl (by decide)
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
  have e_1879_4 := arithEq_of_rows h (row := 1879) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_4
  simp only [← c114657, ← c114658, ← c114659] at e_1879_4
  have e_1879_5 := arithEq_of_rows h (row := 1879) (i := 5) rfl (by decide)
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
  have e_1879_6 := arithEq_of_rows h (row := 1879) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_6
  simp only [← c114663, ← c114664, ← c114665] at e_1879_6
  have e_1879_7 := arithEq_of_rows h (row := 1879) (i := 7) rfl (by decide)
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
  have e_1879_8 := arithEq_of_rows h (row := 1879) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_8
  simp only [← c114669, ← c114670, ← c114671] at e_1879_8
  have e_1879_9 := arithEq_of_rows h (row := 1879) (i := 9) rfl (by decide)
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
  have e_1879_10 := arithEq_of_rows h (row := 1879) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_10
  simp only [← c114675, ← c114676, ← c114677] at e_1879_10
  have e_1879_11 := arithEq_of_rows h (row := 1879) (i := 11) rfl (by decide)
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
  have e_1879_12 := arithEq_of_rows h (row := 1879) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_12
  simp only [← c114681, ← c114682, ← c114683] at e_1879_12
  have e_1879_13 := arithEq_of_rows h (row := 1879) (i := 13) rfl (by decide)
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
  have e_1879_14 := arithEq_of_rows h (row := 1879) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_14
  simp only [← c114687, ← c114688, ← c114689] at e_1879_14
  have e_1879_15 := arithEq_of_rows h (row := 1879) (i := 15) rfl (by decide)
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
  have e_1879_16 := arithEq_of_rows h (row := 1879) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_16
  simp only [← c114693, ← c114694, ← c114695] at e_1879_16
  have e_1879_17 := arithEq_of_rows h (row := 1879) (i := 17) rfl (by decide)
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
  have e_1879_18 := arithEq_of_rows h (row := 1879) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_18
  simp only [← c114699, ← c114700, ← c114701] at e_1879_18
  have e_1879_19 := arithEq_of_rows h (row := 1879) (i := 19) rfl (by decide)
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
  have e_1880_0 := arithEq_of_rows h (row := 1880) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_0
  simp only [← c114705, ← c114706, ← c114707] at e_1880_0
  have e_1880_1 := arithEq_of_rows h (row := 1880) (i := 1) rfl (by decide)
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
  have e_1880_2 := arithEq_of_rows h (row := 1880) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_2
  simp only [← c114711, ← c114712, ← c114713] at e_1880_2
  have e_1880_3 := arithEq_of_rows h (row := 1880) (i := 3) rfl (by decide)
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
  have e_1880_4 := arithEq_of_rows h (row := 1880) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_4
  simp only [← c114717, ← c114718, ← c114719] at e_1880_4
  have e_1880_5 := arithEq_of_rows h (row := 1880) (i := 5) rfl (by decide)
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
  have e_1880_6 := arithEq_of_rows h (row := 1880) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_6
  simp only [← c114723, ← c114724, ← c114725] at e_1880_6
  have e_1880_7 := arithEq_of_rows h (row := 1880) (i := 7) rfl (by decide)
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
  have e_1880_8 := arithEq_of_rows h (row := 1880) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_8
  simp only [← c114729, ← c114730, ← c114731] at e_1880_8
  have e_1880_9 := arithEq_of_rows h (row := 1880) (i := 9) rfl (by decide)
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
  have e_1880_10 := arithEq_of_rows h (row := 1880) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_10
  simp only [← c114735, ← c114736, ← c114737] at e_1880_10
  have e_1880_11 := arithEq_of_rows h (row := 1880) (i := 11) rfl (by decide)
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
  have e_1880_12 := arithEq_of_rows h (row := 1880) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_12
  simp only [← c114741, ← c114742, ← c114743] at e_1880_12
  have e_1880_13 := arithEq_of_rows h (row := 1880) (i := 13) rfl (by decide)
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
  have e_1880_14 := arithEq_of_rows h (row := 1880) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_14
  simp only [← c114747, ← c114748, ← c114749] at e_1880_14
  have e_1880_15 := arithEq_of_rows h (row := 1880) (i := 15) rfl (by decide)
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
  have e_1880_16 := arithEq_of_rows h (row := 1880) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_16
  simp only [← c114753, ← c114754, ← c114755] at e_1880_16
  have e_1880_17 := arithEq_of_rows h (row := 1880) (i := 17) rfl (by decide)
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
  have e_1880_18 := arithEq_of_rows h (row := 1880) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_18
  simp only [← c114759, ← c114760, ← c114761] at e_1880_18
  have e_1880_19 := arithEq_of_rows h (row := 1880) (i := 19) rfl (by decide)
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
  have e_1881_0 := arithEq_of_rows h (row := 1881) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_0
  simp only [← c114765, ← c114766, ← c114767] at e_1881_0
  have e_1881_1 := arithEq_of_rows h (row := 1881) (i := 1) rfl (by decide)
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
  have e_1881_2 := arithEq_of_rows h (row := 1881) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_2
  simp only [← c114771, ← c114772, ← c114773] at e_1881_2
  have e_1881_3 := arithEq_of_rows h (row := 1881) (i := 3) rfl (by decide)
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
  have e_1881_4 := arithEq_of_rows h (row := 1881) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_4
  simp only [← c114777, ← c114778, ← c114779] at e_1881_4
  have e_1881_5 := arithEq_of_rows h (row := 1881) (i := 5) rfl (by decide)
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
  have e_1881_6 := arithEq_of_rows h (row := 1881) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_6
  simp only [← c114783, ← c114784, ← c114785] at e_1881_6
  have e_1881_7 := arithEq_of_rows h (row := 1881) (i := 7) rfl (by decide)
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
  have e_1881_8 := arithEq_of_rows h (row := 1881) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_8
  simp only [← c114789, ← c114790, ← c114791] at e_1881_8
  have e_1881_9 := arithEq_of_rows h (row := 1881) (i := 9) rfl (by decide)
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
  have e_1881_10 := arithEq_of_rows h (row := 1881) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_10
  simp only [← c114795, ← c114796, ← c114797] at e_1881_10
  have e_1881_11 := arithEq_of_rows h (row := 1881) (i := 11) rfl (by decide)
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
  have e_1881_12 := arithEq_of_rows h (row := 1881) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_12
  simp only [← c114801, ← c114802, ← c114803] at e_1881_12
  have e_1881_13 := arithEq_of_rows h (row := 1881) (i := 13) rfl (by decide)
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
  have e_1881_14 := arithEq_of_rows h (row := 1881) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_14
  simp only [← c114807, ← c114808, ← c114809] at e_1881_14
  have e_1881_15 := arithEq_of_rows h (row := 1881) (i := 15) rfl (by decide)
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
  have e_1881_16 := arithEq_of_rows h (row := 1881) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_16
  simp only [← c114813, ← c114814, ← c114815] at e_1881_16
  have e_1881_17 := arithEq_of_rows h (row := 1881) (i := 17) rfl (by decide)
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
  have e_1881_18 := arithEq_of_rows h (row := 1881) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_18
  simp only [← c114819, ← c114820, ← c114821] at e_1881_18
  have e_1881_19 := arithEq_of_rows h (row := 1881) (i := 19) rfl (by decide)
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
  have e_1882_0 := arithEq_of_rows h (row := 1882) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_0
  simp only [← c114825, ← c114826, ← c114827] at e_1882_0
  have e_1882_1 := arithEq_of_rows h (row := 1882) (i := 1) rfl (by decide)
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
  have e_1882_2 := arithEq_of_rows h (row := 1882) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_2
  simp only [← c114831, ← c114832, ← c114833] at e_1882_2
  have e_1882_3 := arithEq_of_rows h (row := 1882) (i := 3) rfl (by decide)
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
  have e_1882_4 := arithEq_of_rows h (row := 1882) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_4
  simp only [← c114837, ← c114838, ← c114839] at e_1882_4
  have e_1882_5 := arithEq_of_rows h (row := 1882) (i := 5) rfl (by decide)
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
  have e_1882_6 := arithEq_of_rows h (row := 1882) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_6
  simp only [← c114843, ← c114844, ← c114845] at e_1882_6
  have e_1882_7 := arithEq_of_rows h (row := 1882) (i := 7) rfl (by decide)
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
  have e_1882_8 := arithEq_of_rows h (row := 1882) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_8
  simp only [← c114849, ← c114850, ← c114851] at e_1882_8
  have e_1882_9 := arithEq_of_rows h (row := 1882) (i := 9) rfl (by decide)
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
  have e_1882_10 := arithEq_of_rows h (row := 1882) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_10
  simp only [← c114855, ← c114856, ← c114857] at e_1882_10
  have e_1882_11 := arithEq_of_rows h (row := 1882) (i := 11) rfl (by decide)
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
  have e_1882_12 := arithEq_of_rows h (row := 1882) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_12
  simp only [← c114861, ← c114862, ← c114863] at e_1882_12
  have e_1882_13 := arithEq_of_rows h (row := 1882) (i := 13) rfl (by decide)
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
  have e_1882_14 := arithEq_of_rows h (row := 1882) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_14
  simp only [← c114867, ← c114868, ← c114869] at e_1882_14
  have e_1882_15 := arithEq_of_rows h (row := 1882) (i := 15) rfl (by decide)
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
  have e_1882_16 := arithEq_of_rows h (row := 1882) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_16
  simp only [← c114873, ← c114874, ← c114875] at e_1882_16
  have e_1882_17 := arithEq_of_rows h (row := 1882) (i := 17) rfl (by decide)
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
  have e_1882_18 := arithEq_of_rows h (row := 1882) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_18
  simp only [← c114879, ← c114880, ← c114881] at e_1882_18
  have e_1882_19 := arithEq_of_rows h (row := 1882) (i := 19) rfl (by decide)
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
  have e_1883_0 := arithEq_of_rows h (row := 1883) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_0
  simp only [← c114885, ← c114886, ← c114887] at e_1883_0
  have e_1883_1 := arithEq_of_rows h (row := 1883) (i := 1) rfl (by decide)
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
  have e_1883_2 := arithEq_of_rows h (row := 1883) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_2
  simp only [← c114891, ← c114892, ← c114893] at e_1883_2
  have e_1883_3 := arithEq_of_rows h (row := 1883) (i := 3) rfl (by decide)
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
  have e_1883_4 := arithEq_of_rows h (row := 1883) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_4
  simp only [← c114897, ← c114898, ← c114899] at e_1883_4
  have e_1883_5 := arithEq_of_rows h (row := 1883) (i := 5) rfl (by decide)
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
  have e_1883_6 := arithEq_of_rows h (row := 1883) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_6
  simp only [← c114903, ← c114904, ← c114905] at e_1883_6
  have e_1883_7 := arithEq_of_rows h (row := 1883) (i := 7) rfl (by decide)
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
  have e_1883_8 := arithEq_of_rows h (row := 1883) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_8
  simp only [← c114909, ← c114910, ← c114911] at e_1883_8
  have e_1883_9 := arithEq_of_rows h (row := 1883) (i := 9) rfl (by decide)
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
  have e_1883_10 := arithEq_of_rows h (row := 1883) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_10
  simp only [← c114915, ← c114916, ← c114917] at e_1883_10
  have e_1883_11 := arithEq_of_rows h (row := 1883) (i := 11) rfl (by decide)
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
  have e_1883_12 := arithEq_of_rows h (row := 1883) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_12
  simp only [← c114921, ← c114922, ← c114923] at e_1883_12
  have e_1883_13 := arithEq_of_rows h (row := 1883) (i := 13) rfl (by decide)
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
  have e_1883_14 := arithEq_of_rows h (row := 1883) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_14
  simp only [← c114927, ← c114928, ← c114929] at e_1883_14
  have e_1883_15 := arithEq_of_rows h (row := 1883) (i := 15) rfl (by decide)
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
  have e_1883_16 := arithEq_of_rows h (row := 1883) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_16
  simp only [← c114933, ← c114934, ← c114935] at e_1883_16
  have e_1883_17 := arithEq_of_rows h (row := 1883) (i := 17) rfl (by decide)
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
  have e_1883_18 := arithEq_of_rows h (row := 1883) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_18
  simp only [← c114939, ← c114940, ← c114941] at e_1883_18
  have e_1883_19 := arithEq_of_rows h (row := 1883) (i := 19) rfl (by decide)
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
  have e_1884_0 := arithEq_of_rows h (row := 1884) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_0
  simp only [← c114945, ← c114946, ← c114947] at e_1884_0
  have e_1884_1 := arithEq_of_rows h (row := 1884) (i := 1) rfl (by decide)
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
  have e_1884_2 := arithEq_of_rows h (row := 1884) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_2
  simp only [← c114951, ← c114952, ← c114953] at e_1884_2
  have e_1884_3 := arithEq_of_rows h (row := 1884) (i := 3) rfl (by decide)
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
  have e_1884_4 := arithEq_of_rows h (row := 1884) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_4
  simp only [← c114957, ← c114958, ← c114959] at e_1884_4
  have e_1884_5 := arithEq_of_rows h (row := 1884) (i := 5) rfl (by decide)
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
  have e_1884_6 := arithEq_of_rows h (row := 1884) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_6
  simp only [← c114963, ← c114964, ← c114965] at e_1884_6
  have e_1884_7 := arithEq_of_rows h (row := 1884) (i := 7) rfl (by decide)
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
  have e_1884_8 := arithEq_of_rows h (row := 1884) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_8
  simp only [← c114969, ← c114970, ← c114971] at e_1884_8
  have e_1884_9 := arithEq_of_rows h (row := 1884) (i := 9) rfl (by decide)
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
  have e_1884_10 := arithEq_of_rows h (row := 1884) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_10
  simp only [← c114975, ← c114976, ← c114977] at e_1884_10
  have e_1884_11 := arithEq_of_rows h (row := 1884) (i := 11) rfl (by decide)
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
  have e_1884_12 := arithEq_of_rows h (row := 1884) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_12
  simp only [← c114981, ← c114982, ← c114983] at e_1884_12
  have e_1884_13 := arithEq_of_rows h (row := 1884) (i := 13) rfl (by decide)
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
  have e_1884_14 := arithEq_of_rows h (row := 1884) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_14
  simp only [← c114987, ← c114988, ← c114989] at e_1884_14
  have e_1884_15 := arithEq_of_rows h (row := 1884) (i := 15) rfl (by decide)
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
  have e_1884_16 := arithEq_of_rows h (row := 1884) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_16
  simp only [← c114993, ← c114994, ← c114995] at e_1884_16
  have e_1884_17 := arithEq_of_rows h (row := 1884) (i := 17) rfl (by decide)
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
  have e_1884_18 := arithEq_of_rows h (row := 1884) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_18
  simp only [← c114999, ← c115000, ← c115001] at e_1884_18
  have e_1884_19 := arithEq_of_rows h (row := 1884) (i := 19) rfl (by decide)
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
  have e_1885_0 := arithEq_of_rows h (row := 1885) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_0
  simp only [← c115005, ← c115006, ← c115007] at e_1885_0
  have e_1885_1 := arithEq_of_rows h (row := 1885) (i := 1) rfl (by decide)
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
  have e_1885_2 := arithEq_of_rows h (row := 1885) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_2
  simp only [← c115011, ← c115012, ← c115013] at e_1885_2
  have e_1885_3 := arithEq_of_rows h (row := 1885) (i := 3) rfl (by decide)
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
  have e_1885_4 := arithEq_of_rows h (row := 1885) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_4
  simp only [← c115017, ← c115018, ← c115019] at e_1885_4
  have e_1885_5 := arithEq_of_rows h (row := 1885) (i := 5) rfl (by decide)
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
  have e_1885_6 := arithEq_of_rows h (row := 1885) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_6
  simp only [← c115023, ← c115024, ← c115025] at e_1885_6
  have e_1885_7 := arithEq_of_rows h (row := 1885) (i := 7) rfl (by decide)
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
  have e_1885_8 := arithEq_of_rows h (row := 1885) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_8
  simp only [← c115029, ← c115030, ← c115031] at e_1885_8
  have e_1885_9 := arithEq_of_rows h (row := 1885) (i := 9) rfl (by decide)
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
  have e_1885_10 := arithEq_of_rows h (row := 1885) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_10
  simp only [← c115035, ← c115036, ← c115037] at e_1885_10
  have e_1885_11 := arithEq_of_rows h (row := 1885) (i := 11) rfl (by decide)
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
  have e_1885_12 := arithEq_of_rows h (row := 1885) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_12
  simp only [← c115041, ← c115042, ← c115043] at e_1885_12
  have e_1885_13 := arithEq_of_rows h (row := 1885) (i := 13) rfl (by decide)
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
  have e_1885_14 := arithEq_of_rows h (row := 1885) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_14
  simp only [← c115047, ← c115048, ← c115049] at e_1885_14
  have e_1885_15 := arithEq_of_rows h (row := 1885) (i := 15) rfl (by decide)
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
  have e_1885_16 := arithEq_of_rows h (row := 1885) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_16
  simp only [← c115053, ← c115054, ← c115055] at e_1885_16
  have e_1885_17 := arithEq_of_rows h (row := 1885) (i := 17) rfl (by decide)
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
  have e_1885_18 := arithEq_of_rows h (row := 1885) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_18
  simp only [← c115059, ← c115060, ← c115061] at e_1885_18
  have e_1885_19 := arithEq_of_rows h (row := 1885) (i := 19) rfl (by decide)
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
  have e_1886_0 := arithEq_of_rows h (row := 1886) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_0
  simp only [← c115065, ← c115066, ← c115067] at e_1886_0
  have e_1886_1 := arithEq_of_rows h (row := 1886) (i := 1) rfl (by decide)
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
  have e_1886_2 := arithEq_of_rows h (row := 1886) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_2
  simp only [← c115071, ← c115072, ← c115073] at e_1886_2
  have e_1886_3 := arithEq_of_rows h (row := 1886) (i := 3) rfl (by decide)
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
  have e_1886_4 := arithEq_of_rows h (row := 1886) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_4
  simp only [← c115077, ← c115078, ← c115079] at e_1886_4
  have e_1886_5 := arithEq_of_rows h (row := 1886) (i := 5) rfl (by decide)
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
  have e_1886_6 := arithEq_of_rows h (row := 1886) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_6
  simp only [← c115083, ← c115084, ← c115085] at e_1886_6
  have e_1886_7 := arithEq_of_rows h (row := 1886) (i := 7) rfl (by decide)
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
  have e_1886_8 := arithEq_of_rows h (row := 1886) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_8
  simp only [← c115089, ← c115090, ← c115091] at e_1886_8
  have e_1886_9 := arithEq_of_rows h (row := 1886) (i := 9) rfl (by decide)
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
  have e_1886_10 := arithEq_of_rows h (row := 1886) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_10
  simp only [← c115095, ← c115096, ← c115097] at e_1886_10
  have e_1886_11 := arithEq_of_rows h (row := 1886) (i := 11) rfl (by decide)
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
  have e_1886_12 := arithEq_of_rows h (row := 1886) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_12
  simp only [← c115101, ← c115102, ← c115103] at e_1886_12
  have e_1886_13 := arithEq_of_rows h (row := 1886) (i := 13) rfl (by decide)
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
  have e_1886_14 := arithEq_of_rows h (row := 1886) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_14
  simp only [← c115107, ← c115108, ← c115109] at e_1886_14
  have e_1886_15 := arithEq_of_rows h (row := 1886) (i := 15) rfl (by decide)
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
  have e_1886_16 := arithEq_of_rows h (row := 1886) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_16
  simp only [← c115113, ← c115114, ← c115115] at e_1886_16
  have e_1886_17 := arithEq_of_rows h (row := 1886) (i := 17) rfl (by decide)
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
  have e_1886_18 := arithEq_of_rows h (row := 1886) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_18
  simp only [← c115119, ← c115120, ← c115121] at e_1886_18
  have e_1886_19 := arithEq_of_rows h (row := 1886) (i := 19) rfl (by decide)
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
  have e_1887_0 := arithEq_of_rows h (row := 1887) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_0
  simp only [← c115125, ← c115126, ← c115127] at e_1887_0
  have e_1887_1 := arithEq_of_rows h (row := 1887) (i := 1) rfl (by decide)
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
  have e_1887_2 := arithEq_of_rows h (row := 1887) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_2
  simp only [← c115131, ← c115132, ← c115133] at e_1887_2
  have e_1887_3 := arithEq_of_rows h (row := 1887) (i := 3) rfl (by decide)
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
  have e_1887_4 := arithEq_of_rows h (row := 1887) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_4
  simp only [← c115137, ← c115138, ← c115139] at e_1887_4
  have e_1887_5 := arithEq_of_rows h (row := 1887) (i := 5) rfl (by decide)
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
  have e_1887_6 := arithEq_of_rows h (row := 1887) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_6
  simp only [← c115143, ← c115144, ← c115145] at e_1887_6
  have e_1887_7 := arithEq_of_rows h (row := 1887) (i := 7) rfl (by decide)
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
  have e_1887_8 := arithEq_of_rows h (row := 1887) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_8
  simp only [← c115149, ← c115150, ← c115151] at e_1887_8
  have e_1887_9 := arithEq_of_rows h (row := 1887) (i := 9) rfl (by decide)
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
  have e_1887_10 := arithEq_of_rows h (row := 1887) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_10
  simp only [← c115155, ← c115156, ← c115157] at e_1887_10
  have e_1887_11 := arithEq_of_rows h (row := 1887) (i := 11) rfl (by decide)
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
  have e_1887_12 := arithEq_of_rows h (row := 1887) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_12
  simp only [← c115161, ← c115162, ← c115163] at e_1887_12
  have e_1887_13 := arithEq_of_rows h (row := 1887) (i := 13) rfl (by decide)
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
  have e_1887_14 := arithEq_of_rows h (row := 1887) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_14
  simp only [← c115167, ← c115168, ← c115169] at e_1887_14
  have e_1887_15 := arithEq_of_rows h (row := 1887) (i := 15) rfl (by decide)
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
  have e_1887_16 := arithEq_of_rows h (row := 1887) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_16
  simp only [← c115173, ← c115174, ← c115175] at e_1887_16
  have e_1887_17 := arithEq_of_rows h (row := 1887) (i := 17) rfl (by decide)
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
  have e_1887_18 := arithEq_of_rows h (row := 1887) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_18
  simp only [← c115179, ← c115180, ← c115181] at e_1887_18
  have e_1887_19 := arithEq_of_rows h (row := 1887) (i := 19) rfl (by decide)
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
  have e_1888_0 := arithEq_of_rows h (row := 1888) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_0
  simp only [← c115185, ← c115186, ← c115187] at e_1888_0
  have e_1888_1 := arithEq_of_rows h (row := 1888) (i := 1) rfl (by decide)
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
  have e_1888_2 := arithEq_of_rows h (row := 1888) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_2
  simp only [← c115191, ← c115192, ← c115193] at e_1888_2
  have e_1888_3 := arithEq_of_rows h (row := 1888) (i := 3) rfl (by decide)
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
  have e_1888_4 := arithEq_of_rows h (row := 1888) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_4
  simp only [← c115197, ← c115198, ← c115199] at e_1888_4
  have e_1888_5 := arithEq_of_rows h (row := 1888) (i := 5) rfl (by decide)
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
  have e_1888_6 := arithEq_of_rows h (row := 1888) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_6
  simp only [← c115203, ← c115204, ← c115205] at e_1888_6
  have e_1888_7 := arithEq_of_rows h (row := 1888) (i := 7) rfl (by decide)
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
  have e_1888_8 := arithEq_of_rows h (row := 1888) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_8
  simp only [← c115209, ← c115210, ← c115211] at e_1888_8
  have e_1888_9 := arithEq_of_rows h (row := 1888) (i := 9) rfl (by decide)
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
  have e_1888_10 := arithEq_of_rows h (row := 1888) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_10
  simp only [← c115215, ← c115216, ← c115217] at e_1888_10
  have e_1888_11 := arithEq_of_rows h (row := 1888) (i := 11) rfl (by decide)
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
  have e_1888_12 := arithEq_of_rows h (row := 1888) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_12
  simp only [← c115221, ← c115222, ← c115223] at e_1888_12
  have e_1888_13 := arithEq_of_rows h (row := 1888) (i := 13) rfl (by decide)
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
  have e_1888_14 := arithEq_of_rows h (row := 1888) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_14
  simp only [← c115227, ← c115228, ← c115229] at e_1888_14
  have e_1888_15 := arithEq_of_rows h (row := 1888) (i := 15) rfl (by decide)
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
  have e_1888_16 := arithEq_of_rows h (row := 1888) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_16
  simp only [← c115233, ← c115234, ← c115235] at e_1888_16
  have e_1888_17 := arithEq_of_rows h (row := 1888) (i := 17) rfl (by decide)
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
  have e_1888_18 := arithEq_of_rows h (row := 1888) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_18
  simp only [← c115239, ← c115240, ← c115241] at e_1888_18
  have e_1888_19 := arithEq_of_rows h (row := 1888) (i := 19) rfl (by decide)
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
  have e_1889_0 := arithEq_of_rows h (row := 1889) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_0
  simp only [← c115245, ← c115246, ← c115247] at e_1889_0
  have e_1889_1 := arithEq_of_rows h (row := 1889) (i := 1) rfl (by decide)
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
  have e_1889_2 := arithEq_of_rows h (row := 1889) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_2
  simp only [← c115251, ← c115252, ← c115253] at e_1889_2
  have e_1889_3 := arithEq_of_rows h (row := 1889) (i := 3) rfl (by decide)
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
  have e_1889_4 := arithEq_of_rows h (row := 1889) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_4
  simp only [← c115257, ← c115258, ← c115259] at e_1889_4
  have e_1889_5 := arithEq_of_rows h (row := 1889) (i := 5) rfl (by decide)
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
  have e_1889_6 := arithEq_of_rows h (row := 1889) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_6
  simp only [← c115263, ← c115264, ← c115265] at e_1889_6
  have e_1889_7 := arithEq_of_rows h (row := 1889) (i := 7) rfl (by decide)
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
  have e_1889_8 := arithEq_of_rows h (row := 1889) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_8
  simp only [← c115269, ← c115270, ← c115271] at e_1889_8
  have e_1889_9 := arithEq_of_rows h (row := 1889) (i := 9) rfl (by decide)
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
  have e_1889_10 := arithEq_of_rows h (row := 1889) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_10
  simp only [← c115275, ← c115276, ← c115277] at e_1889_10
  have e_1889_11 := arithEq_of_rows h (row := 1889) (i := 11) rfl (by decide)
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
  have e_1889_12 := arithEq_of_rows h (row := 1889) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_12
  simp only [← c115281, ← c115282, ← c115283] at e_1889_12
  have e_1889_13 := arithEq_of_rows h (row := 1889) (i := 13) rfl (by decide)
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
  have e_1889_14 := arithEq_of_rows h (row := 1889) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_14
  simp only [← c115287, ← c115288, ← c115289] at e_1889_14
  have e_1889_15 := arithEq_of_rows h (row := 1889) (i := 15) rfl (by decide)
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
  have e_1889_16 := arithEq_of_rows h (row := 1889) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_16
  simp only [← c115293, ← c115294, ← c115295] at e_1889_16
  have e_1889_17 := arithEq_of_rows h (row := 1889) (i := 17) rfl (by decide)
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
  have e_1889_18 := arithEq_of_rows h (row := 1889) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_18
  simp only [← c115299, ← c115300, ← c115301] at e_1889_18
  have e_1889_19 := arithEq_of_rows h (row := 1889) (i := 19) rfl (by decide)
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
  have e_1890_0 := arithEq_of_rows h (row := 1890) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_0
  simp only [← c115305, ← c115306, ← c115307] at e_1890_0
  have e_1890_1 := arithEq_of_rows h (row := 1890) (i := 1) rfl (by decide)
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
  have e_1890_2 := arithEq_of_rows h (row := 1890) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_2
  simp only [← c115311, ← c115312, ← c115313] at e_1890_2
  have e_1890_3 := arithEq_of_rows h (row := 1890) (i := 3) rfl (by decide)
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
  have e_1890_4 := arithEq_of_rows h (row := 1890) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_4
  simp only [← c115317, ← c115318, ← c115319] at e_1890_4
  have e_1890_5 := arithEq_of_rows h (row := 1890) (i := 5) rfl (by decide)
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
  have e_1890_6 := arithEq_of_rows h (row := 1890) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_6
  simp only [← c115323, ← c115324, ← c115325] at e_1890_6
  have e_1890_7 := arithEq_of_rows h (row := 1890) (i := 7) rfl (by decide)
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
  have e_1890_8 := arithEq_of_rows h (row := 1890) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_8
  simp only [← c115329, ← c115330, ← c115331] at e_1890_8
  have e_1890_9 := arithEq_of_rows h (row := 1890) (i := 9) rfl (by decide)
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
  have e_1890_10 := arithEq_of_rows h (row := 1890) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_10
  simp only [← c115335, ← c115336, ← c115337] at e_1890_10
  have e_1890_11 := arithEq_of_rows h (row := 1890) (i := 11) rfl (by decide)
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
  have e_1890_12 := arithEq_of_rows h (row := 1890) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_12
  simp only [← c115341, ← c115342, ← c115343] at e_1890_12
  have e_1890_13 := arithEq_of_rows h (row := 1890) (i := 13) rfl (by decide)
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
  have e_1890_14 := arithEq_of_rows h (row := 1890) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_14
  simp only [← c115347, ← c115348, ← c115349] at e_1890_14
  have e_1890_15 := arithEq_of_rows h (row := 1890) (i := 15) rfl (by decide)
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
  have e_1890_16 := arithEq_of_rows h (row := 1890) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_16
  simp only [← c115353, ← c115354, ← c115355] at e_1890_16
  have e_1890_17 := arithEq_of_rows h (row := 1890) (i := 17) rfl (by decide)
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
  have e_1890_18 := arithEq_of_rows h (row := 1890) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_18
  simp only [← c115359, ← c115360, ← c115361] at e_1890_18
  have e_1890_19 := arithEq_of_rows h (row := 1890) (i := 19) rfl (by decide)
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
  have e_1891_0 := arithEq_of_rows h (row := 1891) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_0
  simp only [← c115365, ← c115366, ← c115367] at e_1891_0
  have e_1891_1 := arithEq_of_rows h (row := 1891) (i := 1) rfl (by decide)
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
  have e_1891_2 := arithEq_of_rows h (row := 1891) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_2
  simp only [← c115371, ← c115372, ← c115373] at e_1891_2
  have e_1891_3 := arithEq_of_rows h (row := 1891) (i := 3) rfl (by decide)
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
  have e_1891_4 := arithEq_of_rows h (row := 1891) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_4
  simp only [← c115377, ← c115378, ← c115379] at e_1891_4
  have e_1891_5 := arithEq_of_rows h (row := 1891) (i := 5) rfl (by decide)
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
  have e_1891_6 := arithEq_of_rows h (row := 1891) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_6
  simp only [← c115383, ← c115384, ← c115385] at e_1891_6
  have e_1891_7 := arithEq_of_rows h (row := 1891) (i := 7) rfl (by decide)
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
  have e_1891_8 := arithEq_of_rows h (row := 1891) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_8
  simp only [← c115389, ← c115390, ← c115391] at e_1891_8
  have e_1891_9 := arithEq_of_rows h (row := 1891) (i := 9) rfl (by decide)
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
  have e_1891_10 := arithEq_of_rows h (row := 1891) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_10
  simp only [← c115395, ← c115396, ← c115397] at e_1891_10
  have e_1891_11 := arithEq_of_rows h (row := 1891) (i := 11) rfl (by decide)
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
  have e_1891_12 := arithEq_of_rows h (row := 1891) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_12
  simp only [← c115401, ← c115402, ← c115403] at e_1891_12
  have e_1891_13 := arithEq_of_rows h (row := 1891) (i := 13) rfl (by decide)
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
  have e_1891_14 := arithEq_of_rows h (row := 1891) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_14
  simp only [← c115407, ← c115408, ← c115409] at e_1891_14
  have e_1891_15 := arithEq_of_rows h (row := 1891) (i := 15) rfl (by decide)
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
  have e_1891_16 := arithEq_of_rows h (row := 1891) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_16
  simp only [← c115413, ← c115414, ← c115415] at e_1891_16
  have e_1891_17 := arithEq_of_rows h (row := 1891) (i := 17) rfl (by decide)
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
  have e_1891_18 := arithEq_of_rows h (row := 1891) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_18
  simp only [← c115419, ← c115420, ← c115421] at e_1891_18
  have e_1891_19 := arithEq_of_rows h (row := 1891) (i := 19) rfl (by decide)
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
  have e_1892_0 := arithEq_of_rows h (row := 1892) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_0
  simp only [← c115425, ← c115426, ← c115427] at e_1892_0
  have e_1892_1 := arithEq_of_rows h (row := 1892) (i := 1) rfl (by decide)
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
  have e_1892_2 := arithEq_of_rows h (row := 1892) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_2
  simp only [← c115431, ← c115432, ← c115433] at e_1892_2
  have e_1892_3 := arithEq_of_rows h (row := 1892) (i := 3) rfl (by decide)
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
  have e_1892_4 := arithEq_of_rows h (row := 1892) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_4
  simp only [← c115437, ← c115438, ← c115439] at e_1892_4
  have e_1892_5 := arithEq_of_rows h (row := 1892) (i := 5) rfl (by decide)
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
  have e_1892_6 := arithEq_of_rows h (row := 1892) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_6
  simp only [← c115443, ← c115444, ← c115445] at e_1892_6
  have e_1892_7 := arithEq_of_rows h (row := 1892) (i := 7) rfl (by decide)
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
  have e_1892_8 := arithEq_of_rows h (row := 1892) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_8
  simp only [← c115449, ← c115450, ← c115451] at e_1892_8
  have e_1892_9 := arithEq_of_rows h (row := 1892) (i := 9) rfl (by decide)
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
  have e_1892_10 := arithEq_of_rows h (row := 1892) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_10
  simp only [← c115455, ← c115456, ← c115457] at e_1892_10
  have e_1892_11 := arithEq_of_rows h (row := 1892) (i := 11) rfl (by decide)
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
  have e_1892_12 := arithEq_of_rows h (row := 1892) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_12
  simp only [← c115461, ← c115462, ← c115463] at e_1892_12
  have e_1892_13 := arithEq_of_rows h (row := 1892) (i := 13) rfl (by decide)
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
  have e_1892_14 := arithEq_of_rows h (row := 1892) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_14
  simp only [← c115467, ← c115468, ← c115469] at e_1892_14
  have e_1892_15 := arithEq_of_rows h (row := 1892) (i := 15) rfl (by decide)
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
  have e_1892_16 := arithEq_of_rows h (row := 1892) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_16
  simp only [← c115473, ← c115474, ← c115475] at e_1892_16
  have e_1892_17 := arithEq_of_rows h (row := 1892) (i := 17) rfl (by decide)
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
  have e_1892_18 := arithEq_of_rows h (row := 1892) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_18
  simp only [← c115479, ← c115480, ← c115481] at e_1892_18
  have e_1892_19 := arithEq_of_rows h (row := 1892) (i := 19) rfl (by decide)
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
  have e_1893_0 := arithEq_of_rows h (row := 1893) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_0
  simp only [← c115485, ← c115486, ← c115487] at e_1893_0
  have e_1893_1 := arithEq_of_rows h (row := 1893) (i := 1) rfl (by decide)
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
  have e_1893_2 := arithEq_of_rows h (row := 1893) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_2
  simp only [← c115491, ← c115492, ← c115493] at e_1893_2
  have e_1893_3 := arithEq_of_rows h (row := 1893) (i := 3) rfl (by decide)
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
  have e_1893_4 := arithEq_of_rows h (row := 1893) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_4
  simp only [← c115497, ← c115498, ← c115499] at e_1893_4
  have e_1893_5 := arithEq_of_rows h (row := 1893) (i := 5) rfl (by decide)
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
  have e_1893_6 := arithEq_of_rows h (row := 1893) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_6
  simp only [← c115503, ← c115504, ← c115505] at e_1893_6
  have e_1893_7 := arithEq_of_rows h (row := 1893) (i := 7) rfl (by decide)
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
  have e_1893_8 := arithEq_of_rows h (row := 1893) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_8
  simp only [← c115509, ← c115510, ← c115511] at e_1893_8
  have e_1893_9 := arithEq_of_rows h (row := 1893) (i := 9) rfl (by decide)
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
  have e_1893_10 := arithEq_of_rows h (row := 1893) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_10
  simp only [← c115515, ← c115516, ← c115517] at e_1893_10
  have e_1893_11 := arithEq_of_rows h (row := 1893) (i := 11) rfl (by decide)
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
  have e_1893_12 := arithEq_of_rows h (row := 1893) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_12
  simp only [← c115521, ← c115522, ← c115523] at e_1893_12
  have e_1893_13 := arithEq_of_rows h (row := 1893) (i := 13) rfl (by decide)
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
  have e_1893_14 := arithEq_of_rows h (row := 1893) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_14
  simp only [← c115527, ← c115528, ← c115529] at e_1893_14
  have e_1893_15 := arithEq_of_rows h (row := 1893) (i := 15) rfl (by decide)
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
  have e_1893_16 := arithEq_of_rows h (row := 1893) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_16
  simp only [← c115533, ← c115534, ← c115535] at e_1893_16
  have e_1893_17 := arithEq_of_rows h (row := 1893) (i := 17) rfl (by decide)
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
  have e_1893_18 := arithEq_of_rows h (row := 1893) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_18
  simp only [← c115539, ← c115540, ← c115541] at e_1893_18
  have e_1893_19 := arithEq_of_rows h (row := 1893) (i := 19) rfl (by decide)
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
  have e_1894_0 := arithEq_of_rows h (row := 1894) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_0
  simp only [← c115545, ← c115546, ← c115547] at e_1894_0
  have e_1894_1 := arithEq_of_rows h (row := 1894) (i := 1) rfl (by decide)
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
  have e_1894_2 := arithEq_of_rows h (row := 1894) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_2
  simp only [← c115551, ← c115552, ← c115553] at e_1894_2
  have e_1894_3 := arithEq_of_rows h (row := 1894) (i := 3) rfl (by decide)
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
  have e_1894_4 := arithEq_of_rows h (row := 1894) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_4
  simp only [← c115557, ← c115558, ← c115559] at e_1894_4
  have e_1894_5 := arithEq_of_rows h (row := 1894) (i := 5) rfl (by decide)
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
  have e_1894_6 := arithEq_of_rows h (row := 1894) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_6
  simp only [← c115563, ← c115564, ← c115565] at e_1894_6
  have e_1894_7 := arithEq_of_rows h (row := 1894) (i := 7) rfl (by decide)
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
  have e_1894_8 := arithEq_of_rows h (row := 1894) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_8
  simp only [← c115569, ← c115570, ← c115571] at e_1894_8
  have e_1894_9 := arithEq_of_rows h (row := 1894) (i := 9) rfl (by decide)
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
  have e_1894_10 := arithEq_of_rows h (row := 1894) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_10
  simp only [← c115575, ← c115576, ← c115577] at e_1894_10
  have e_1894_11 := arithEq_of_rows h (row := 1894) (i := 11) rfl (by decide)
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
  have e_1894_12 := arithEq_of_rows h (row := 1894) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_12
  simp only [← c115581, ← c115582, ← c115583] at e_1894_12
  have e_1894_13 := arithEq_of_rows h (row := 1894) (i := 13) rfl (by decide)
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
  have e_1894_14 := arithEq_of_rows h (row := 1894) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_14
  simp only [← c115587, ← c115588, ← c115589] at e_1894_14
  have e_1894_15 := arithEq_of_rows h (row := 1894) (i := 15) rfl (by decide)
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
  have e_1894_16 := arithEq_of_rows h (row := 1894) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_16
  simp only [← c115593, ← c115594, ← c115595] at e_1894_16
  have e_1894_17 := arithEq_of_rows h (row := 1894) (i := 17) rfl (by decide)
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
  have e_1894_18 := arithEq_of_rows h (row := 1894) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_18
  simp only [← c115599, ← c115600, ← c115601] at e_1894_18
  have e_1894_19 := arithEq_of_rows h (row := 1894) (i := 19) rfl (by decide)
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
  have e_1895_0 := arithEq_of_rows h (row := 1895) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_0
  simp only [← c115605, ← c115606, ← c115607] at e_1895_0
  have e_1895_1 := arithEq_of_rows h (row := 1895) (i := 1) rfl (by decide)
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
  have e_1895_2 := arithEq_of_rows h (row := 1895) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_2
  simp only [← c115611, ← c115612, ← c115613] at e_1895_2
  have e_1895_3 := arithEq_of_rows h (row := 1895) (i := 3) rfl (by decide)
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
  have e_1895_4 := arithEq_of_rows h (row := 1895) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_4
  simp only [← c115617, ← c115618, ← c115619] at e_1895_4
  have e_1895_5 := arithEq_of_rows h (row := 1895) (i := 5) rfl (by decide)
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
  have e_1895_6 := arithEq_of_rows h (row := 1895) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_6
  simp only [← c115623, ← c115624, ← c115625] at e_1895_6
  have e_1895_7 := arithEq_of_rows h (row := 1895) (i := 7) rfl (by decide)
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
  have e_1895_8 := arithEq_of_rows h (row := 1895) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_8
  simp only [← c115629, ← c115630, ← c115631] at e_1895_8
  have e_1895_9 := arithEq_of_rows h (row := 1895) (i := 9) rfl (by decide)
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
  have e_1895_10 := arithEq_of_rows h (row := 1895) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_10
  simp only [← c115635, ← c115636, ← c115637] at e_1895_10
  have e_1895_11 := arithEq_of_rows h (row := 1895) (i := 11) rfl (by decide)
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
  have e_1895_12 := arithEq_of_rows h (row := 1895) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_12
  simp only [← c115641, ← c115642, ← c115643] at e_1895_12
  have e_1895_13 := arithEq_of_rows h (row := 1895) (i := 13) rfl (by decide)
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
  have e_1895_14 := arithEq_of_rows h (row := 1895) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_14
  simp only [← c115647, ← c115648, ← c115649] at e_1895_14
  have e_1895_15 := arithEq_of_rows h (row := 1895) (i := 15) rfl (by decide)
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
  have e_1895_16 := arithEq_of_rows h (row := 1895) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_16
  simp only [← c115653, ← c115654, ← c115655] at e_1895_16
  have e_1895_17 := arithEq_of_rows h (row := 1895) (i := 17) rfl (by decide)
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
  have e_1895_18 := arithEq_of_rows h (row := 1895) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_18
  simp only [← c115659, ← c115660, ← c115661] at e_1895_18
  have e_1895_19 := arithEq_of_rows h (row := 1895) (i := 19) rfl (by decide)
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
  have e_1896_0 := arithEq_of_rows h (row := 1896) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_0
  simp only [← c115665, ← c115666, ← c115667] at e_1896_0
  have e_1896_1 := arithEq_of_rows h (row := 1896) (i := 1) rfl (by decide)
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
  have e_1896_2 := arithEq_of_rows h (row := 1896) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_2
  simp only [← c115671, ← c115672, ← c115673] at e_1896_2
  have e_1896_3 := arithEq_of_rows h (row := 1896) (i := 3) rfl (by decide)
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
  have e_1896_4 := arithEq_of_rows h (row := 1896) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_4
  simp only [← c115677, ← c115678, ← c115679] at e_1896_4
  have e_1896_5 := arithEq_of_rows h (row := 1896) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_5
  simp only [← c115680, ← c115681, k2, ← c115682] at e_1896_5
  have hr := e_1896_5
  simp only [e_1896_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
