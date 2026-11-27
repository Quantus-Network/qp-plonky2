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

theorem publicBatchWrapper89_f19753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15593)) := by
  have c123024 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123025 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123026 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123027 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123028 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123029 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_4
  simp only [← c123024, ← c123025, ← c123026] at e_2017_4
  have e_2017_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_5
  simp only [← c123027, ← c123028, k2, ← c123029] at e_2017_5
  have hr := e_2017_5
  simp only [e_2017_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15594)) := by
  have c123030 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123031 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123032 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123033 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123034 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123035 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_6
  simp only [← c123030, ← c123031, ← c123032] at e_2017_6
  have e_2017_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_7
  simp only [← c123033, ← c123034, k2, ← c123035] at e_2017_7
  have hr := e_2017_7
  simp only [e_2017_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15595)) := by
  have c123036 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123037 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123038 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123039 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123040 := (publicBatchWrapper89_copies3845 a h).1
  have c123041 := (publicBatchWrapper89_copies3845 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_8
  simp only [← c123036, ← c123037, ← c123038] at e_2017_8
  have e_2017_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_9
  simp only [← c123039, ← c123040, k2, ← c123041] at e_2017_9
  have hr := e_2017_9
  simp only [e_2017_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15596)) := by
  have c123042 := (publicBatchWrapper89_copies3845 a h).2.2.1
  have c123043 := (publicBatchWrapper89_copies3845 a h).2.2.2.1
  have c123044 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.1
  have c123045 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.1
  have c123046 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.1
  have c123047 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_10
  simp only [← c123042, ← c123043, ← c123044] at e_2017_10
  have e_2017_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_11
  simp only [← c123045, ← c123046, k2, ← c123047] at e_2017_11
  have hr := e_2017_11
  simp only [e_2017_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15597)) := by
  have c123048 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.1
  have c123049 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.1
  have c123050 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123051 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123052 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123053 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_12
  simp only [← c123048, ← c123049, ← c123050] at e_2017_12
  have e_2017_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_13
  simp only [← c123051, ← c123052, k2, ← c123053] at e_2017_13
  have hr := e_2017_13
  simp only [e_2017_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15598)) := by
  have c123054 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123055 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123056 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123057 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123058 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123059 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_14
  simp only [← c123054, ← c123055, ← c123056] at e_2017_14
  have e_2017_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_15
  simp only [← c123057, ← c123058, k2, ← c123059] at e_2017_15
  have hr := e_2017_15
  simp only [e_2017_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15599)) := by
  have c123060 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123061 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123062 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123063 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123064 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123065 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_16
  simp only [← c123060, ← c123061, ← c123062] at e_2017_16
  have e_2017_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_17
  simp only [← c123063, ← c123064, k2, ← c123065] at e_2017_17
  have hr := e_2017_17
  simp only [e_2017_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15600)) := by
  have c123066 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123067 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123068 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123069 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123070 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123071 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_18
  simp only [← c123066, ← c123067, ← c123068] at e_2017_18
  have e_2017_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_19
  simp only [← c123069, ← c123070, k2, ← c123071] at e_2017_19
  have hr := e_2017_19
  simp only [e_2017_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15601)) := by
  have c123072 := (publicBatchWrapper89_copies3846 a h).1
  have c123073 := (publicBatchWrapper89_copies3846 a h).2.1
  have c123074 := (publicBatchWrapper89_copies3846 a h).2.2.1
  have c123075 := (publicBatchWrapper89_copies3846 a h).2.2.2.1
  have c123076 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.1
  have c123077 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_0
  simp only [← c123072, ← c123073, ← c123074] at e_2018_0
  have e_2018_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_1
  simp only [← c123075, ← c123076, k2, ← c123077] at e_2018_1
  have hr := e_2018_1
  simp only [e_2018_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15770)) := by
  have c123078 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.1
  have c123079 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.1
  have c123080 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.1
  have c123081 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.1
  have c123082 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123083 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_2
  simp only [← c123078, ← c123079, ← c123080] at e_2018_2
  have e_2018_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_3
  simp only [← c123081, ← c123082, k2, ← c123083] at e_2018_3
  have hr := e_2018_3
  simp only [e_2018_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15771)) := by
  have c123084 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123085 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123086 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123087 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123088 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123089 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_4
  simp only [← c123084, ← c123085, ← c123086] at e_2018_4
  have e_2018_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_5
  simp only [← c123087, ← c123088, k2, ← c123089] at e_2018_5
  have hr := e_2018_5
  simp only [e_2018_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15772)) := by
  have c123090 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123091 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123092 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123093 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123094 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123095 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_6
  simp only [← c123090, ← c123091, ← c123092] at e_2018_6
  have e_2018_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_7
  simp only [← c123093, ← c123094, k2, ← c123095] at e_2018_7
  have hr := e_2018_7
  simp only [e_2018_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15773)) := by
  have c123096 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123097 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123098 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123099 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123100 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123101 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_8
  simp only [← c123096, ← c123097, ← c123098] at e_2018_8
  have e_2018_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_9
  simp only [← c123099, ← c123100, k2, ← c123101] at e_2018_9
  have hr := e_2018_9
  simp only [e_2018_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15774)) := by
  have c123102 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123103 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123104 := (publicBatchWrapper89_copies3847 a h).1
  have c123105 := (publicBatchWrapper89_copies3847 a h).2.1
  have c123106 := (publicBatchWrapper89_copies3847 a h).2.2.1
  have c123107 := (publicBatchWrapper89_copies3847 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_10
  simp only [← c123102, ← c123103, ← c123104] at e_2018_10
  have e_2018_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_11
  simp only [← c123105, ← c123106, k2, ← c123107] at e_2018_11
  have hr := e_2018_11
  simp only [e_2018_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15775)) := by
  have c123108 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.1
  have c123109 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.1
  have c123110 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.1
  have c123111 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.1
  have c123112 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.1
  have c123113 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_12
  simp only [← c123108, ← c123109, ← c123110] at e_2018_12
  have e_2018_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_13
  simp only [← c123111, ← c123112, k2, ← c123113] at e_2018_13
  have hr := e_2018_13
  simp only [e_2018_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15776)) := by
  have c123114 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123115 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123116 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123117 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123118 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123119 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_14
  simp only [← c123114, ← c123115, ← c123116] at e_2018_14
  have e_2018_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_15
  simp only [← c123117, ← c123118, k2, ← c123119] at e_2018_15
  have hr := e_2018_15
  simp only [e_2018_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15777)) := by
  have c123120 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123121 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123122 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123123 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123124 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123125 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_16
  simp only [← c123120, ← c123121, ← c123122] at e_2018_16
  have e_2018_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_17
  simp only [← c123123, ← c123124, k2, ← c123125] at e_2018_17
  have hr := e_2018_17
  simp only [e_2018_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15778)) := by
  have c123126 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123127 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123128 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123129 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123130 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123131 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_18
  simp only [← c123126, ← c123127, ← c123128] at e_2018_18
  have e_2018_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2018 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_19
  simp only [← c123129, ← c123130, k2, ← c123131] at e_2018_19
  have hr := e_2018_19
  simp only [e_2018_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15779)) := by
  have c123132 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123133 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123134 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123135 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123136 := (publicBatchWrapper89_copies3848 a h).1
  have c123137 := (publicBatchWrapper89_copies3848 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_0
  simp only [← c123132, ← c123133, ← c123134] at e_2019_0
  have e_2019_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_1
  simp only [← c123135, ← c123136, k2, ← c123137] at e_2019_1
  have hr := e_2019_1
  simp only [e_2019_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15780)) := by
  have c123138 := (publicBatchWrapper89_copies3848 a h).2.2.1
  have c123139 := (publicBatchWrapper89_copies3848 a h).2.2.2.1
  have c123140 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.1
  have c123141 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.1
  have c123142 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.1
  have c123143 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_2
  simp only [← c123138, ← c123139, ← c123140] at e_2019_2
  have e_2019_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_3
  simp only [← c123141, ← c123142, k2, ← c123143] at e_2019_3
  have hr := e_2019_3
  simp only [e_2019_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15781)) := by
  have c123144 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.1
  have c123145 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.1
  have c123146 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123147 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123148 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123149 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_4
  simp only [← c123144, ← c123145, ← c123146] at e_2019_4
  have e_2019_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_5
  simp only [← c123147, ← c123148, k2, ← c123149] at e_2019_5
  have hr := e_2019_5
  simp only [e_2019_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15782)) := by
  have c123150 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123151 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123152 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123153 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123154 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123155 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_6
  simp only [← c123150, ← c123151, ← c123152] at e_2019_6
  have e_2019_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_7
  simp only [← c123153, ← c123154, k2, ← c123155] at e_2019_7
  have hr := e_2019_7
  simp only [e_2019_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15783)) := by
  have c123156 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123157 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123158 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123159 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123160 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123161 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_8
  simp only [← c123156, ← c123157, ← c123158] at e_2019_8
  have e_2019_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_9
  simp only [← c123159, ← c123160, k2, ← c123161] at e_2019_9
  have hr := e_2019_9
  simp only [e_2019_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15784)) := by
  have c123162 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123163 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123164 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123165 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123166 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123167 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_10
  simp only [← c123162, ← c123163, ← c123164] at e_2019_10
  have e_2019_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_11
  simp only [← c123165, ← c123166, k2, ← c123167] at e_2019_11
  have hr := e_2019_11
  simp only [e_2019_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15785)) := by
  have c123168 := (publicBatchWrapper89_copies3849 a h).1
  have c123169 := (publicBatchWrapper89_copies3849 a h).2.1
  have c123170 := (publicBatchWrapper89_copies3849 a h).2.2.1
  have c123171 := (publicBatchWrapper89_copies3849 a h).2.2.2.1
  have c123172 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.1
  have c123173 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_12
  simp only [← c123168, ← c123169, ← c123170] at e_2019_12
  have e_2019_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_13
  simp only [← c123171, ← c123172, k2, ← c123173] at e_2019_13
  have hr := e_2019_13
  simp only [e_2019_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15786)) := by
  have c123174 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.1
  have c123175 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.1
  have c123176 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.1
  have c123177 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.1
  have c123178 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123179 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_14
  simp only [← c123174, ← c123175, ← c123176] at e_2019_14
  have e_2019_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_15
  simp only [← c123177, ← c123178, k2, ← c123179] at e_2019_15
  have hr := e_2019_15
  simp only [e_2019_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15787)) := by
  have c123180 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123181 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123182 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123183 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123184 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123185 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_16
  simp only [← c123180, ← c123181, ← c123182] at e_2019_16
  have e_2019_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_17
  simp only [← c123183, ← c123184, k2, ← c123185] at e_2019_17
  have hr := e_2019_17
  simp only [e_2019_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15788)) := by
  have c123186 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123187 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123188 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123189 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123190 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123191 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_18
  simp only [← c123186, ← c123187, ← c123188] at e_2019_18
  have e_2019_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2019 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_19
  simp only [← c123189, ← c123190, k2, ← c123191] at e_2019_19
  have hr := e_2019_19
  simp only [e_2019_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15789)) := by
  have c123192 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123193 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123194 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123195 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123196 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123197 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_0
  simp only [← c123192, ← c123193, ← c123194] at e_2020_0
  have e_2020_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_1
  simp only [← c123195, ← c123196, k2, ← c123197] at e_2020_1
  have hr := e_2020_1
  simp only [e_2020_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15790)) := by
  have c123198 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123199 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123200 := (publicBatchWrapper89_copies3850 a h).1
  have c123201 := (publicBatchWrapper89_copies3850 a h).2.1
  have c123202 := (publicBatchWrapper89_copies3850 a h).2.2.1
  have c123203 := (publicBatchWrapper89_copies3850 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_2
  simp only [← c123198, ← c123199, ← c123200] at e_2020_2
  have e_2020_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_3
  simp only [← c123201, ← c123202, k2, ← c123203] at e_2020_3
  have hr := e_2020_3
  simp only [e_2020_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15791)) := by
  have c123204 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.1
  have c123205 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.1
  have c123206 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.1
  have c123207 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.1
  have c123208 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.1
  have c123209 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_4
  simp only [← c123204, ← c123205, ← c123206] at e_2020_4
  have e_2020_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_5
  simp only [← c123207, ← c123208, k2, ← c123209] at e_2020_5
  have hr := e_2020_5
  simp only [e_2020_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15792)) := by
  have c123210 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123211 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123212 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123213 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123214 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123215 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_6
  simp only [← c123210, ← c123211, ← c123212] at e_2020_6
  have e_2020_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_7
  simp only [← c123213, ← c123214, k2, ← c123215] at e_2020_7
  have hr := e_2020_7
  simp only [e_2020_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15793)) := by
  have c123216 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123217 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123218 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123219 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123220 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123221 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_8
  simp only [← c123216, ← c123217, ← c123218] at e_2020_8
  have e_2020_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_9
  simp only [← c123219, ← c123220, k2, ← c123221] at e_2020_9
  have hr := e_2020_9
  simp only [e_2020_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15794)) := by
  have c123222 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123223 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123224 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123225 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123226 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123227 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_10
  simp only [← c123222, ← c123223, ← c123224] at e_2020_10
  have e_2020_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_11
  simp only [← c123225, ← c123226, k2, ← c123227] at e_2020_11
  have hr := e_2020_11
  simp only [e_2020_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15795)) := by
  have c123228 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123229 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123230 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123231 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123232 := (publicBatchWrapper89_copies3851 a h).1
  have c123233 := (publicBatchWrapper89_copies3851 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_12
  simp only [← c123228, ← c123229, ← c123230] at e_2020_12
  have e_2020_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_13
  simp only [← c123231, ← c123232, k2, ← c123233] at e_2020_13
  have hr := e_2020_13
  simp only [e_2020_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15796)) := by
  have c123234 := (publicBatchWrapper89_copies3851 a h).2.2.1
  have c123235 := (publicBatchWrapper89_copies3851 a h).2.2.2.1
  have c123236 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.1
  have c123237 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.1
  have c123238 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.1
  have c123239 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_14
  simp only [← c123234, ← c123235, ← c123236] at e_2020_14
  have e_2020_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_15
  simp only [← c123237, ← c123238, k2, ← c123239] at e_2020_15
  have hr := e_2020_15
  simp only [e_2020_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15797)) := by
  have c123240 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.1
  have c123241 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.1
  have c123242 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123243 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123244 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123245 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_16
  simp only [← c123240, ← c123241, ← c123242] at e_2020_16
  have e_2020_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_17
  simp only [← c123243, ← c123244, k2, ← c123245] at e_2020_17
  have hr := e_2020_17
  simp only [e_2020_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15798)) := by
  have c123246 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123247 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123248 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123249 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123250 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123251 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_18
  simp only [← c123246, ← c123247, ← c123248] at e_2020_18
  have e_2020_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_19
  simp only [← c123249, ← c123250, k2, ← c123251] at e_2020_19
  have hr := e_2020_19
  simp only [e_2020_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15799)) := by
  have c123252 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123253 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123254 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123255 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123256 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123257 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_0
  simp only [← c123252, ← c123253, ← c123254] at e_2021_0
  have e_2021_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_1
  simp only [← c123255, ← c123256, k2, ← c123257] at e_2021_1
  have hr := e_2021_1
  simp only [e_2021_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15800)) := by
  have c123258 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123259 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123260 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123261 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123262 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123263 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_2
  simp only [← c123258, ← c123259, ← c123260] at e_2021_2
  have e_2021_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_3
  simp only [← c123261, ← c123262, k2, ← c123263] at e_2021_3
  have hr := e_2021_3
  simp only [e_2021_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15801)) := by
  have c123264 := (publicBatchWrapper89_copies3852 a h).1
  have c123265 := (publicBatchWrapper89_copies3852 a h).2.1
  have c123266 := (publicBatchWrapper89_copies3852 a h).2.2.1
  have c123267 := (publicBatchWrapper89_copies3852 a h).2.2.2.1
  have c123268 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.1
  have c123269 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_4
  simp only [← c123264, ← c123265, ← c123266] at e_2021_4
  have e_2021_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_5
  simp only [← c123267, ← c123268, k2, ← c123269] at e_2021_5
  have hr := e_2021_5
  simp only [e_2021_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15970)) := by
  have c123270 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.1
  have c123271 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.1
  have c123272 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.1
  have c123273 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.1
  have c123274 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123275 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_6
  simp only [← c123270, ← c123271, ← c123272] at e_2021_6
  have e_2021_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_7
  simp only [← c123273, ← c123274, k2, ← c123275] at e_2021_7
  have hr := e_2021_7
  simp only [e_2021_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15971)) := by
  have c123276 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123277 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123278 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123279 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123280 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123281 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_8
  simp only [← c123276, ← c123277, ← c123278] at e_2021_8
  have e_2021_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_9
  simp only [← c123279, ← c123280, k2, ← c123281] at e_2021_9
  have hr := e_2021_9
  simp only [e_2021_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15972)) := by
  have c123282 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123283 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123284 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123285 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123286 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123287 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_10
  simp only [← c123282, ← c123283, ← c123284] at e_2021_10
  have e_2021_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_11
  simp only [← c123285, ← c123286, k2, ← c123287] at e_2021_11
  have hr := e_2021_11
  simp only [e_2021_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15973)) := by
  have c123288 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123289 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123290 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123291 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123292 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123293 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_12
  simp only [← c123288, ← c123289, ← c123290] at e_2021_12
  have e_2021_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_13
  simp only [← c123291, ← c123292, k2, ← c123293] at e_2021_13
  have hr := e_2021_13
  simp only [e_2021_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15974)) := by
  have c123294 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123295 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123296 := (publicBatchWrapper89_copies3853 a h).1
  have c123297 := (publicBatchWrapper89_copies3853 a h).2.1
  have c123298 := (publicBatchWrapper89_copies3853 a h).2.2.1
  have c123299 := (publicBatchWrapper89_copies3853 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_14
  simp only [← c123294, ← c123295, ← c123296] at e_2021_14
  have e_2021_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_15
  simp only [← c123297, ← c123298, k2, ← c123299] at e_2021_15
  have hr := e_2021_15
  simp only [e_2021_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15975)) := by
  have c123300 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.1
  have c123301 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.1
  have c123302 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.1
  have c123303 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.1
  have c123304 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.1
  have c123305 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_16
  simp only [← c123300, ← c123301, ← c123302] at e_2021_16
  have e_2021_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_17
  simp only [← c123303, ← c123304, k2, ← c123305] at e_2021_17
  have hr := e_2021_17
  simp only [e_2021_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15976)) := by
  have c123306 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123307 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123308 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123309 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123310 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123311 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_18
  simp only [← c123306, ← c123307, ← c123308] at e_2021_18
  have e_2021_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_19
  simp only [← c123309, ← c123310, k2, ← c123311] at e_2021_19
  have hr := e_2021_19
  simp only [e_2021_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15977)) := by
  have c123312 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123313 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123314 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123315 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123316 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123317 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_0
  simp only [← c123312, ← c123313, ← c123314] at e_2022_0
  have e_2022_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_1
  simp only [← c123315, ← c123316, k2, ← c123317] at e_2022_1
  have hr := e_2022_1
  simp only [e_2022_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15978)) := by
  have c123318 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123319 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123320 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123321 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123322 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123323 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_2
  simp only [← c123318, ← c123319, ← c123320] at e_2022_2
  have e_2022_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_3
  simp only [← c123321, ← c123322, k2, ← c123323] at e_2022_3
  have hr := e_2022_3
  simp only [e_2022_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15979)) := by
  have c123324 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123325 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123326 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123327 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123328 := (publicBatchWrapper89_copies3854 a h).1
  have c123329 := (publicBatchWrapper89_copies3854 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_4
  simp only [← c123324, ← c123325, ← c123326] at e_2022_4
  have e_2022_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_5
  simp only [← c123327, ← c123328, k2, ← c123329] at e_2022_5
  have hr := e_2022_5
  simp only [e_2022_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15980)) := by
  have c123330 := (publicBatchWrapper89_copies3854 a h).2.2.1
  have c123331 := (publicBatchWrapper89_copies3854 a h).2.2.2.1
  have c123332 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.1
  have c123333 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.1
  have c123334 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.1
  have c123335 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_6
  simp only [← c123330, ← c123331, ← c123332] at e_2022_6
  have e_2022_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_7
  simp only [← c123333, ← c123334, k2, ← c123335] at e_2022_7
  have hr := e_2022_7
  simp only [e_2022_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15981)) := by
  have c123336 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.1
  have c123337 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.1
  have c123338 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123339 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123340 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123341 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_8
  simp only [← c123336, ← c123337, ← c123338] at e_2022_8
  have e_2022_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_9
  simp only [← c123339, ← c123340, k2, ← c123341] at e_2022_9
  have hr := e_2022_9
  simp only [e_2022_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15982)) := by
  have c123342 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123343 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123344 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123345 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123346 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123347 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_10
  simp only [← c123342, ← c123343, ← c123344] at e_2022_10
  have e_2022_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_11
  simp only [← c123345, ← c123346, k2, ← c123347] at e_2022_11
  have hr := e_2022_11
  simp only [e_2022_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15983)) := by
  have c123348 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123349 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123350 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123351 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123352 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123353 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_12
  simp only [← c123348, ← c123349, ← c123350] at e_2022_12
  have e_2022_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_13
  simp only [← c123351, ← c123352, k2, ← c123353] at e_2022_13
  have hr := e_2022_13
  simp only [e_2022_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15984)) := by
  have c123354 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123355 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123356 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123357 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123358 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123359 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_14
  simp only [← c123354, ← c123355, ← c123356] at e_2022_14
  have e_2022_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_15
  simp only [← c123357, ← c123358, k2, ← c123359] at e_2022_15
  have hr := e_2022_15
  simp only [e_2022_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15985)) := by
  have c123360 := (publicBatchWrapper89_copies3855 a h).1
  have c123361 := (publicBatchWrapper89_copies3855 a h).2.1
  have c123362 := (publicBatchWrapper89_copies3855 a h).2.2.1
  have c123363 := (publicBatchWrapper89_copies3855 a h).2.2.2.1
  have c123364 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.1
  have c123365 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_16
  simp only [← c123360, ← c123361, ← c123362] at e_2022_16
  have e_2022_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_17
  simp only [← c123363, ← c123364, k2, ← c123365] at e_2022_17
  have hr := e_2022_17
  simp only [e_2022_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15986)) := by
  have c123366 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.1
  have c123367 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.1
  have c123368 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.1
  have c123369 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.1
  have c123370 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123371 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_18
  simp only [← c123366, ← c123367, ← c123368] at e_2022_18
  have e_2022_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_19
  simp only [← c123369, ← c123370, k2, ← c123371] at e_2022_19
  have hr := e_2022_19
  simp only [e_2022_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15987)) := by
  have c123372 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123373 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123374 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123375 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123376 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123377 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_0
  simp only [← c123372, ← c123373, ← c123374] at e_2023_0
  have e_2023_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_1
  simp only [← c123375, ← c123376, k2, ← c123377] at e_2023_1
  have hr := e_2023_1
  simp only [e_2023_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15988)) := by
  have c123378 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123379 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123380 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123381 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123382 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123383 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_2
  simp only [← c123378, ← c123379, ← c123380] at e_2023_2
  have e_2023_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_3
  simp only [← c123381, ← c123382, k2, ← c123383] at e_2023_3
  have hr := e_2023_3
  simp only [e_2023_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15989)) := by
  have c123384 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123385 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123386 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123387 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123388 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123389 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_4
  simp only [← c123384, ← c123385, ← c123386] at e_2023_4
  have e_2023_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_5
  simp only [← c123387, ← c123388, k2, ← c123389] at e_2023_5
  have hr := e_2023_5
  simp only [e_2023_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15990)) := by
  have c123390 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123391 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123392 := (publicBatchWrapper89_copies3856 a h).1
  have c123393 := (publicBatchWrapper89_copies3856 a h).2.1
  have c123394 := (publicBatchWrapper89_copies3856 a h).2.2.1
  have c123395 := (publicBatchWrapper89_copies3856 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_6
  simp only [← c123390, ← c123391, ← c123392] at e_2023_6
  have e_2023_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_7
  simp only [← c123393, ← c123394, k2, ← c123395] at e_2023_7
  have hr := e_2023_7
  simp only [e_2023_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15991)) := by
  have c123396 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.1
  have c123397 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.1
  have c123398 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.1
  have c123399 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.1
  have c123400 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.1
  have c123401 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_8
  simp only [← c123396, ← c123397, ← c123398] at e_2023_8
  have e_2023_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_9
  simp only [← c123399, ← c123400, k2, ← c123401] at e_2023_9
  have hr := e_2023_9
  simp only [e_2023_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15992)) := by
  have c123402 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123403 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123404 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123405 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123406 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123407 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_10
  simp only [← c123402, ← c123403, ← c123404] at e_2023_10
  have e_2023_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_11
  simp only [← c123405, ← c123406, k2, ← c123407] at e_2023_11
  have hr := e_2023_11
  simp only [e_2023_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15993)) := by
  have c123408 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123409 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123410 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123411 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123412 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123413 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_12
  simp only [← c123408, ← c123409, ← c123410] at e_2023_12
  have e_2023_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_13
  simp only [← c123411, ← c123412, k2, ← c123413] at e_2023_13
  have hr := e_2023_13
  simp only [e_2023_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15994)) := by
  have c123414 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123415 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123416 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123417 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123418 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123419 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_14
  simp only [← c123414, ← c123415, ← c123416] at e_2023_14
  have e_2023_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_15
  simp only [← c123417, ← c123418, k2, ← c123419] at e_2023_15
  have hr := e_2023_15
  simp only [e_2023_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15995)) := by
  have c123420 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123421 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123422 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123423 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123424 := (publicBatchWrapper89_copies3857 a h).1
  have c123425 := (publicBatchWrapper89_copies3857 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_16
  simp only [← c123420, ← c123421, ← c123422] at e_2023_16
  have e_2023_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_17
  simp only [← c123423, ← c123424, k2, ← c123425] at e_2023_17
  have hr := e_2023_17
  simp only [e_2023_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15996)) := by
  have c123426 := (publicBatchWrapper89_copies3857 a h).2.2.1
  have c123427 := (publicBatchWrapper89_copies3857 a h).2.2.2.1
  have c123428 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.1
  have c123429 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.1
  have c123430 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.1
  have c123431 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_18
  simp only [← c123426, ← c123427, ← c123428] at e_2023_18
  have e_2023_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_19
  simp only [← c123429, ← c123430, k2, ← c123431] at e_2023_19
  have hr := e_2023_19
  simp only [e_2023_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15997)) := by
  have c123432 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.1
  have c123433 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.1
  have c123434 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123435 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123436 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123437 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_0
  simp only [← c123432, ← c123433, ← c123434] at e_2024_0
  have e_2024_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_1
  simp only [← c123435, ← c123436, k2, ← c123437] at e_2024_1
  have hr := e_2024_1
  simp only [e_2024_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15998)) := by
  have c123438 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123439 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123440 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123441 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123442 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123443 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_2
  simp only [← c123438, ← c123439, ← c123440] at e_2024_2
  have e_2024_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_3
  simp only [← c123441, ← c123442, k2, ← c123443] at e_2024_3
  have hr := e_2024_3
  simp only [e_2024_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15999)) := by
  have c123444 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123445 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123446 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123447 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123448 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123449 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_4
  simp only [← c123444, ← c123445, ← c123446] at e_2024_4
  have e_2024_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_5
  simp only [← c123447, ← c123448, k2, ← c123449] at e_2024_5
  have hr := e_2024_5
  simp only [e_2024_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 16000)) := by
  have c123450 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123451 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123452 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123453 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123454 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123455 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_6
  simp only [← c123450, ← c123451, ← c123452] at e_2024_6
  have e_2024_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_7
  simp only [← c123453, ← c123454, k2, ← c123455] at e_2024_7
  have hr := e_2024_7
  simp only [e_2024_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 16001)) := by
  have c123456 := (publicBatchWrapper89_copies3858 a h).1
  have c123457 := (publicBatchWrapper89_copies3858 a h).2.1
  have c123458 := (publicBatchWrapper89_copies3858 a h).2.2.1
  have c123459 := (publicBatchWrapper89_copies3858 a h).2.2.2.1
  have c123460 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.1
  have c123461 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_8
  simp only [← c123456, ← c123457, ← c123458] at e_2024_8
  have e_2024_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_9
  simp only [← c123459, ← c123460, k2, ← c123461] at e_2024_9
  have hr := e_2024_9
  simp only [e_2024_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16170)) := by
  have c123462 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.1
  have c123463 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.1
  have c123464 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.1
  have c123465 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.1
  have c123466 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123467 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_10
  simp only [← c123462, ← c123463, ← c123464] at e_2024_10
  have e_2024_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_11
  simp only [← c123465, ← c123466, k2, ← c123467] at e_2024_11
  have hr := e_2024_11
  simp only [e_2024_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16171)) := by
  have c123468 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123469 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123470 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123471 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123472 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123473 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_12
  simp only [← c123468, ← c123469, ← c123470] at e_2024_12
  have e_2024_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_13
  simp only [← c123471, ← c123472, k2, ← c123473] at e_2024_13
  have hr := e_2024_13
  simp only [e_2024_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16172)) := by
  have c123474 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123475 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123476 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123477 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123478 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123479 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_14
  simp only [← c123474, ← c123475, ← c123476] at e_2024_14
  have e_2024_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_15
  simp only [← c123477, ← c123478, k2, ← c123479] at e_2024_15
  have hr := e_2024_15
  simp only [e_2024_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16173)) := by
  have c123480 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123481 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123482 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123483 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123484 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123485 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_16
  simp only [← c123480, ← c123481, ← c123482] at e_2024_16
  have e_2024_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_17
  simp only [← c123483, ← c123484, k2, ← c123485] at e_2024_17
  have hr := e_2024_17
  simp only [e_2024_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16174)) := by
  have c123486 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123487 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123488 := (publicBatchWrapper89_copies3859 a h).1
  have c123489 := (publicBatchWrapper89_copies3859 a h).2.1
  have c123490 := (publicBatchWrapper89_copies3859 a h).2.2.1
  have c123491 := (publicBatchWrapper89_copies3859 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_18
  simp only [← c123486, ← c123487, ← c123488] at e_2024_18
  have e_2024_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_19
  simp only [← c123489, ← c123490, k2, ← c123491] at e_2024_19
  have hr := e_2024_19
  simp only [e_2024_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16175)) := by
  have c123492 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.1
  have c123493 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.1
  have c123494 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.1
  have c123495 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.1
  have c123496 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.1
  have c123497 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_0
  simp only [← c123492, ← c123493, ← c123494] at e_2025_0
  have e_2025_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_1
  simp only [← c123495, ← c123496, k2, ← c123497] at e_2025_1
  have hr := e_2025_1
  simp only [e_2025_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16176)) := by
  have c123498 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123499 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123500 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123501 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123502 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123503 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_2
  simp only [← c123498, ← c123499, ← c123500] at e_2025_2
  have e_2025_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_3
  simp only [← c123501, ← c123502, k2, ← c123503] at e_2025_3
  have hr := e_2025_3
  simp only [e_2025_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16177)) := by
  have c123504 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123505 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123506 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123507 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123508 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123509 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_4
  simp only [← c123504, ← c123505, ← c123506] at e_2025_4
  have e_2025_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_5
  simp only [← c123507, ← c123508, k2, ← c123509] at e_2025_5
  have hr := e_2025_5
  simp only [e_2025_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16178)) := by
  have c123510 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123511 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123512 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123513 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123514 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123515 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_6
  simp only [← c123510, ← c123511, ← c123512] at e_2025_6
  have e_2025_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_7
  simp only [← c123513, ← c123514, k2, ← c123515] at e_2025_7
  have hr := e_2025_7
  simp only [e_2025_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16179)) := by
  have c123516 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123517 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123518 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123519 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123520 := (publicBatchWrapper89_copies3860 a h).1
  have c123521 := (publicBatchWrapper89_copies3860 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_8
  simp only [← c123516, ← c123517, ← c123518] at e_2025_8
  have e_2025_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_9
  simp only [← c123519, ← c123520, k2, ← c123521] at e_2025_9
  have hr := e_2025_9
  simp only [e_2025_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16180)) := by
  have c123522 := (publicBatchWrapper89_copies3860 a h).2.2.1
  have c123523 := (publicBatchWrapper89_copies3860 a h).2.2.2.1
  have c123524 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.1
  have c123525 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.1
  have c123526 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.1
  have c123527 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_10
  simp only [← c123522, ← c123523, ← c123524] at e_2025_10
  have e_2025_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_11
  simp only [← c123525, ← c123526, k2, ← c123527] at e_2025_11
  have hr := e_2025_11
  simp only [e_2025_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16181)) := by
  have c123528 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.1
  have c123529 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.1
  have c123530 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123531 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123532 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123533 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_12
  simp only [← c123528, ← c123529, ← c123530] at e_2025_12
  have e_2025_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_13
  simp only [← c123531, ← c123532, k2, ← c123533] at e_2025_13
  have hr := e_2025_13
  simp only [e_2025_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16182)) := by
  have c123534 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123535 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123536 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123537 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123538 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123539 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_14
  simp only [← c123534, ← c123535, ← c123536] at e_2025_14
  have e_2025_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_15
  simp only [← c123537, ← c123538, k2, ← c123539] at e_2025_15
  have hr := e_2025_15
  simp only [e_2025_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16183)) := by
  have c123540 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123541 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123542 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123543 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123544 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123545 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_16
  simp only [← c123540, ← c123541, ← c123542] at e_2025_16
  have e_2025_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_17
  simp only [← c123543, ← c123544, k2, ← c123545] at e_2025_17
  have hr := e_2025_17
  simp only [e_2025_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16184)) := by
  have c123546 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123547 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123548 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123549 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123550 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123551 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_18
  simp only [← c123546, ← c123547, ← c123548] at e_2025_18
  have e_2025_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_19
  simp only [← c123549, ← c123550, k2, ← c123551] at e_2025_19
  have hr := e_2025_19
  simp only [e_2025_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16185)) := by
  have c123552 := (publicBatchWrapper89_copies3861 a h).1
  have c123553 := (publicBatchWrapper89_copies3861 a h).2.1
  have c123554 := (publicBatchWrapper89_copies3861 a h).2.2.1
  have c123555 := (publicBatchWrapper89_copies3861 a h).2.2.2.1
  have c123556 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.1
  have c123557 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_0
  simp only [← c123552, ← c123553, ← c123554] at e_2026_0
  have e_2026_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_1
  simp only [← c123555, ← c123556, k2, ← c123557] at e_2026_1
  have hr := e_2026_1
  simp only [e_2026_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16186)) := by
  have c123558 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.1
  have c123559 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.1
  have c123560 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.1
  have c123561 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.1
  have c123562 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123563 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_2
  simp only [← c123558, ← c123559, ← c123560] at e_2026_2
  have e_2026_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_3
  simp only [← c123561, ← c123562, k2, ← c123563] at e_2026_3
  have hr := e_2026_3
  simp only [e_2026_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16187)) := by
  have c123564 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123565 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123566 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123567 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123568 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123569 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_4
  simp only [← c123564, ← c123565, ← c123566] at e_2026_4
  have e_2026_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_5
  simp only [← c123567, ← c123568, k2, ← c123569] at e_2026_5
  have hr := e_2026_5
  simp only [e_2026_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16188)) := by
  have c123570 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123571 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123572 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123573 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123574 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123575 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_6
  simp only [← c123570, ← c123571, ← c123572] at e_2026_6
  have e_2026_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_7
  simp only [← c123573, ← c123574, k2, ← c123575] at e_2026_7
  have hr := e_2026_7
  simp only [e_2026_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16189)) := by
  have c123576 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123577 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123578 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123579 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123580 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123581 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_8
  simp only [← c123576, ← c123577, ← c123578] at e_2026_8
  have e_2026_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_9
  simp only [← c123579, ← c123580, k2, ← c123581] at e_2026_9
  have hr := e_2026_9
  simp only [e_2026_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16190)) := by
  have c123582 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123583 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123584 := (publicBatchWrapper89_copies3862 a h).1
  have c123585 := (publicBatchWrapper89_copies3862 a h).2.1
  have c123586 := (publicBatchWrapper89_copies3862 a h).2.2.1
  have c123587 := (publicBatchWrapper89_copies3862 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_10
  simp only [← c123582, ← c123583, ← c123584] at e_2026_10
  have e_2026_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_11
  simp only [← c123585, ← c123586, k2, ← c123587] at e_2026_11
  have hr := e_2026_11
  simp only [e_2026_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16191)) := by
  have c123588 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.1
  have c123589 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.1
  have c123590 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.1
  have c123591 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.1
  have c123592 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.1
  have c123593 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_12
  simp only [← c123588, ← c123589, ← c123590] at e_2026_12
  have e_2026_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_13
  simp only [← c123591, ← c123592, k2, ← c123593] at e_2026_13
  have hr := e_2026_13
  simp only [e_2026_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16192)) := by
  have c123594 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123595 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123596 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123597 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123598 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123599 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_14
  simp only [← c123594, ← c123595, ← c123596] at e_2026_14
  have e_2026_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_15
  simp only [← c123597, ← c123598, k2, ← c123599] at e_2026_15
  have hr := e_2026_15
  simp only [e_2026_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16193)) := by
  have c123600 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123601 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123602 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123603 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123604 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123605 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_16
  simp only [← c123600, ← c123601, ← c123602] at e_2026_16
  have e_2026_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_17
  simp only [← c123603, ← c123604, k2, ← c123605] at e_2026_17
  have hr := e_2026_17
  simp only [e_2026_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16194)) := by
  have c123606 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123607 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123608 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123609 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123610 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123611 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_18
  simp only [← c123606, ← c123607, ← c123608] at e_2026_18
  have e_2026_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_19
  simp only [← c123609, ← c123610, k2, ← c123611] at e_2026_19
  have hr := e_2026_19
  simp only [e_2026_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16195)) := by
  have c123612 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123613 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123614 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123615 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123616 := (publicBatchWrapper89_copies3863 a h).1
  have c123617 := (publicBatchWrapper89_copies3863 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_0
  simp only [← c123612, ← c123613, ← c123614] at e_2027_0
  have e_2027_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_1
  simp only [← c123615, ← c123616, k2, ← c123617] at e_2027_1
  have hr := e_2027_1
  simp only [e_2027_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16196)) := by
  have c123618 := (publicBatchWrapper89_copies3863 a h).2.2.1
  have c123619 := (publicBatchWrapper89_copies3863 a h).2.2.2.1
  have c123620 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.1
  have c123621 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.1
  have c123622 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.1
  have c123623 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_2
  simp only [← c123618, ← c123619, ← c123620] at e_2027_2
  have e_2027_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_3
  simp only [← c123621, ← c123622, k2, ← c123623] at e_2027_3
  have hr := e_2027_3
  simp only [e_2027_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16197)) := by
  have c123624 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.1
  have c123625 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.1
  have c123626 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123627 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123628 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123629 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_4
  simp only [← c123624, ← c123625, ← c123626] at e_2027_4
  have e_2027_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_5
  simp only [← c123627, ← c123628, k2, ← c123629] at e_2027_5
  have hr := e_2027_5
  simp only [e_2027_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16198)) := by
  have c123630 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123631 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123632 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123633 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123634 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123635 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_6
  simp only [← c123630, ← c123631, ← c123632] at e_2027_6
  have e_2027_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_7
  simp only [← c123633, ← c123634, k2, ← c123635] at e_2027_7
  have hr := e_2027_7
  simp only [e_2027_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16199)) := by
  have c123636 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123637 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123638 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123639 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123640 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123641 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_8
  simp only [← c123636, ← c123637, ← c123638] at e_2027_8
  have e_2027_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_9
  simp only [← c123639, ← c123640, k2, ← c123641] at e_2027_9
  have hr := e_2027_9
  simp only [e_2027_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16200)) := by
  have c123642 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123643 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123644 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123645 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123646 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123647 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_10
  simp only [← c123642, ← c123643, ← c123644] at e_2027_10
  have e_2027_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_11
  simp only [← c123645, ← c123646, k2, ← c123647] at e_2027_11
  have hr := e_2027_11
  simp only [e_2027_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16201)) := by
  have c123648 := (publicBatchWrapper89_copies3864 a h).1
  have c123649 := (publicBatchWrapper89_copies3864 a h).2.1
  have c123650 := (publicBatchWrapper89_copies3864 a h).2.2.1
  have c123651 := (publicBatchWrapper89_copies3864 a h).2.2.2.1
  have c123652 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.1
  have c123653 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_12
  simp only [← c123648, ← c123649, ← c123650] at e_2027_12
  have e_2027_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_13
  simp only [← c123651, ← c123652, k2, ← c123653] at e_2027_13
  have hr := e_2027_13
  simp only [e_2027_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16370)) := by
  have c123654 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.1
  have c123655 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.1
  have c123656 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.1
  have c123657 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.1
  have c123658 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123659 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_14
  simp only [← c123654, ← c123655, ← c123656] at e_2027_14
  have e_2027_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_15
  simp only [← c123657, ← c123658, k2, ← c123659] at e_2027_15
  have hr := e_2027_15
  simp only [e_2027_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16371)) := by
  have c123660 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123661 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123662 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123663 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123664 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123665 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_16
  simp only [← c123660, ← c123661, ← c123662] at e_2027_16
  have e_2027_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_17
  simp only [← c123663, ← c123664, k2, ← c123665] at e_2027_17
  have hr := e_2027_17
  simp only [e_2027_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16372)) := by
  have c123666 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123667 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123668 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123669 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123670 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123671 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_18
  simp only [← c123666, ← c123667, ← c123668] at e_2027_18
  have e_2027_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_19
  simp only [← c123669, ← c123670, k2, ← c123671] at e_2027_19
  have hr := e_2027_19
  simp only [e_2027_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16373)) := by
  have c123672 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123673 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123674 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123675 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123676 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123677 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_0
  simp only [← c123672, ← c123673, ← c123674] at e_2028_0
  have e_2028_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_1
  simp only [← c123675, ← c123676, k2, ← c123677] at e_2028_1
  have hr := e_2028_1
  simp only [e_2028_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16374)) := by
  have c123678 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123679 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123680 := (publicBatchWrapper89_copies3865 a h).1
  have c123681 := (publicBatchWrapper89_copies3865 a h).2.1
  have c123682 := (publicBatchWrapper89_copies3865 a h).2.2.1
  have c123683 := (publicBatchWrapper89_copies3865 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_2
  simp only [← c123678, ← c123679, ← c123680] at e_2028_2
  have e_2028_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_3
  simp only [← c123681, ← c123682, k2, ← c123683] at e_2028_3
  have hr := e_2028_3
  simp only [e_2028_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16375)) := by
  have c123684 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.1
  have c123685 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.1
  have c123686 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.1
  have c123687 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.1
  have c123688 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.1
  have c123689 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_4
  simp only [← c123684, ← c123685, ← c123686] at e_2028_4
  have e_2028_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_5
  simp only [← c123687, ← c123688, k2, ← c123689] at e_2028_5
  have hr := e_2028_5
  simp only [e_2028_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16376)) := by
  have c123690 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123691 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123692 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123693 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123694 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123695 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_6
  simp only [← c123690, ← c123691, ← c123692] at e_2028_6
  have e_2028_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_7
  simp only [← c123693, ← c123694, k2, ← c123695] at e_2028_7
  have hr := e_2028_7
  simp only [e_2028_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16377)) := by
  have c123696 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123697 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123698 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123699 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123700 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123701 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_8
  simp only [← c123696, ← c123697, ← c123698] at e_2028_8
  have e_2028_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_9
  simp only [← c123699, ← c123700, k2, ← c123701] at e_2028_9
  have hr := e_2028_9
  simp only [e_2028_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16378)) := by
  have c123702 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123703 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123704 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123705 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123706 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123707 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_10
  simp only [← c123702, ← c123703, ← c123704] at e_2028_10
  have e_2028_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_11
  simp only [← c123705, ← c123706, k2, ← c123707] at e_2028_11
  have hr := e_2028_11
  simp only [e_2028_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16379)) := by
  have c123708 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123709 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123710 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123711 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123712 := (publicBatchWrapper89_copies3866 a h).1
  have c123713 := (publicBatchWrapper89_copies3866 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_12
  simp only [← c123708, ← c123709, ← c123710] at e_2028_12
  have e_2028_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_13
  simp only [← c123711, ← c123712, k2, ← c123713] at e_2028_13
  have hr := e_2028_13
  simp only [e_2028_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16380)) := by
  have c123714 := (publicBatchWrapper89_copies3866 a h).2.2.1
  have c123715 := (publicBatchWrapper89_copies3866 a h).2.2.2.1
  have c123716 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.1
  have c123717 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.1
  have c123718 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.1
  have c123719 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_14
  simp only [← c123714, ← c123715, ← c123716] at e_2028_14
  have e_2028_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_15
  simp only [← c123717, ← c123718, k2, ← c123719] at e_2028_15
  have hr := e_2028_15
  simp only [e_2028_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16381)) := by
  have c123720 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.1
  have c123721 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.1
  have c123722 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123723 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123724 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123725 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_16
  simp only [← c123720, ← c123721, ← c123722] at e_2028_16
  have e_2028_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_17
  simp only [← c123723, ← c123724, k2, ← c123725] at e_2028_17
  have hr := e_2028_17
  simp only [e_2028_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16382)) := by
  have c123726 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123727 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123728 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123729 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123730 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123731 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_18
  simp only [← c123726, ← c123727, ← c123728] at e_2028_18
  have e_2028_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_19
  simp only [← c123729, ← c123730, k2, ← c123731] at e_2028_19
  have hr := e_2028_19
  simp only [e_2028_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16383)) := by
  have c123732 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123733 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123734 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123735 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123736 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123737 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_0
  simp only [← c123732, ← c123733, ← c123734] at e_2029_0
  have e_2029_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_1
  simp only [← c123735, ← c123736, k2, ← c123737] at e_2029_1
  have hr := e_2029_1
  simp only [e_2029_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16384)) := by
  have c123738 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123739 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123740 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123741 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123742 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123743 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_2
  simp only [← c123738, ← c123739, ← c123740] at e_2029_2
  have e_2029_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_3
  simp only [← c123741, ← c123742, k2, ← c123743] at e_2029_3
  have hr := e_2029_3
  simp only [e_2029_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16385)) := by
  have c123744 := (publicBatchWrapper89_copies3867 a h).1
  have c123745 := (publicBatchWrapper89_copies3867 a h).2.1
  have c123746 := (publicBatchWrapper89_copies3867 a h).2.2.1
  have c123747 := (publicBatchWrapper89_copies3867 a h).2.2.2.1
  have c123748 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.1
  have c123749 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_4
  simp only [← c123744, ← c123745, ← c123746] at e_2029_4
  have e_2029_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_5
  simp only [← c123747, ← c123748, k2, ← c123749] at e_2029_5
  have hr := e_2029_5
  simp only [e_2029_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16386)) := by
  have c123750 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.1
  have c123751 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.1
  have c123752 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.1
  have c123753 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.1
  have c123754 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123755 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_6
  simp only [← c123750, ← c123751, ← c123752] at e_2029_6
  have e_2029_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_7
  simp only [← c123753, ← c123754, k2, ← c123755] at e_2029_7
  have hr := e_2029_7
  simp only [e_2029_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16387)) := by
  have c123756 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123757 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123758 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123759 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123760 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123761 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_8
  simp only [← c123756, ← c123757, ← c123758] at e_2029_8
  have e_2029_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_9
  simp only [← c123759, ← c123760, k2, ← c123761] at e_2029_9
  have hr := e_2029_9
  simp only [e_2029_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16388)) := by
  have c123762 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123763 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123764 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123765 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123766 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123767 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_10
  simp only [← c123762, ← c123763, ← c123764] at e_2029_10
  have e_2029_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_11
  simp only [← c123765, ← c123766, k2, ← c123767] at e_2029_11
  have hr := e_2029_11
  simp only [e_2029_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16389)) := by
  have c123768 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123769 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123770 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123771 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123772 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123773 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_12
  simp only [← c123768, ← c123769, ← c123770] at e_2029_12
  have e_2029_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_13
  simp only [← c123771, ← c123772, k2, ← c123773] at e_2029_13
  have hr := e_2029_13
  simp only [e_2029_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16390)) := by
  have c123774 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123775 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123776 := (publicBatchWrapper89_copies3868 a h).1
  have c123777 := (publicBatchWrapper89_copies3868 a h).2.1
  have c123778 := (publicBatchWrapper89_copies3868 a h).2.2.1
  have c123779 := (publicBatchWrapper89_copies3868 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_14
  simp only [← c123774, ← c123775, ← c123776] at e_2029_14
  have e_2029_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_15
  simp only [← c123777, ← c123778, k2, ← c123779] at e_2029_15
  have hr := e_2029_15
  simp only [e_2029_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16391)) := by
  have c123780 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.1
  have c123781 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.1
  have c123782 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.1
  have c123783 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.1
  have c123784 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.1
  have c123785 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_16
  simp only [← c123780, ← c123781, ← c123782] at e_2029_16
  have e_2029_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_17
  simp only [← c123783, ← c123784, k2, ← c123785] at e_2029_17
  have hr := e_2029_17
  simp only [e_2029_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16392)) := by
  have c123786 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123787 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123788 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123789 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123790 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123791 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_18
  simp only [← c123786, ← c123787, ← c123788] at e_2029_18
  have e_2029_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_19
  simp only [← c123789, ← c123790, k2, ← c123791] at e_2029_19
  have hr := e_2029_19
  simp only [e_2029_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16393)) := by
  have c123792 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123793 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123794 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123795 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123796 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123797 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_0
  simp only [← c123792, ← c123793, ← c123794] at e_2030_0
  have e_2030_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_1
  simp only [← c123795, ← c123796, k2, ← c123797] at e_2030_1
  have hr := e_2030_1
  simp only [e_2030_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16394)) := by
  have c123798 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123799 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123800 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123801 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123802 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123803 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_2
  simp only [← c123798, ← c123799, ← c123800] at e_2030_2
  have e_2030_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_3
  simp only [← c123801, ← c123802, k2, ← c123803] at e_2030_3
  have hr := e_2030_3
  simp only [e_2030_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16395)) := by
  have c123804 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123805 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123806 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123807 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123808 := (publicBatchWrapper89_copies3869 a h).1
  have c123809 := (publicBatchWrapper89_copies3869 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_4
  simp only [← c123804, ← c123805, ← c123806] at e_2030_4
  have e_2030_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_5
  simp only [← c123807, ← c123808, k2, ← c123809] at e_2030_5
  have hr := e_2030_5
  simp only [e_2030_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16396)) := by
  have c123810 := (publicBatchWrapper89_copies3869 a h).2.2.1
  have c123811 := (publicBatchWrapper89_copies3869 a h).2.2.2.1
  have c123812 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.1
  have c123813 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.1
  have c123814 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.1
  have c123815 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_6
  simp only [← c123810, ← c123811, ← c123812] at e_2030_6
  have e_2030_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_7
  simp only [← c123813, ← c123814, k2, ← c123815] at e_2030_7
  have hr := e_2030_7
  simp only [e_2030_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16397)) := by
  have c123816 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.1
  have c123817 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.1
  have c123818 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123819 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123820 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123821 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_8
  simp only [← c123816, ← c123817, ← c123818] at e_2030_8
  have e_2030_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_9
  simp only [← c123819, ← c123820, k2, ← c123821] at e_2030_9
  have hr := e_2030_9
  simp only [e_2030_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16398)) := by
  have c123822 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123823 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123824 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123825 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123826 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123827 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_10
  simp only [← c123822, ← c123823, ← c123824] at e_2030_10
  have e_2030_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_11
  simp only [← c123825, ← c123826, k2, ← c123827] at e_2030_11
  have hr := e_2030_11
  simp only [e_2030_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16399)) := by
  have c123828 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123829 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123830 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123831 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123832 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123833 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_12
  simp only [← c123828, ← c123829, ← c123830] at e_2030_12
  have e_2030_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_13
  simp only [← c123831, ← c123832, k2, ← c123833] at e_2030_13
  have hr := e_2030_13
  simp only [e_2030_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16400)) := by
  have c123834 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123835 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123836 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123837 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123838 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123839 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_14
  simp only [← c123834, ← c123835, ← c123836] at e_2030_14
  have e_2030_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_15
  simp only [← c123837, ← c123838, k2, ← c123839] at e_2030_15
  have hr := e_2030_15
  simp only [e_2030_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16401)) := by
  have c123840 := (publicBatchWrapper89_copies3870 a h).1
  have c123841 := (publicBatchWrapper89_copies3870 a h).2.1
  have c123842 := (publicBatchWrapper89_copies3870 a h).2.2.1
  have c123843 := (publicBatchWrapper89_copies3870 a h).2.2.2.1
  have c123844 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.1
  have c123845 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_16
  simp only [← c123840, ← c123841, ← c123842] at e_2030_16
  have e_2030_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_17
  simp only [← c123843, ← c123844, k2, ← c123845] at e_2030_17
  have hr := e_2030_17
  simp only [e_2030_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16570)) := by
  have c123846 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.1
  have c123847 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.1
  have c123848 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.1
  have c123849 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.1
  have c123850 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123851 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_18
  simp only [← c123846, ← c123847, ← c123848] at e_2030_18
  have e_2030_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_19
  simp only [← c123849, ← c123850, k2, ← c123851] at e_2030_19
  have hr := e_2030_19
  simp only [e_2030_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16571)) := by
  have c123852 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123853 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123854 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123855 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123856 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123857 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_0
  simp only [← c123852, ← c123853, ← c123854] at e_2031_0
  have e_2031_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_1
  simp only [← c123855, ← c123856, k2, ← c123857] at e_2031_1
  have hr := e_2031_1
  simp only [e_2031_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16572)) := by
  have c123858 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123859 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123860 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123861 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123862 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123863 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_2
  simp only [← c123858, ← c123859, ← c123860] at e_2031_2
  have e_2031_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_3
  simp only [← c123861, ← c123862, k2, ← c123863] at e_2031_3
  have hr := e_2031_3
  simp only [e_2031_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16573)) := by
  have c123864 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123865 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123866 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123867 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123868 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123869 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_4
  simp only [← c123864, ← c123865, ← c123866] at e_2031_4
  have e_2031_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_5
  simp only [← c123867, ← c123868, k2, ← c123869] at e_2031_5
  have hr := e_2031_5
  simp only [e_2031_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16574)) := by
  have c123870 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123871 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123872 := (publicBatchWrapper89_copies3871 a h).1
  have c123873 := (publicBatchWrapper89_copies3871 a h).2.1
  have c123874 := (publicBatchWrapper89_copies3871 a h).2.2.1
  have c123875 := (publicBatchWrapper89_copies3871 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_6
  simp only [← c123870, ← c123871, ← c123872] at e_2031_6
  have e_2031_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_7
  simp only [← c123873, ← c123874, k2, ← c123875] at e_2031_7
  have hr := e_2031_7
  simp only [e_2031_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16575)) := by
  have c123876 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.1
  have c123877 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.1
  have c123878 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.1
  have c123879 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.1
  have c123880 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.1
  have c123881 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_8
  simp only [← c123876, ← c123877, ← c123878] at e_2031_8
  have e_2031_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_9
  simp only [← c123879, ← c123880, k2, ← c123881] at e_2031_9
  have hr := e_2031_9
  simp only [e_2031_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16576)) := by
  have c123882 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123883 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123884 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123885 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123886 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123887 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_10
  simp only [← c123882, ← c123883, ← c123884] at e_2031_10
  have e_2031_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_11
  simp only [← c123885, ← c123886, k2, ← c123887] at e_2031_11
  have hr := e_2031_11
  simp only [e_2031_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16577)) := by
  have c123888 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123889 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123890 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123891 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123892 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123893 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_12
  simp only [← c123888, ← c123889, ← c123890] at e_2031_12
  have e_2031_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_13
  simp only [← c123891, ← c123892, k2, ← c123893] at e_2031_13
  have hr := e_2031_13
  simp only [e_2031_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16578)) := by
  have c123894 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123895 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123896 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123897 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123898 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123899 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_14
  simp only [← c123894, ← c123895, ← c123896] at e_2031_14
  have e_2031_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_15
  simp only [← c123897, ← c123898, k2, ← c123899] at e_2031_15
  have hr := e_2031_15
  simp only [e_2031_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16579)) := by
  have c123900 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123901 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123902 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123903 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123904 := (publicBatchWrapper89_copies3872 a h).1
  have c123905 := (publicBatchWrapper89_copies3872 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_16
  simp only [← c123900, ← c123901, ← c123902] at e_2031_16
  have e_2031_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_17
  simp only [← c123903, ← c123904, k2, ← c123905] at e_2031_17
  have hr := e_2031_17
  simp only [e_2031_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16580)) := by
  have c123906 := (publicBatchWrapper89_copies3872 a h).2.2.1
  have c123907 := (publicBatchWrapper89_copies3872 a h).2.2.2.1
  have c123908 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.1
  have c123909 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.1
  have c123910 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.1
  have c123911 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_18
  simp only [← c123906, ← c123907, ← c123908] at e_2031_18
  have e_2031_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_19
  simp only [← c123909, ← c123910, k2, ← c123911] at e_2031_19
  have hr := e_2031_19
  simp only [e_2031_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16581)) := by
  have c123912 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.1
  have c123913 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.1
  have c123914 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123915 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123916 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123917 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_0
  simp only [← c123912, ← c123913, ← c123914] at e_2032_0
  have e_2032_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_1
  simp only [← c123915, ← c123916, k2, ← c123917] at e_2032_1
  have hr := e_2032_1
  simp only [e_2032_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16582)) := by
  have c123918 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123919 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123920 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123921 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123922 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123923 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_2
  simp only [← c123918, ← c123919, ← c123920] at e_2032_2
  have e_2032_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_3
  simp only [← c123921, ← c123922, k2, ← c123923] at e_2032_3
  have hr := e_2032_3
  simp only [e_2032_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16583)) := by
  have c123924 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123925 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123926 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123927 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123928 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123929 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_4
  simp only [← c123924, ← c123925, ← c123926] at e_2032_4
  have e_2032_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_5
  simp only [← c123927, ← c123928, k2, ← c123929] at e_2032_5
  have hr := e_2032_5
  simp only [e_2032_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16584)) := by
  have c123930 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123931 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123932 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123933 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123934 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123935 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_6
  simp only [← c123930, ← c123931, ← c123932] at e_2032_6
  have e_2032_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_7
  simp only [← c123933, ← c123934, k2, ← c123935] at e_2032_7
  have hr := e_2032_7
  simp only [e_2032_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16585)) := by
  have c123936 := (publicBatchWrapper89_copies3873 a h).1
  have c123937 := (publicBatchWrapper89_copies3873 a h).2.1
  have c123938 := (publicBatchWrapper89_copies3873 a h).2.2.1
  have c123939 := (publicBatchWrapper89_copies3873 a h).2.2.2.1
  have c123940 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.1
  have c123941 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_8
  simp only [← c123936, ← c123937, ← c123938] at e_2032_8
  have e_2032_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_9
  simp only [← c123939, ← c123940, k2, ← c123941] at e_2032_9
  have hr := e_2032_9
  simp only [e_2032_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16586)) := by
  have c123942 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.1
  have c123943 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.1
  have c123944 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.1
  have c123945 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.1
  have c123946 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123947 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_10
  simp only [← c123942, ← c123943, ← c123944] at e_2032_10
  have e_2032_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_11
  simp only [← c123945, ← c123946, k2, ← c123947] at e_2032_11
  have hr := e_2032_11
  simp only [e_2032_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16587)) := by
  have c123948 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123949 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123950 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123951 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123952 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123953 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_12
  simp only [← c123948, ← c123949, ← c123950] at e_2032_12
  have e_2032_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_13
  simp only [← c123951, ← c123952, k2, ← c123953] at e_2032_13
  have hr := e_2032_13
  simp only [e_2032_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16588)) := by
  have c123954 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123955 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123956 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123957 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123958 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123959 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_14
  simp only [← c123954, ← c123955, ← c123956] at e_2032_14
  have e_2032_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_15
  simp only [← c123957, ← c123958, k2, ← c123959] at e_2032_15
  have hr := e_2032_15
  simp only [e_2032_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16589)) := by
  have c123960 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123961 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123962 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123963 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123964 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123965 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_16
  simp only [← c123960, ← c123961, ← c123962] at e_2032_16
  have e_2032_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_17
  simp only [← c123963, ← c123964, k2, ← c123965] at e_2032_17
  have hr := e_2032_17
  simp only [e_2032_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16590)) := by
  have c123966 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123967 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123968 := (publicBatchWrapper89_copies3874 a h).1
  have c123969 := (publicBatchWrapper89_copies3874 a h).2.1
  have c123970 := (publicBatchWrapper89_copies3874 a h).2.2.1
  have c123971 := (publicBatchWrapper89_copies3874 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_18
  simp only [← c123966, ← c123967, ← c123968] at e_2032_18
  have e_2032_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_19
  simp only [← c123969, ← c123970, k2, ← c123971] at e_2032_19
  have hr := e_2032_19
  simp only [e_2032_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16591)) := by
  have c123972 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.1
  have c123973 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.1
  have c123974 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.1
  have c123975 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.1
  have c123976 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.1
  have c123977 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_0
  simp only [← c123972, ← c123973, ← c123974] at e_2033_0
  have e_2033_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_1
  simp only [← c123975, ← c123976, k2, ← c123977] at e_2033_1
  have hr := e_2033_1
  simp only [e_2033_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16592)) := by
  have c123978 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123979 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123980 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123981 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123982 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123983 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_2
  simp only [← c123978, ← c123979, ← c123980] at e_2033_2
  have e_2033_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_3
  simp only [← c123981, ← c123982, k2, ← c123983] at e_2033_3
  have hr := e_2033_3
  simp only [e_2033_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16593)) := by
  have c123984 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123985 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123986 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123987 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123988 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123989 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_4
  simp only [← c123984, ← c123985, ← c123986] at e_2033_4
  have e_2033_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_5
  simp only [← c123987, ← c123988, k2, ← c123989] at e_2033_5
  have hr := e_2033_5
  simp only [e_2033_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16594)) := by
  have c123990 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123991 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123992 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123993 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123994 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123995 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_6
  simp only [← c123990, ← c123991, ← c123992] at e_2033_6
  have e_2033_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_7
  simp only [← c123993, ← c123994, k2, ← c123995] at e_2033_7
  have hr := e_2033_7
  simp only [e_2033_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16595)) := by
  have c123996 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123997 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123998 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123999 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124000 := (publicBatchWrapper89_copies3875 a h).1
  have c124001 := (publicBatchWrapper89_copies3875 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_8
  simp only [← c123996, ← c123997, ← c123998] at e_2033_8
  have e_2033_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_9
  simp only [← c123999, ← c124000, k2, ← c124001] at e_2033_9
  have hr := e_2033_9
  simp only [e_2033_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16596)) := by
  have c124002 := (publicBatchWrapper89_copies3875 a h).2.2.1
  have c124003 := (publicBatchWrapper89_copies3875 a h).2.2.2.1
  have c124004 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.1
  have c124005 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.1
  have c124006 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.1
  have c124007 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_10
  simp only [← c124002, ← c124003, ← c124004] at e_2033_10
  have e_2033_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_11
  simp only [← c124005, ← c124006, k2, ← c124007] at e_2033_11
  have hr := e_2033_11
  simp only [e_2033_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16597)) := by
  have c124008 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.1
  have c124009 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.1
  have c124010 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124011 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124012 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124013 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_12
  simp only [← c124008, ← c124009, ← c124010] at e_2033_12
  have e_2033_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_13
  simp only [← c124011, ← c124012, k2, ← c124013] at e_2033_13
  have hr := e_2033_13
  simp only [e_2033_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16598)) := by
  have c124014 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124015 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124016 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124017 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124018 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124019 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_14
  simp only [← c124014, ← c124015, ← c124016] at e_2033_14
  have e_2033_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_15
  simp only [← c124017, ← c124018, k2, ← c124019] at e_2033_15
  have hr := e_2033_15
  simp only [e_2033_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16599)) := by
  have c124020 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124021 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124022 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124023 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124024 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124025 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_16
  simp only [← c124020, ← c124021, ← c124022] at e_2033_16
  have e_2033_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_17
  simp only [← c124023, ← c124024, k2, ← c124025] at e_2033_17
  have hr := e_2033_17
  simp only [e_2033_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16600)) := by
  have c124026 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124027 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124028 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124029 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124030 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124031 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_18
  simp only [← c124026, ← c124027, ← c124028] at e_2033_18
  have e_2033_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_19
  simp only [← c124029, ← c124030, k2, ← c124031] at e_2033_19
  have hr := e_2033_19
  simp only [e_2033_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16601)) := by
  have c124032 := (publicBatchWrapper89_copies3876 a h).1
  have c124033 := (publicBatchWrapper89_copies3876 a h).2.1
  have c124034 := (publicBatchWrapper89_copies3876 a h).2.2.1
  have c124035 := (publicBatchWrapper89_copies3876 a h).2.2.2.1
  have c124036 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.1
  have c124037 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_0
  simp only [← c124032, ← c124033, ← c124034] at e_2034_0
  have e_2034_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_1
  simp only [← c124035, ← c124036, k2, ← c124037] at e_2034_1
  have hr := e_2034_1
  simp only [e_2034_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16770)) := by
  have c124038 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.1
  have c124039 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.1
  have c124040 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.1
  have c124041 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.1
  have c124042 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124043 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_2
  simp only [← c124038, ← c124039, ← c124040] at e_2034_2
  have e_2034_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_3
  simp only [← c124041, ← c124042, k2, ← c124043] at e_2034_3
  have hr := e_2034_3
  simp only [e_2034_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16771)) := by
  have c124044 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124045 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124046 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124047 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124048 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124049 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_4
  simp only [← c124044, ← c124045, ← c124046] at e_2034_4
  have e_2034_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_5
  simp only [← c124047, ← c124048, k2, ← c124049] at e_2034_5
  have hr := e_2034_5
  simp only [e_2034_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16772)) := by
  have c124050 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124051 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124052 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124053 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124054 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124055 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_6
  simp only [← c124050, ← c124051, ← c124052] at e_2034_6
  have e_2034_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_7
  simp only [← c124053, ← c124054, k2, ← c124055] at e_2034_7
  have hr := e_2034_7
  simp only [e_2034_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16773)) := by
  have c124056 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124057 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124058 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124059 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124060 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124061 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_8
  simp only [← c124056, ← c124057, ← c124058] at e_2034_8
  have e_2034_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_9
  simp only [← c124059, ← c124060, k2, ← c124061] at e_2034_9
  have hr := e_2034_9
  simp only [e_2034_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16774)) := by
  have c124062 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124063 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124064 := (publicBatchWrapper89_copies3877 a h).1
  have c124065 := (publicBatchWrapper89_copies3877 a h).2.1
  have c124066 := (publicBatchWrapper89_copies3877 a h).2.2.1
  have c124067 := (publicBatchWrapper89_copies3877 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_10
  simp only [← c124062, ← c124063, ← c124064] at e_2034_10
  have e_2034_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_11
  simp only [← c124065, ← c124066, k2, ← c124067] at e_2034_11
  have hr := e_2034_11
  simp only [e_2034_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16775)) := by
  have c124068 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.1
  have c124069 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.1
  have c124070 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.1
  have c124071 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.1
  have c124072 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.1
  have c124073 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_12
  simp only [← c124068, ← c124069, ← c124070] at e_2034_12
  have e_2034_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_13
  simp only [← c124071, ← c124072, k2, ← c124073] at e_2034_13
  have hr := e_2034_13
  simp only [e_2034_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16776)) := by
  have c124074 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124075 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124076 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124077 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124078 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124079 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_14
  simp only [← c124074, ← c124075, ← c124076] at e_2034_14
  have e_2034_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_15
  simp only [← c124077, ← c124078, k2, ← c124079] at e_2034_15
  have hr := e_2034_15
  simp only [e_2034_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16777)) := by
  have c124080 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124081 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124082 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124083 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124084 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124085 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_16
  simp only [← c124080, ← c124081, ← c124082] at e_2034_16
  have e_2034_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_17
  simp only [← c124083, ← c124084, k2, ← c124085] at e_2034_17
  have hr := e_2034_17
  simp only [e_2034_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16778)) := by
  have c124086 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124087 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124088 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124089 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124090 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124091 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_18
  simp only [← c124086, ← c124087, ← c124088] at e_2034_18
  have e_2034_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_19
  simp only [← c124089, ← c124090, k2, ← c124091] at e_2034_19
  have hr := e_2034_19
  simp only [e_2034_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16779)) := by
  have c124092 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124093 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124094 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124095 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124096 := (publicBatchWrapper89_copies3878 a h).1
  have c124097 := (publicBatchWrapper89_copies3878 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_0
  simp only [← c124092, ← c124093, ← c124094] at e_2035_0
  have e_2035_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_1
  simp only [← c124095, ← c124096, k2, ← c124097] at e_2035_1
  have hr := e_2035_1
  simp only [e_2035_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16780)) := by
  have c124098 := (publicBatchWrapper89_copies3878 a h).2.2.1
  have c124099 := (publicBatchWrapper89_copies3878 a h).2.2.2.1
  have c124100 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.1
  have c124101 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.1
  have c124102 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.1
  have c124103 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_2
  simp only [← c124098, ← c124099, ← c124100] at e_2035_2
  have e_2035_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_3
  simp only [← c124101, ← c124102, k2, ← c124103] at e_2035_3
  have hr := e_2035_3
  simp only [e_2035_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16781)) := by
  have c124104 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.1
  have c124105 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.1
  have c124106 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124107 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124108 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124109 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_4
  simp only [← c124104, ← c124105, ← c124106] at e_2035_4
  have e_2035_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_5
  simp only [← c124107, ← c124108, k2, ← c124109] at e_2035_5
  have hr := e_2035_5
  simp only [e_2035_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16782)) := by
  have c124110 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124111 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124112 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124113 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124114 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124115 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_6
  simp only [← c124110, ← c124111, ← c124112] at e_2035_6
  have e_2035_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_7
  simp only [← c124113, ← c124114, k2, ← c124115] at e_2035_7
  have hr := e_2035_7
  simp only [e_2035_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16783)) := by
  have c124116 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124117 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124118 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124119 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124120 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124121 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_8
  simp only [← c124116, ← c124117, ← c124118] at e_2035_8
  have e_2035_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_9
  simp only [← c124119, ← c124120, k2, ← c124121] at e_2035_9
  have hr := e_2035_9
  simp only [e_2035_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16784)) := by
  have c124122 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124123 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124124 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124125 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124126 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124127 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_10
  simp only [← c124122, ← c124123, ← c124124] at e_2035_10
  have e_2035_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_11
  simp only [← c124125, ← c124126, k2, ← c124127] at e_2035_11
  have hr := e_2035_11
  simp only [e_2035_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16785)) := by
  have c124128 := (publicBatchWrapper89_copies3879 a h).1
  have c124129 := (publicBatchWrapper89_copies3879 a h).2.1
  have c124130 := (publicBatchWrapper89_copies3879 a h).2.2.1
  have c124131 := (publicBatchWrapper89_copies3879 a h).2.2.2.1
  have c124132 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.1
  have c124133 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_12
  simp only [← c124128, ← c124129, ← c124130] at e_2035_12
  have e_2035_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_13
  simp only [← c124131, ← c124132, k2, ← c124133] at e_2035_13
  have hr := e_2035_13
  simp only [e_2035_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16786)) := by
  have c124134 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.1
  have c124135 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.1
  have c124136 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.1
  have c124137 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.1
  have c124138 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124139 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_14
  simp only [← c124134, ← c124135, ← c124136] at e_2035_14
  have e_2035_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_15
  simp only [← c124137, ← c124138, k2, ← c124139] at e_2035_15
  have hr := e_2035_15
  simp only [e_2035_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16787)) := by
  have c124140 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124141 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124142 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124143 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124144 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124145 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_16
  simp only [← c124140, ← c124141, ← c124142] at e_2035_16
  have e_2035_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_17
  simp only [← c124143, ← c124144, k2, ← c124145] at e_2035_17
  have hr := e_2035_17
  simp only [e_2035_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16788)) := by
  have c124146 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124147 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124148 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124149 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124150 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124151 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_18
  simp only [← c124146, ← c124147, ← c124148] at e_2035_18
  have e_2035_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_19
  simp only [← c124149, ← c124150, k2, ← c124151] at e_2035_19
  have hr := e_2035_19
  simp only [e_2035_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16789)) := by
  have c124152 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124153 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124154 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124155 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124156 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124157 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_0
  simp only [← c124152, ← c124153, ← c124154] at e_2036_0
  have e_2036_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_1
  simp only [← c124155, ← c124156, k2, ← c124157] at e_2036_1
  have hr := e_2036_1
  simp only [e_2036_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16790)) := by
  have c124158 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124159 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124160 := (publicBatchWrapper89_copies3880 a h).1
  have c124161 := (publicBatchWrapper89_copies3880 a h).2.1
  have c124162 := (publicBatchWrapper89_copies3880 a h).2.2.1
  have c124163 := (publicBatchWrapper89_copies3880 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_2
  simp only [← c124158, ← c124159, ← c124160] at e_2036_2
  have e_2036_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_3
  simp only [← c124161, ← c124162, k2, ← c124163] at e_2036_3
  have hr := e_2036_3
  simp only [e_2036_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16791)) := by
  have c124164 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.1
  have c124165 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.1
  have c124166 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.1
  have c124167 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.1
  have c124168 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.1
  have c124169 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_4
  simp only [← c124164, ← c124165, ← c124166] at e_2036_4
  have e_2036_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_5
  simp only [← c124167, ← c124168, k2, ← c124169] at e_2036_5
  have hr := e_2036_5
  simp only [e_2036_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16792)) := by
  have c124170 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124171 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124172 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124173 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124174 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124175 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_6
  simp only [← c124170, ← c124171, ← c124172] at e_2036_6
  have e_2036_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_7
  simp only [← c124173, ← c124174, k2, ← c124175] at e_2036_7
  have hr := e_2036_7
  simp only [e_2036_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16793)) := by
  have c124176 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124177 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124178 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124179 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124180 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124181 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_8
  simp only [← c124176, ← c124177, ← c124178] at e_2036_8
  have e_2036_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_9
  simp only [← c124179, ← c124180, k2, ← c124181] at e_2036_9
  have hr := e_2036_9
  simp only [e_2036_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16794)) := by
  have c124182 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124183 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124184 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124185 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124186 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124187 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_10
  simp only [← c124182, ← c124183, ← c124184] at e_2036_10
  have e_2036_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_11
  simp only [← c124185, ← c124186, k2, ← c124187] at e_2036_11
  have hr := e_2036_11
  simp only [e_2036_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16795)) := by
  have c124188 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124189 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124190 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124191 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124192 := (publicBatchWrapper89_copies3881 a h).1
  have c124193 := (publicBatchWrapper89_copies3881 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_12
  simp only [← c124188, ← c124189, ← c124190] at e_2036_12
  have e_2036_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_13
  simp only [← c124191, ← c124192, k2, ← c124193] at e_2036_13
  have hr := e_2036_13
  simp only [e_2036_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16796)) := by
  have c124194 := (publicBatchWrapper89_copies3881 a h).2.2.1
  have c124195 := (publicBatchWrapper89_copies3881 a h).2.2.2.1
  have c124196 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.1
  have c124197 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.1
  have c124198 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.1
  have c124199 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_14
  simp only [← c124194, ← c124195, ← c124196] at e_2036_14
  have e_2036_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_15
  simp only [← c124197, ← c124198, k2, ← c124199] at e_2036_15
  have hr := e_2036_15
  simp only [e_2036_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16797)) := by
  have c124200 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.1
  have c124201 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.1
  have c124202 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124203 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124204 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124205 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_16
  simp only [← c124200, ← c124201, ← c124202] at e_2036_16
  have e_2036_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_17
  simp only [← c124203, ← c124204, k2, ← c124205] at e_2036_17
  have hr := e_2036_17
  simp only [e_2036_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16798)) := by
  have c124206 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124207 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124208 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124209 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124210 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124211 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_18
  simp only [← c124206, ← c124207, ← c124208] at e_2036_18
  have e_2036_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_19
  simp only [← c124209, ← c124210, k2, ← c124211] at e_2036_19
  have hr := e_2036_19
  simp only [e_2036_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16799)) := by
  have c124212 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124213 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124214 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124215 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124216 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124217 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_0
  simp only [← c124212, ← c124213, ← c124214] at e_2037_0
  have e_2037_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_1
  simp only [← c124215, ← c124216, k2, ← c124217] at e_2037_1
  have hr := e_2037_1
  simp only [e_2037_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16800)) := by
  have c124218 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124219 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124220 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124221 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124222 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124223 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_2
  simp only [← c124218, ← c124219, ← c124220] at e_2037_2
  have e_2037_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_3
  simp only [← c124221, ← c124222, k2, ← c124223] at e_2037_3
  have hr := e_2037_3
  simp only [e_2037_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16801)) := by
  have c124224 := (publicBatchWrapper89_copies3882 a h).1
  have c124225 := (publicBatchWrapper89_copies3882 a h).2.1
  have c124226 := (publicBatchWrapper89_copies3882 a h).2.2.1
  have c124227 := (publicBatchWrapper89_copies3882 a h).2.2.2.1
  have c124228 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.1
  have c124229 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_4
  simp only [← c124224, ← c124225, ← c124226] at e_2037_4
  have e_2037_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_5
  simp only [← c124227, ← c124228, k2, ← c124229] at e_2037_5
  have hr := e_2037_5
  simp only [e_2037_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16970)) := by
  have c124230 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.1
  have c124231 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.1
  have c124232 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.1
  have c124233 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.1
  have c124234 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124235 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_6
  simp only [← c124230, ← c124231, ← c124232] at e_2037_6
  have e_2037_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_7
  simp only [← c124233, ← c124234, k2, ← c124235] at e_2037_7
  have hr := e_2037_7
  simp only [e_2037_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16971)) := by
  have c124236 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124237 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124238 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124239 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124240 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124241 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_8
  simp only [← c124236, ← c124237, ← c124238] at e_2037_8
  have e_2037_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_9
  simp only [← c124239, ← c124240, k2, ← c124241] at e_2037_9
  have hr := e_2037_9
  simp only [e_2037_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16972)) := by
  have c124242 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124243 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124244 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124245 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124246 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124247 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_10
  simp only [← c124242, ← c124243, ← c124244] at e_2037_10
  have e_2037_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_11
  simp only [← c124245, ← c124246, k2, ← c124247] at e_2037_11
  have hr := e_2037_11
  simp only [e_2037_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16973)) := by
  have c124248 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124249 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124250 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124251 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124252 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124253 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_12
  simp only [← c124248, ← c124249, ← c124250] at e_2037_12
  have e_2037_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_13
  simp only [← c124251, ← c124252, k2, ← c124253] at e_2037_13
  have hr := e_2037_13
  simp only [e_2037_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16974)) := by
  have c124254 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124255 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124256 := (publicBatchWrapper89_copies3883 a h).1
  have c124257 := (publicBatchWrapper89_copies3883 a h).2.1
  have c124258 := (publicBatchWrapper89_copies3883 a h).2.2.1
  have c124259 := (publicBatchWrapper89_copies3883 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_14
  simp only [← c124254, ← c124255, ← c124256] at e_2037_14
  have e_2037_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_15
  simp only [← c124257, ← c124258, k2, ← c124259] at e_2037_15
  have hr := e_2037_15
  simp only [e_2037_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16975)) := by
  have c124260 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.1
  have c124261 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.1
  have c124262 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.1
  have c124263 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.1
  have c124264 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.1
  have c124265 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_16
  simp only [← c124260, ← c124261, ← c124262] at e_2037_16
  have e_2037_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_17
  simp only [← c124263, ← c124264, k2, ← c124265] at e_2037_17
  have hr := e_2037_17
  simp only [e_2037_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16976)) := by
  have c124266 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124267 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124268 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124269 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124270 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124271 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_18
  simp only [← c124266, ← c124267, ← c124268] at e_2037_18
  have e_2037_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_19
  simp only [← c124269, ← c124270, k2, ← c124271] at e_2037_19
  have hr := e_2037_19
  simp only [e_2037_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16977)) := by
  have c124272 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124273 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124274 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124275 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124276 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124277 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_0
  simp only [← c124272, ← c124273, ← c124274] at e_2038_0
  have e_2038_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_1
  simp only [← c124275, ← c124276, k2, ← c124277] at e_2038_1
  have hr := e_2038_1
  simp only [e_2038_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16978)) := by
  have c124278 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124279 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124280 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124281 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124282 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124283 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_2
  simp only [← c124278, ← c124279, ← c124280] at e_2038_2
  have e_2038_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_3
  simp only [← c124281, ← c124282, k2, ← c124283] at e_2038_3
  have hr := e_2038_3
  simp only [e_2038_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16979)) := by
  have c124284 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124285 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124286 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124287 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124288 := (publicBatchWrapper89_copies3884 a h).1
  have c124289 := (publicBatchWrapper89_copies3884 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_4
  simp only [← c124284, ← c124285, ← c124286] at e_2038_4
  have e_2038_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_5
  simp only [← c124287, ← c124288, k2, ← c124289] at e_2038_5
  have hr := e_2038_5
  simp only [e_2038_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16980)) := by
  have c124290 := (publicBatchWrapper89_copies3884 a h).2.2.1
  have c124291 := (publicBatchWrapper89_copies3884 a h).2.2.2.1
  have c124292 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.1
  have c124293 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.1
  have c124294 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.1
  have c124295 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_6
  simp only [← c124290, ← c124291, ← c124292] at e_2038_6
  have e_2038_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_7
  simp only [← c124293, ← c124294, k2, ← c124295] at e_2038_7
  have hr := e_2038_7
  simp only [e_2038_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16981)) := by
  have c124296 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.1
  have c124297 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.1
  have c124298 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124299 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124300 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124301 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_8
  simp only [← c124296, ← c124297, ← c124298] at e_2038_8
  have e_2038_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_9
  simp only [← c124299, ← c124300, k2, ← c124301] at e_2038_9
  have hr := e_2038_9
  simp only [e_2038_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16982)) := by
  have c124302 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124303 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124304 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124305 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124306 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124307 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_10
  simp only [← c124302, ← c124303, ← c124304] at e_2038_10
  have e_2038_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_11
  simp only [← c124305, ← c124306, k2, ← c124307] at e_2038_11
  have hr := e_2038_11
  simp only [e_2038_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16983)) := by
  have c124308 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124309 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124310 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124311 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124312 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124313 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_12
  simp only [← c124308, ← c124309, ← c124310] at e_2038_12
  have e_2038_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_13
  simp only [← c124311, ← c124312, k2, ← c124313] at e_2038_13
  have hr := e_2038_13
  simp only [e_2038_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16984)) := by
  have c124314 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124315 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124316 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124317 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124318 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124319 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_14
  simp only [← c124314, ← c124315, ← c124316] at e_2038_14
  have e_2038_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_15
  simp only [← c124317, ← c124318, k2, ← c124319] at e_2038_15
  have hr := e_2038_15
  simp only [e_2038_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16985)) := by
  have c124320 := (publicBatchWrapper89_copies3885 a h).1
  have c124321 := (publicBatchWrapper89_copies3885 a h).2.1
  have c124322 := (publicBatchWrapper89_copies3885 a h).2.2.1
  have c124323 := (publicBatchWrapper89_copies3885 a h).2.2.2.1
  have c124324 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.1
  have c124325 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_16
  simp only [← c124320, ← c124321, ← c124322] at e_2038_16
  have e_2038_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_17
  simp only [← c124323, ← c124324, k2, ← c124325] at e_2038_17
  have hr := e_2038_17
  simp only [e_2038_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16986)) := by
  have c124326 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.1
  have c124327 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.1
  have c124328 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.1
  have c124329 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.1
  have c124330 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124331 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_18
  simp only [← c124326, ← c124327, ← c124328] at e_2038_18
  have e_2038_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_19
  simp only [← c124329, ← c124330, k2, ← c124331] at e_2038_19
  have hr := e_2038_19
  simp only [e_2038_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16987)) := by
  have c124332 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124333 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124334 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124335 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124336 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124337 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_0
  simp only [← c124332, ← c124333, ← c124334] at e_2039_0
  have e_2039_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_1
  simp only [← c124335, ← c124336, k2, ← c124337] at e_2039_1
  have hr := e_2039_1
  simp only [e_2039_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16988)) := by
  have c124338 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124339 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124340 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124341 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124342 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124343 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_2
  simp only [← c124338, ← c124339, ← c124340] at e_2039_2
  have e_2039_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_3
  simp only [← c124341, ← c124342, k2, ← c124343] at e_2039_3
  have hr := e_2039_3
  simp only [e_2039_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16989)) := by
  have c124344 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124345 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124346 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124347 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124348 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124349 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_4
  simp only [← c124344, ← c124345, ← c124346] at e_2039_4
  have e_2039_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_5
  simp only [← c124347, ← c124348, k2, ← c124349] at e_2039_5
  have hr := e_2039_5
  simp only [e_2039_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16990)) := by
  have c124350 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124351 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124352 := (publicBatchWrapper89_copies3886 a h).1
  have c124353 := (publicBatchWrapper89_copies3886 a h).2.1
  have c124354 := (publicBatchWrapper89_copies3886 a h).2.2.1
  have c124355 := (publicBatchWrapper89_copies3886 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_6
  simp only [← c124350, ← c124351, ← c124352] at e_2039_6
  have e_2039_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_7
  simp only [← c124353, ← c124354, k2, ← c124355] at e_2039_7
  have hr := e_2039_7
  simp only [e_2039_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16991)) := by
  have c124356 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.1
  have c124357 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.1
  have c124358 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.1
  have c124359 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.1
  have c124360 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.1
  have c124361 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_8
  simp only [← c124356, ← c124357, ← c124358] at e_2039_8
  have e_2039_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_9
  simp only [← c124359, ← c124360, k2, ← c124361] at e_2039_9
  have hr := e_2039_9
  simp only [e_2039_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16992)) := by
  have c124362 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124363 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124364 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124365 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124366 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124367 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_10
  simp only [← c124362, ← c124363, ← c124364] at e_2039_10
  have e_2039_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_11
  simp only [← c124365, ← c124366, k2, ← c124367] at e_2039_11
  have hr := e_2039_11
  simp only [e_2039_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16993)) := by
  have c124368 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124369 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124370 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124371 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124372 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124373 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_12
  simp only [← c124368, ← c124369, ← c124370] at e_2039_12
  have e_2039_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_13
  simp only [← c124371, ← c124372, k2, ← c124373] at e_2039_13
  have hr := e_2039_13
  simp only [e_2039_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16994)) := by
  have c124374 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124375 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124376 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124377 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124378 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124379 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_14
  simp only [← c124374, ← c124375, ← c124376] at e_2039_14
  have e_2039_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_15
  simp only [← c124377, ← c124378, k2, ← c124379] at e_2039_15
  have hr := e_2039_15
  simp only [e_2039_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16995)) := by
  have c124380 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124381 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124382 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124383 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124384 := (publicBatchWrapper89_copies3887 a h).1
  have c124385 := (publicBatchWrapper89_copies3887 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_16
  simp only [← c124380, ← c124381, ← c124382] at e_2039_16
  have e_2039_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_17
  simp only [← c124383, ← c124384, k2, ← c124385] at e_2039_17
  have hr := e_2039_17
  simp only [e_2039_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16996)) := by
  have c124386 := (publicBatchWrapper89_copies3887 a h).2.2.1
  have c124387 := (publicBatchWrapper89_copies3887 a h).2.2.2.1
  have c124388 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.1
  have c124389 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.1
  have c124390 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.1
  have c124391 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_18
  simp only [← c124386, ← c124387, ← c124388] at e_2039_18
  have e_2039_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_19
  simp only [← c124389, ← c124390, k2, ← c124391] at e_2039_19
  have hr := e_2039_19
  simp only [e_2039_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16997)) := by
  have c124392 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.1
  have c124393 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.1
  have c124394 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124395 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124396 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124397 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_0
  simp only [← c124392, ← c124393, ← c124394] at e_2040_0
  have e_2040_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_1
  simp only [← c124395, ← c124396, k2, ← c124397] at e_2040_1
  have hr := e_2040_1
  simp only [e_2040_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16998)) := by
  have c124398 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124399 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124400 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124401 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124402 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124403 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_2
  simp only [← c124398, ← c124399, ← c124400] at e_2040_2
  have e_2040_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_3
  simp only [← c124401, ← c124402, k2, ← c124403] at e_2040_3
  have hr := e_2040_3
  simp only [e_2040_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16999)) := by
  have c124404 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124405 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124406 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124407 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124408 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124409 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_4
  simp only [← c124404, ← c124405, ← c124406] at e_2040_4
  have e_2040_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_5
  simp only [← c124407, ← c124408, k2, ← c124409] at e_2040_5
  have hr := e_2040_5
  simp only [e_2040_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 17000)) := by
  have c124410 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124411 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124412 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124413 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124414 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124415 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_6
  simp only [← c124410, ← c124411, ← c124412] at e_2040_6
  have e_2040_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_7
  simp only [← c124413, ← c124414, k2, ← c124415] at e_2040_7
  have hr := e_2040_7
  simp only [e_2040_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 17001)) := by
  have c124416 := (publicBatchWrapper89_copies3888 a h).1
  have c124417 := (publicBatchWrapper89_copies3888 a h).2.1
  have c124418 := (publicBatchWrapper89_copies3888 a h).2.2.1
  have c124419 := (publicBatchWrapper89_copies3888 a h).2.2.2.1
  have c124420 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.1
  have c124421 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_8
  simp only [← c124416, ← c124417, ← c124418] at e_2040_8
  have e_2040_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_9
  simp only [← c124419, ← c124420, k2, ← c124421] at e_2040_9
  have hr := e_2040_9
  simp only [e_2040_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17170)) := by
  have c124422 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.1
  have c124423 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.1
  have c124424 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.1
  have c124425 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.1
  have c124426 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124427 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_10
  simp only [← c124422, ← c124423, ← c124424] at e_2040_10
  have e_2040_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_11
  simp only [← c124425, ← c124426, k2, ← c124427] at e_2040_11
  have hr := e_2040_11
  simp only [e_2040_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17171)) := by
  have c124428 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124429 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124430 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124431 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124432 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124433 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_12
  simp only [← c124428, ← c124429, ← c124430] at e_2040_12
  have e_2040_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_13
  simp only [← c124431, ← c124432, k2, ← c124433] at e_2040_13
  have hr := e_2040_13
  simp only [e_2040_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17172)) := by
  have c124434 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124435 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124436 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124437 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124438 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124439 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_14
  simp only [← c124434, ← c124435, ← c124436] at e_2040_14
  have e_2040_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_15
  simp only [← c124437, ← c124438, k2, ← c124439] at e_2040_15
  have hr := e_2040_15
  simp only [e_2040_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17173)) := by
  have c124440 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124441 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124442 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124443 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124444 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124445 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_16
  simp only [← c124440, ← c124441, ← c124442] at e_2040_16
  have e_2040_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_17
  simp only [← c124443, ← c124444, k2, ← c124445] at e_2040_17
  have hr := e_2040_17
  simp only [e_2040_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17174)) := by
  have c124446 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124447 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124448 := (publicBatchWrapper89_copies3889 a h).1
  have c124449 := (publicBatchWrapper89_copies3889 a h).2.1
  have c124450 := (publicBatchWrapper89_copies3889 a h).2.2.1
  have c124451 := (publicBatchWrapper89_copies3889 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_18
  simp only [← c124446, ← c124447, ← c124448] at e_2040_18
  have e_2040_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_19
  simp only [← c124449, ← c124450, k2, ← c124451] at e_2040_19
  have hr := e_2040_19
  simp only [e_2040_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17175)) := by
  have c124452 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.1
  have c124453 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.1
  have c124454 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.1
  have c124455 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.1
  have c124456 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.1
  have c124457 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_0
  simp only [← c124452, ← c124453, ← c124454] at e_2041_0
  have e_2041_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_1
  simp only [← c124455, ← c124456, k2, ← c124457] at e_2041_1
  have hr := e_2041_1
  simp only [e_2041_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17176)) := by
  have c124458 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124459 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124460 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124461 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124462 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124463 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_2
  simp only [← c124458, ← c124459, ← c124460] at e_2041_2
  have e_2041_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_3
  simp only [← c124461, ← c124462, k2, ← c124463] at e_2041_3
  have hr := e_2041_3
  simp only [e_2041_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17177)) := by
  have c124464 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124465 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124466 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124467 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124468 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124469 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_4
  simp only [← c124464, ← c124465, ← c124466] at e_2041_4
  have e_2041_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_5
  simp only [← c124467, ← c124468, k2, ← c124469] at e_2041_5
  have hr := e_2041_5
  simp only [e_2041_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17178)) := by
  have c124470 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124471 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124472 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124473 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124474 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124475 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_6
  simp only [← c124470, ← c124471, ← c124472] at e_2041_6
  have e_2041_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_7
  simp only [← c124473, ← c124474, k2, ← c124475] at e_2041_7
  have hr := e_2041_7
  simp only [e_2041_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17179)) := by
  have c124476 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124477 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124478 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124479 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124480 := (publicBatchWrapper89_copies3890 a h).1
  have c124481 := (publicBatchWrapper89_copies3890 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_8
  simp only [← c124476, ← c124477, ← c124478] at e_2041_8
  have e_2041_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_9
  simp only [← c124479, ← c124480, k2, ← c124481] at e_2041_9
  have hr := e_2041_9
  simp only [e_2041_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17180)) := by
  have c124482 := (publicBatchWrapper89_copies3890 a h).2.2.1
  have c124483 := (publicBatchWrapper89_copies3890 a h).2.2.2.1
  have c124484 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.1
  have c124485 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.1
  have c124486 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.1
  have c124487 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_10
  simp only [← c124482, ← c124483, ← c124484] at e_2041_10
  have e_2041_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_11
  simp only [← c124485, ← c124486, k2, ← c124487] at e_2041_11
  have hr := e_2041_11
  simp only [e_2041_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17181)) := by
  have c124488 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.1
  have c124489 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.1
  have c124490 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124491 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124492 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124493 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_12
  simp only [← c124488, ← c124489, ← c124490] at e_2041_12
  have e_2041_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_13
  simp only [← c124491, ← c124492, k2, ← c124493] at e_2041_13
  have hr := e_2041_13
  simp only [e_2041_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17182)) := by
  have c124494 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124495 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124496 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124497 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124498 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124499 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_14
  simp only [← c124494, ← c124495, ← c124496] at e_2041_14
  have e_2041_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_15
  simp only [← c124497, ← c124498, k2, ← c124499] at e_2041_15
  have hr := e_2041_15
  simp only [e_2041_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17183)) := by
  have c124500 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124501 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124502 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124503 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124504 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124505 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_16
  simp only [← c124500, ← c124501, ← c124502] at e_2041_16
  have e_2041_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_17
  simp only [← c124503, ← c124504, k2, ← c124505] at e_2041_17
  have hr := e_2041_17
  simp only [e_2041_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17184)) := by
  have c124506 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124507 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124508 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124509 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124510 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124511 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_18
  simp only [← c124506, ← c124507, ← c124508] at e_2041_18
  have e_2041_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_19
  simp only [← c124509, ← c124510, k2, ← c124511] at e_2041_19
  have hr := e_2041_19
  simp only [e_2041_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17185)) := by
  have c124512 := (publicBatchWrapper89_copies3891 a h).1
  have c124513 := (publicBatchWrapper89_copies3891 a h).2.1
  have c124514 := (publicBatchWrapper89_copies3891 a h).2.2.1
  have c124515 := (publicBatchWrapper89_copies3891 a h).2.2.2.1
  have c124516 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.1
  have c124517 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_0
  simp only [← c124512, ← c124513, ← c124514] at e_2042_0
  have e_2042_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_1
  simp only [← c124515, ← c124516, k2, ← c124517] at e_2042_1
  have hr := e_2042_1
  simp only [e_2042_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17186)) := by
  have c124518 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.1
  have c124519 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.1
  have c124520 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.1
  have c124521 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.1
  have c124522 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124523 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_2
  simp only [← c124518, ← c124519, ← c124520] at e_2042_2
  have e_2042_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_3
  simp only [← c124521, ← c124522, k2, ← c124523] at e_2042_3
  have hr := e_2042_3
  simp only [e_2042_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17187)) := by
  have c124524 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124525 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124526 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124527 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124528 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124529 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_4
  simp only [← c124524, ← c124525, ← c124526] at e_2042_4
  have e_2042_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_5
  simp only [← c124527, ← c124528, k2, ← c124529] at e_2042_5
  have hr := e_2042_5
  simp only [e_2042_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17188)) := by
  have c124530 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124531 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124532 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124533 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124534 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124535 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_6
  simp only [← c124530, ← c124531, ← c124532] at e_2042_6
  have e_2042_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_7
  simp only [← c124533, ← c124534, k2, ← c124535] at e_2042_7
  have hr := e_2042_7
  simp only [e_2042_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17189)) := by
  have c124536 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124537 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124538 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124539 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124540 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124541 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_8
  simp only [← c124536, ← c124537, ← c124538] at e_2042_8
  have e_2042_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_9
  simp only [← c124539, ← c124540, k2, ← c124541] at e_2042_9
  have hr := e_2042_9
  simp only [e_2042_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17190)) := by
  have c124542 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124543 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124544 := (publicBatchWrapper89_copies3892 a h).1
  have c124545 := (publicBatchWrapper89_copies3892 a h).2.1
  have c124546 := (publicBatchWrapper89_copies3892 a h).2.2.1
  have c124547 := (publicBatchWrapper89_copies3892 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_10
  simp only [← c124542, ← c124543, ← c124544] at e_2042_10
  have e_2042_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_11
  simp only [← c124545, ← c124546, k2, ← c124547] at e_2042_11
  have hr := e_2042_11
  simp only [e_2042_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17191)) := by
  have c124548 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.1
  have c124549 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.1
  have c124550 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.1
  have c124551 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.1
  have c124552 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.1
  have c124553 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_12
  simp only [← c124548, ← c124549, ← c124550] at e_2042_12
  have e_2042_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_13
  simp only [← c124551, ← c124552, k2, ← c124553] at e_2042_13
  have hr := e_2042_13
  simp only [e_2042_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17192)) := by
  have c124554 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124555 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124556 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124557 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124558 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124559 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_14
  simp only [← c124554, ← c124555, ← c124556] at e_2042_14
  have e_2042_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_15
  simp only [← c124557, ← c124558, k2, ← c124559] at e_2042_15
  have hr := e_2042_15
  simp only [e_2042_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17193)) := by
  have c124560 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124561 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124562 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124563 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124564 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124565 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_16
  simp only [← c124560, ← c124561, ← c124562] at e_2042_16
  have e_2042_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_17
  simp only [← c124563, ← c124564, k2, ← c124565] at e_2042_17
  have hr := e_2042_17
  simp only [e_2042_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17194)) := by
  have c124566 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124567 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124568 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124569 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124570 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124571 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_18
  simp only [← c124566, ← c124567, ← c124568] at e_2042_18
  have e_2042_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_19
  simp only [← c124569, ← c124570, k2, ← c124571] at e_2042_19
  have hr := e_2042_19
  simp only [e_2042_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17195)) := by
  have c124572 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124573 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124574 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124575 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124576 := (publicBatchWrapper89_copies3893 a h).1
  have c124577 := (publicBatchWrapper89_copies3893 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_0
  simp only [← c124572, ← c124573, ← c124574] at e_2043_0
  have e_2043_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_1
  simp only [← c124575, ← c124576, k2, ← c124577] at e_2043_1
  have hr := e_2043_1
  simp only [e_2043_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17196)) := by
  have c124578 := (publicBatchWrapper89_copies3893 a h).2.2.1
  have c124579 := (publicBatchWrapper89_copies3893 a h).2.2.2.1
  have c124580 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.1
  have c124581 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.1
  have c124582 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.1
  have c124583 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_2
  simp only [← c124578, ← c124579, ← c124580] at e_2043_2
  have e_2043_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_3
  simp only [← c124581, ← c124582, k2, ← c124583] at e_2043_3
  have hr := e_2043_3
  simp only [e_2043_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17197)) := by
  have c124584 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.1
  have c124585 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.1
  have c124586 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124587 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124588 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124589 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_4
  simp only [← c124584, ← c124585, ← c124586] at e_2043_4
  have e_2043_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_5
  simp only [← c124587, ← c124588, k2, ← c124589] at e_2043_5
  have hr := e_2043_5
  simp only [e_2043_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17198)) := by
  have c124590 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124591 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124592 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124593 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124594 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124595 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_6
  simp only [← c124590, ← c124591, ← c124592] at e_2043_6
  have e_2043_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_7
  simp only [← c124593, ← c124594, k2, ← c124595] at e_2043_7
  have hr := e_2043_7
  simp only [e_2043_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17199)) := by
  have c124596 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124597 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124598 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124599 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124600 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124601 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_8
  simp only [← c124596, ← c124597, ← c124598] at e_2043_8
  have e_2043_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_9
  simp only [← c124599, ← c124600, k2, ← c124601] at e_2043_9
  have hr := e_2043_9
  simp only [e_2043_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17200)) := by
  have c124602 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124603 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124604 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124605 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124606 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124607 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_10
  simp only [← c124602, ← c124603, ← c124604] at e_2043_10
  have e_2043_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_11
  simp only [← c124605, ← c124606, k2, ← c124607] at e_2043_11
  have hr := e_2043_11
  simp only [e_2043_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17201)) := by
  have c124608 := (publicBatchWrapper89_copies3894 a h).1
  have c124609 := (publicBatchWrapper89_copies3894 a h).2.1
  have c124610 := (publicBatchWrapper89_copies3894 a h).2.2.1
  have c124611 := (publicBatchWrapper89_copies3894 a h).2.2.2.1
  have c124612 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.1
  have c124613 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_12
  simp only [← c124608, ← c124609, ← c124610] at e_2043_12
  have e_2043_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_13
  simp only [← c124611, ← c124612, k2, ← c124613] at e_2043_13
  have hr := e_2043_13
  simp only [e_2043_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17370)) := by
  have c124614 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.1
  have c124615 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.1
  have c124616 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.1
  have c124617 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.1
  have c124618 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124619 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_14
  simp only [← c124614, ← c124615, ← c124616] at e_2043_14
  have e_2043_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_15
  simp only [← c124617, ← c124618, k2, ← c124619] at e_2043_15
  have hr := e_2043_15
  simp only [e_2043_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17371)) := by
  have c124620 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124621 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124622 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124623 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124624 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124625 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_16
  simp only [← c124620, ← c124621, ← c124622] at e_2043_16
  have e_2043_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_17
  simp only [← c124623, ← c124624, k2, ← c124625] at e_2043_17
  have hr := e_2043_17
  simp only [e_2043_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17372)) := by
  have c124626 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124627 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124628 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124629 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124630 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124631 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_18
  simp only [← c124626, ← c124627, ← c124628] at e_2043_18
  have e_2043_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_19
  simp only [← c124629, ← c124630, k2, ← c124631] at e_2043_19
  have hr := e_2043_19
  simp only [e_2043_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17373)) := by
  have c124632 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124633 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124634 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124635 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124636 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124637 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_0
  simp only [← c124632, ← c124633, ← c124634] at e_2044_0
  have e_2044_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_1
  simp only [← c124635, ← c124636, k2, ← c124637] at e_2044_1
  have hr := e_2044_1
  simp only [e_2044_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17374)) := by
  have c124638 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124639 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124640 := (publicBatchWrapper89_copies3895 a h).1
  have c124641 := (publicBatchWrapper89_copies3895 a h).2.1
  have c124642 := (publicBatchWrapper89_copies3895 a h).2.2.1
  have c124643 := (publicBatchWrapper89_copies3895 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_2
  simp only [← c124638, ← c124639, ← c124640] at e_2044_2
  have e_2044_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_3
  simp only [← c124641, ← c124642, k2, ← c124643] at e_2044_3
  have hr := e_2044_3
  simp only [e_2044_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17375)) := by
  have c124644 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.1
  have c124645 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.1
  have c124646 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.1
  have c124647 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.1
  have c124648 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.1
  have c124649 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_4
  simp only [← c124644, ← c124645, ← c124646] at e_2044_4
  have e_2044_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_5
  simp only [← c124647, ← c124648, k2, ← c124649] at e_2044_5
  have hr := e_2044_5
  simp only [e_2044_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17376)) := by
  have c124650 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124651 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124652 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124653 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124654 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124655 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_6
  simp only [← c124650, ← c124651, ← c124652] at e_2044_6
  have e_2044_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_7
  simp only [← c124653, ← c124654, k2, ← c124655] at e_2044_7
  have hr := e_2044_7
  simp only [e_2044_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17377)) := by
  have c124656 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124657 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124658 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124659 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124660 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124661 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_8
  simp only [← c124656, ← c124657, ← c124658] at e_2044_8
  have e_2044_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_9
  simp only [← c124659, ← c124660, k2, ← c124661] at e_2044_9
  have hr := e_2044_9
  simp only [e_2044_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17378)) := by
  have c124662 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124663 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124664 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124665 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124666 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124667 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_10
  simp only [← c124662, ← c124663, ← c124664] at e_2044_10
  have e_2044_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_11
  simp only [← c124665, ← c124666, k2, ← c124667] at e_2044_11
  have hr := e_2044_11
  simp only [e_2044_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17379)) := by
  have c124668 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124669 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124670 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124671 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124672 := (publicBatchWrapper89_copies3896 a h).1
  have c124673 := (publicBatchWrapper89_copies3896 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_12
  simp only [← c124668, ← c124669, ← c124670] at e_2044_12
  have e_2044_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_13
  simp only [← c124671, ← c124672, k2, ← c124673] at e_2044_13
  have hr := e_2044_13
  simp only [e_2044_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17380)) := by
  have c124674 := (publicBatchWrapper89_copies3896 a h).2.2.1
  have c124675 := (publicBatchWrapper89_copies3896 a h).2.2.2.1
  have c124676 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.1
  have c124677 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.1
  have c124678 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.1
  have c124679 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_14
  simp only [← c124674, ← c124675, ← c124676] at e_2044_14
  have e_2044_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_15
  simp only [← c124677, ← c124678, k2, ← c124679] at e_2044_15
  have hr := e_2044_15
  simp only [e_2044_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17381)) := by
  have c124680 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.1
  have c124681 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.1
  have c124682 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124683 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124684 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124685 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_16
  simp only [← c124680, ← c124681, ← c124682] at e_2044_16
  have e_2044_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_17
  simp only [← c124683, ← c124684, k2, ← c124685] at e_2044_17
  have hr := e_2044_17
  simp only [e_2044_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17382)) := by
  have c124686 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124687 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124688 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124689 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124690 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124691 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_18
  simp only [← c124686, ← c124687, ← c124688] at e_2044_18
  have e_2044_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_19
  simp only [← c124689, ← c124690, k2, ← c124691] at e_2044_19
  have hr := e_2044_19
  simp only [e_2044_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17383)) := by
  have c124692 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124693 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124694 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124695 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124696 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124697 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_0
  simp only [← c124692, ← c124693, ← c124694] at e_2045_0
  have e_2045_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_1
  simp only [← c124695, ← c124696, k2, ← c124697] at e_2045_1
  have hr := e_2045_1
  simp only [e_2045_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17384)) := by
  have c124698 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124699 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124700 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124701 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124702 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124703 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_2
  simp only [← c124698, ← c124699, ← c124700] at e_2045_2
  have e_2045_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_3
  simp only [← c124701, ← c124702, k2, ← c124703] at e_2045_3
  have hr := e_2045_3
  simp only [e_2045_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17385)) := by
  have c124704 := (publicBatchWrapper89_copies3897 a h).1
  have c124705 := (publicBatchWrapper89_copies3897 a h).2.1
  have c124706 := (publicBatchWrapper89_copies3897 a h).2.2.1
  have c124707 := (publicBatchWrapper89_copies3897 a h).2.2.2.1
  have c124708 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.1
  have c124709 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_4
  simp only [← c124704, ← c124705, ← c124706] at e_2045_4
  have e_2045_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_5
  simp only [← c124707, ← c124708, k2, ← c124709] at e_2045_5
  have hr := e_2045_5
  simp only [e_2045_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17386)) := by
  have c124710 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.1
  have c124711 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.1
  have c124712 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.1
  have c124713 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.1
  have c124714 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124715 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_6
  simp only [← c124710, ← c124711, ← c124712] at e_2045_6
  have e_2045_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_7
  simp only [← c124713, ← c124714, k2, ← c124715] at e_2045_7
  have hr := e_2045_7
  simp only [e_2045_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17387)) := by
  have c124716 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124717 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124718 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124719 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124720 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124721 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_8
  simp only [← c124716, ← c124717, ← c124718] at e_2045_8
  have e_2045_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_9
  simp only [← c124719, ← c124720, k2, ← c124721] at e_2045_9
  have hr := e_2045_9
  simp only [e_2045_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17388)) := by
  have c124722 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124723 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124724 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124725 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124726 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124727 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_10
  simp only [← c124722, ← c124723, ← c124724] at e_2045_10
  have e_2045_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_11
  simp only [← c124725, ← c124726, k2, ← c124727] at e_2045_11
  have hr := e_2045_11
  simp only [e_2045_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17389)) := by
  have c124728 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124729 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124730 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124731 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124732 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124733 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_12
  simp only [← c124728, ← c124729, ← c124730] at e_2045_12
  have e_2045_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_13
  simp only [← c124731, ← c124732, k2, ← c124733] at e_2045_13
  have hr := e_2045_13
  simp only [e_2045_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17390)) := by
  have c124734 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124735 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124736 := (publicBatchWrapper89_copies3898 a h).1
  have c124737 := (publicBatchWrapper89_copies3898 a h).2.1
  have c124738 := (publicBatchWrapper89_copies3898 a h).2.2.1
  have c124739 := (publicBatchWrapper89_copies3898 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_14
  simp only [← c124734, ← c124735, ← c124736] at e_2045_14
  have e_2045_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_15
  simp only [← c124737, ← c124738, k2, ← c124739] at e_2045_15
  have hr := e_2045_15
  simp only [e_2045_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17391)) := by
  have c124740 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.1
  have c124741 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.1
  have c124742 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.1
  have c124743 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.1
  have c124744 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.1
  have c124745 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_16
  simp only [← c124740, ← c124741, ← c124742] at e_2045_16
  have e_2045_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_17
  simp only [← c124743, ← c124744, k2, ← c124745] at e_2045_17
  have hr := e_2045_17
  simp only [e_2045_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17392)) := by
  have c124746 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124747 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124748 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124749 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124750 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124751 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_18
  simp only [← c124746, ← c124747, ← c124748] at e_2045_18
  have e_2045_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_19
  simp only [← c124749, ← c124750, k2, ← c124751] at e_2045_19
  have hr := e_2045_19
  simp only [e_2045_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17393)) := by
  have c124752 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124753 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124754 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124755 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124756 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124757 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_0
  simp only [← c124752, ← c124753, ← c124754] at e_2046_0
  have e_2046_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_1
  simp only [← c124755, ← c124756, k2, ← c124757] at e_2046_1
  have hr := e_2046_1
  simp only [e_2046_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17394)) := by
  have c124758 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124759 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124760 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124761 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124762 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124763 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_2
  simp only [← c124758, ← c124759, ← c124760] at e_2046_2
  have e_2046_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_3
  simp only [← c124761, ← c124762, k2, ← c124763] at e_2046_3
  have hr := e_2046_3
  simp only [e_2046_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17395)) := by
  have c124764 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124765 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124766 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124767 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124768 := (publicBatchWrapper89_copies3899 a h).1
  have c124769 := (publicBatchWrapper89_copies3899 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_4
  simp only [← c124764, ← c124765, ← c124766] at e_2046_4
  have e_2046_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_5
  simp only [← c124767, ← c124768, k2, ← c124769] at e_2046_5
  have hr := e_2046_5
  simp only [e_2046_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17396)) := by
  have c124770 := (publicBatchWrapper89_copies3899 a h).2.2.1
  have c124771 := (publicBatchWrapper89_copies3899 a h).2.2.2.1
  have c124772 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.1
  have c124773 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.1
  have c124774 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.1
  have c124775 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_6
  simp only [← c124770, ← c124771, ← c124772] at e_2046_6
  have e_2046_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_7
  simp only [← c124773, ← c124774, k2, ← c124775] at e_2046_7
  have hr := e_2046_7
  simp only [e_2046_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17397)) := by
  have c124776 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.1
  have c124777 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.1
  have c124778 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124779 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124780 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124781 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_8
  simp only [← c124776, ← c124777, ← c124778] at e_2046_8
  have e_2046_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_9
  simp only [← c124779, ← c124780, k2, ← c124781] at e_2046_9
  have hr := e_2046_9
  simp only [e_2046_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17398)) := by
  have c124782 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124783 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124784 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124785 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124786 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124787 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_10
  simp only [← c124782, ← c124783, ← c124784] at e_2046_10
  have e_2046_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_11
  simp only [← c124785, ← c124786, k2, ← c124787] at e_2046_11
  have hr := e_2046_11
  simp only [e_2046_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17399)) := by
  have c124788 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124789 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124790 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124791 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124792 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124793 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_12
  simp only [← c124788, ← c124789, ← c124790] at e_2046_12
  have e_2046_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_13
  simp only [← c124791, ← c124792, k2, ← c124793] at e_2046_13
  have hr := e_2046_13
  simp only [e_2046_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17400)) := by
  have c124794 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124795 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124796 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124797 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124798 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124799 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_14
  simp only [← c124794, ← c124795, ← c124796] at e_2046_14
  have e_2046_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_15
  simp only [← c124797, ← c124798, k2, ← c124799] at e_2046_15
  have hr := e_2046_15
  simp only [e_2046_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17401)) := by
  have c124800 := (publicBatchWrapper89_copies3900 a h).1
  have c124801 := (publicBatchWrapper89_copies3900 a h).2.1
  have c124802 := (publicBatchWrapper89_copies3900 a h).2.2.1
  have c124803 := (publicBatchWrapper89_copies3900 a h).2.2.2.1
  have c124804 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.1
  have c124805 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_16
  simp only [← c124800, ← c124801, ← c124802] at e_2046_16
  have e_2046_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_17
  simp only [← c124803, ← c124804, k2, ← c124805] at e_2046_17
  have hr := e_2046_17
  simp only [e_2046_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17570)) := by
  have c124806 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.1
  have c124807 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.1
  have c124808 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.1
  have c124809 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.1
  have c124810 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124811 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_18
  simp only [← c124806, ← c124807, ← c124808] at e_2046_18
  have e_2046_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_19
  simp only [← c124809, ← c124810, k2, ← c124811] at e_2046_19
  have hr := e_2046_19
  simp only [e_2046_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17571)) := by
  have c124812 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124813 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124814 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124815 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124816 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124817 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_0
  simp only [← c124812, ← c124813, ← c124814] at e_2047_0
  have e_2047_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_1
  simp only [← c124815, ← c124816, k2, ← c124817] at e_2047_1
  have hr := e_2047_1
  simp only [e_2047_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17572)) := by
  have c124818 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124819 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124820 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124821 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124822 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124823 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_2
  simp only [← c124818, ← c124819, ← c124820] at e_2047_2
  have e_2047_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_3
  simp only [← c124821, ← c124822, k2, ← c124823] at e_2047_3
  have hr := e_2047_3
  simp only [e_2047_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17573)) := by
  have c124824 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124825 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124826 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124827 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124828 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124829 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_4
  simp only [← c124824, ← c124825, ← c124826] at e_2047_4
  have e_2047_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_5
  simp only [← c124827, ← c124828, k2, ← c124829] at e_2047_5
  have hr := e_2047_5
  simp only [e_2047_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17574)) := by
  have c124830 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124831 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124832 := (publicBatchWrapper89_copies3901 a h).1
  have c124833 := (publicBatchWrapper89_copies3901 a h).2.1
  have c124834 := (publicBatchWrapper89_copies3901 a h).2.2.1
  have c124835 := (publicBatchWrapper89_copies3901 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_6
  simp only [← c124830, ← c124831, ← c124832] at e_2047_6
  have e_2047_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_7
  simp only [← c124833, ← c124834, k2, ← c124835] at e_2047_7
  have hr := e_2047_7
  simp only [e_2047_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17575)) := by
  have c124836 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.1
  have c124837 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.1
  have c124838 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.1
  have c124839 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.1
  have c124840 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.1
  have c124841 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_8
  simp only [← c124836, ← c124837, ← c124838] at e_2047_8
  have e_2047_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_9
  simp only [← c124839, ← c124840, k2, ← c124841] at e_2047_9
  have hr := e_2047_9
  simp only [e_2047_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17576)) := by
  have c124842 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124843 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124844 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124845 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124846 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124847 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_10
  simp only [← c124842, ← c124843, ← c124844] at e_2047_10
  have e_2047_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_11
  simp only [← c124845, ← c124846, k2, ← c124847] at e_2047_11
  have hr := e_2047_11
  simp only [e_2047_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17577)) := by
  have c124848 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124849 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124850 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124851 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124852 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124853 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_12
  simp only [← c124848, ← c124849, ← c124850] at e_2047_12
  have e_2047_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_13
  simp only [← c124851, ← c124852, k2, ← c124853] at e_2047_13
  have hr := e_2047_13
  simp only [e_2047_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17578)) := by
  have c124854 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124855 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124856 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124857 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124858 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124859 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_14
  simp only [← c124854, ← c124855, ← c124856] at e_2047_14
  have e_2047_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_15
  simp only [← c124857, ← c124858, k2, ← c124859] at e_2047_15
  have hr := e_2047_15
  simp only [e_2047_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17579)) := by
  have c124860 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124861 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124862 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124863 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124864 := (publicBatchWrapper89_copies3902 a h).1
  have c124865 := (publicBatchWrapper89_copies3902 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_16
  simp only [← c124860, ← c124861, ← c124862] at e_2047_16
  have e_2047_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_17
  simp only [← c124863, ← c124864, k2, ← c124865] at e_2047_17
  have hr := e_2047_17
  simp only [e_2047_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17580)) := by
  have c124866 := (publicBatchWrapper89_copies3902 a h).2.2.1
  have c124867 := (publicBatchWrapper89_copies3902 a h).2.2.2.1
  have c124868 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.1
  have c124869 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.1
  have c124870 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.1
  have c124871 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_18
  simp only [← c124866, ← c124867, ← c124868] at e_2047_18
  have e_2047_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_19
  simp only [← c124869, ← c124870, k2, ← c124871] at e_2047_19
  have hr := e_2047_19
  simp only [e_2047_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17581)) := by
  have c124872 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.1
  have c124873 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.1
  have c124874 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124875 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124876 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124877 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_0
  simp only [← c124872, ← c124873, ← c124874] at e_2048_0
  have e_2048_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_1
  simp only [← c124875, ← c124876, k2, ← c124877] at e_2048_1
  have hr := e_2048_1
  simp only [e_2048_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17582)) := by
  have c124878 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124879 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124880 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124881 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124882 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124883 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_2
  simp only [← c124878, ← c124879, ← c124880] at e_2048_2
  have e_2048_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_3
  simp only [← c124881, ← c124882, k2, ← c124883] at e_2048_3
  have hr := e_2048_3
  simp only [e_2048_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17583)) := by
  have c124884 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124885 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124886 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124887 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124888 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124889 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_4
  simp only [← c124884, ← c124885, ← c124886] at e_2048_4
  have e_2048_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_5
  simp only [← c124887, ← c124888, k2, ← c124889] at e_2048_5
  have hr := e_2048_5
  simp only [e_2048_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17584)) := by
  have c124890 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124891 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124892 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124893 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124894 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124895 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_6
  simp only [← c124890, ← c124891, ← c124892] at e_2048_6
  have e_2048_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_7
  simp only [← c124893, ← c124894, k2, ← c124895] at e_2048_7
  have hr := e_2048_7
  simp only [e_2048_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17585)) := by
  have c124896 := (publicBatchWrapper89_copies3903 a h).1
  have c124897 := (publicBatchWrapper89_copies3903 a h).2.1
  have c124898 := (publicBatchWrapper89_copies3903 a h).2.2.1
  have c124899 := (publicBatchWrapper89_copies3903 a h).2.2.2.1
  have c124900 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.1
  have c124901 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_8
  simp only [← c124896, ← c124897, ← c124898] at e_2048_8
  have e_2048_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_9
  simp only [← c124899, ← c124900, k2, ← c124901] at e_2048_9
  have hr := e_2048_9
  simp only [e_2048_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17586)) := by
  have c124902 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.1
  have c124903 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.1
  have c124904 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.1
  have c124905 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.1
  have c124906 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124907 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_10
  simp only [← c124902, ← c124903, ← c124904] at e_2048_10
  have e_2048_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_11
  simp only [← c124905, ← c124906, k2, ← c124907] at e_2048_11
  have hr := e_2048_11
  simp only [e_2048_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17587)) := by
  have c124908 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124909 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124910 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124911 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124912 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124913 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_12
  simp only [← c124908, ← c124909, ← c124910] at e_2048_12
  have e_2048_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_13
  simp only [← c124911, ← c124912, k2, ← c124913] at e_2048_13
  have hr := e_2048_13
  simp only [e_2048_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17588)) := by
  have c124914 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124915 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124916 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124917 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124918 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124919 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_14
  simp only [← c124914, ← c124915, ← c124916] at e_2048_14
  have e_2048_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_15
  simp only [← c124917, ← c124918, k2, ← c124919] at e_2048_15
  have hr := e_2048_15
  simp only [e_2048_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17589)) := by
  have c124920 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124921 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124922 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124923 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124924 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124925 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_16
  simp only [← c124920, ← c124921, ← c124922] at e_2048_16
  have e_2048_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_17
  simp only [← c124923, ← c124924, k2, ← c124925] at e_2048_17
  have hr := e_2048_17
  simp only [e_2048_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17590)) := by
  have c124926 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124927 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124928 := (publicBatchWrapper89_copies3904 a h).1
  have c124929 := (publicBatchWrapper89_copies3904 a h).2.1
  have c124930 := (publicBatchWrapper89_copies3904 a h).2.2.1
  have c124931 := (publicBatchWrapper89_copies3904 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_18
  simp only [← c124926, ← c124927, ← c124928] at e_2048_18
  have e_2048_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_19
  simp only [← c124929, ← c124930, k2, ← c124931] at e_2048_19
  have hr := e_2048_19
  simp only [e_2048_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17591)) := by
  have c124932 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.1
  have c124933 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.1
  have c124934 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.1
  have c124935 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.1
  have c124936 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.1
  have c124937 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_0
  simp only [← c124932, ← c124933, ← c124934] at e_2049_0
  have e_2049_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_1
  simp only [← c124935, ← c124936, k2, ← c124937] at e_2049_1
  have hr := e_2049_1
  simp only [e_2049_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17592)) := by
  have c124938 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124939 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124940 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124941 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124942 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124943 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_2
  simp only [← c124938, ← c124939, ← c124940] at e_2049_2
  have e_2049_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_3
  simp only [← c124941, ← c124942, k2, ← c124943] at e_2049_3
  have hr := e_2049_3
  simp only [e_2049_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17593)) := by
  have c124944 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124945 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124946 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124947 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124948 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124949 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_4
  simp only [← c124944, ← c124945, ← c124946] at e_2049_4
  have e_2049_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_5
  simp only [← c124947, ← c124948, k2, ← c124949] at e_2049_5
  have hr := e_2049_5
  simp only [e_2049_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17594)) := by
  have c124950 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124951 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124952 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124953 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124954 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124955 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_6
  simp only [← c124950, ← c124951, ← c124952] at e_2049_6
  have e_2049_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_7
  simp only [← c124953, ← c124954, k2, ← c124955] at e_2049_7
  have hr := e_2049_7
  simp only [e_2049_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17595)) := by
  have c124956 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124957 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124958 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124959 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124960 := (publicBatchWrapper89_copies3905 a h).1
  have c124961 := (publicBatchWrapper89_copies3905 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_8
  simp only [← c124956, ← c124957, ← c124958] at e_2049_8
  have e_2049_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_9
  simp only [← c124959, ← c124960, k2, ← c124961] at e_2049_9
  have hr := e_2049_9
  simp only [e_2049_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17596)) := by
  have c124962 := (publicBatchWrapper89_copies3905 a h).2.2.1
  have c124963 := (publicBatchWrapper89_copies3905 a h).2.2.2.1
  have c124964 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.1
  have c124965 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.1
  have c124966 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.1
  have c124967 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_10
  simp only [← c124962, ← c124963, ← c124964] at e_2049_10
  have e_2049_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_11
  simp only [← c124965, ← c124966, k2, ← c124967] at e_2049_11
  have hr := e_2049_11
  simp only [e_2049_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17597)) := by
  have c124968 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.1
  have c124969 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.1
  have c124970 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124971 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124972 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124973 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_12
  simp only [← c124968, ← c124969, ← c124970] at e_2049_12
  have e_2049_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_13
  simp only [← c124971, ← c124972, k2, ← c124973] at e_2049_13
  have hr := e_2049_13
  simp only [e_2049_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17598)) := by
  have c124974 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124975 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124976 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124977 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124978 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124979 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_14
  simp only [← c124974, ← c124975, ← c124976] at e_2049_14
  have e_2049_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_15
  simp only [← c124977, ← c124978, k2, ← c124979] at e_2049_15
  have hr := e_2049_15
  simp only [e_2049_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17599)) := by
  have c124980 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124981 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124982 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124983 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124984 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124985 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_16
  simp only [← c124980, ← c124981, ← c124982] at e_2049_16
  have e_2049_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_17
  simp only [← c124983, ← c124984, k2, ← c124985] at e_2049_17
  have hr := e_2049_17
  simp only [e_2049_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17600)) := by
  have c124986 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124987 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124988 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124989 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124990 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124991 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_18
  simp only [← c124986, ← c124987, ← c124988] at e_2049_18
  have e_2049_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_19
  simp only [← c124989, ← c124990, k2, ← c124991] at e_2049_19
  have hr := e_2049_19
  simp only [e_2049_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17601)) := by
  have c124992 := (publicBatchWrapper89_copies3906 a h).1
  have c124993 := (publicBatchWrapper89_copies3906 a h).2.1
  have c124994 := (publicBatchWrapper89_copies3906 a h).2.2.1
  have c124995 := (publicBatchWrapper89_copies3906 a h).2.2.2.1
  have c124996 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.1
  have c124997 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_0
  simp only [← c124992, ← c124993, ← c124994] at e_2050_0
  have e_2050_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_1
  simp only [← c124995, ← c124996, k2, ← c124997] at e_2050_1
  have hr := e_2050_1
  simp only [e_2050_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17770)) := by
  have c124998 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.1
  have c124999 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.1
  have c125000 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.1
  have c125001 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.1
  have c125002 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125003 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_2
  simp only [← c124998, ← c124999, ← c125000] at e_2050_2
  have e_2050_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_3
  simp only [← c125001, ← c125002, k2, ← c125003] at e_2050_3
  have hr := e_2050_3
  simp only [e_2050_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17771)) := by
  have c125004 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125005 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125006 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125007 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125008 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125009 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_4
  simp only [← c125004, ← c125005, ← c125006] at e_2050_4
  have e_2050_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_5
  simp only [← c125007, ← c125008, k2, ← c125009] at e_2050_5
  have hr := e_2050_5
  simp only [e_2050_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17772)) := by
  have c125010 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125011 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125012 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125013 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125014 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125015 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_6
  simp only [← c125010, ← c125011, ← c125012] at e_2050_6
  have e_2050_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_7
  simp only [← c125013, ← c125014, k2, ← c125015] at e_2050_7
  have hr := e_2050_7
  simp only [e_2050_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17773)) := by
  have c125016 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125017 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125018 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125019 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125020 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125021 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_8
  simp only [← c125016, ← c125017, ← c125018] at e_2050_8
  have e_2050_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_9
  simp only [← c125019, ← c125020, k2, ← c125021] at e_2050_9
  have hr := e_2050_9
  simp only [e_2050_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17774)) := by
  have c125022 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125023 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125024 := (publicBatchWrapper89_copies3907 a h).1
  have c125025 := (publicBatchWrapper89_copies3907 a h).2.1
  have c125026 := (publicBatchWrapper89_copies3907 a h).2.2.1
  have c125027 := (publicBatchWrapper89_copies3907 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_10
  simp only [← c125022, ← c125023, ← c125024] at e_2050_10
  have e_2050_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_11
  simp only [← c125025, ← c125026, k2, ← c125027] at e_2050_11
  have hr := e_2050_11
  simp only [e_2050_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17775)) := by
  have c125028 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.1
  have c125029 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.1
  have c125030 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.1
  have c125031 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.1
  have c125032 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.1
  have c125033 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_12
  simp only [← c125028, ← c125029, ← c125030] at e_2050_12
  have e_2050_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_13
  simp only [← c125031, ← c125032, k2, ← c125033] at e_2050_13
  have hr := e_2050_13
  simp only [e_2050_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17776)) := by
  have c125034 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125035 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125036 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125037 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125038 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125039 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_14
  simp only [← c125034, ← c125035, ← c125036] at e_2050_14
  have e_2050_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_15
  simp only [← c125037, ← c125038, k2, ← c125039] at e_2050_15
  have hr := e_2050_15
  simp only [e_2050_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17777)) := by
  have c125040 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125041 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125042 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125043 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125044 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125045 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_16
  simp only [← c125040, ← c125041, ← c125042] at e_2050_16
  have e_2050_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_17
  simp only [← c125043, ← c125044, k2, ← c125045] at e_2050_17
  have hr := e_2050_17
  simp only [e_2050_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17778)) := by
  have c125046 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125047 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125048 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125049 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125050 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125051 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_18
  simp only [← c125046, ← c125047, ← c125048] at e_2050_18
  have e_2050_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_19
  simp only [← c125049, ← c125050, k2, ← c125051] at e_2050_19
  have hr := e_2050_19
  simp only [e_2050_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17779)) := by
  have c125052 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125053 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125054 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125055 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125056 := (publicBatchWrapper89_copies3908 a h).1
  have c125057 := (publicBatchWrapper89_copies3908 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_0
  simp only [← c125052, ← c125053, ← c125054] at e_2051_0
  have e_2051_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_1
  simp only [← c125055, ← c125056, k2, ← c125057] at e_2051_1
  have hr := e_2051_1
  simp only [e_2051_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17780)) := by
  have c125058 := (publicBatchWrapper89_copies3908 a h).2.2.1
  have c125059 := (publicBatchWrapper89_copies3908 a h).2.2.2.1
  have c125060 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.1
  have c125061 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.1
  have c125062 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.1
  have c125063 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_2
  simp only [← c125058, ← c125059, ← c125060] at e_2051_2
  have e_2051_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_3
  simp only [← c125061, ← c125062, k2, ← c125063] at e_2051_3
  have hr := e_2051_3
  simp only [e_2051_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17781)) := by
  have c125064 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.1
  have c125065 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.1
  have c125066 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125067 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125068 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125069 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_4
  simp only [← c125064, ← c125065, ← c125066] at e_2051_4
  have e_2051_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_5
  simp only [← c125067, ← c125068, k2, ← c125069] at e_2051_5
  have hr := e_2051_5
  simp only [e_2051_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17782)) := by
  have c125070 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125071 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125072 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125073 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125074 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125075 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_6
  simp only [← c125070, ← c125071, ← c125072] at e_2051_6
  have e_2051_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_7
  simp only [← c125073, ← c125074, k2, ← c125075] at e_2051_7
  have hr := e_2051_7
  simp only [e_2051_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17783)) := by
  have c125076 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125077 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125078 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125079 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125080 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125081 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_8
  simp only [← c125076, ← c125077, ← c125078] at e_2051_8
  have e_2051_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_9
  simp only [← c125079, ← c125080, k2, ← c125081] at e_2051_9
  have hr := e_2051_9
  simp only [e_2051_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17784)) := by
  have c125082 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125083 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125084 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125085 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125086 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125087 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_10
  simp only [← c125082, ← c125083, ← c125084] at e_2051_10
  have e_2051_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_11
  simp only [← c125085, ← c125086, k2, ← c125087] at e_2051_11
  have hr := e_2051_11
  simp only [e_2051_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17785)) := by
  have c125088 := (publicBatchWrapper89_copies3909 a h).1
  have c125089 := (publicBatchWrapper89_copies3909 a h).2.1
  have c125090 := (publicBatchWrapper89_copies3909 a h).2.2.1
  have c125091 := (publicBatchWrapper89_copies3909 a h).2.2.2.1
  have c125092 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.1
  have c125093 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_12
  simp only [← c125088, ← c125089, ← c125090] at e_2051_12
  have e_2051_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_13
  simp only [← c125091, ← c125092, k2, ← c125093] at e_2051_13
  have hr := e_2051_13
  simp only [e_2051_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17786)) := by
  have c125094 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.1
  have c125095 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.1
  have c125096 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.1
  have c125097 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.1
  have c125098 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125099 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_14
  simp only [← c125094, ← c125095, ← c125096] at e_2051_14
  have e_2051_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_15
  simp only [← c125097, ← c125098, k2, ← c125099] at e_2051_15
  have hr := e_2051_15
  simp only [e_2051_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17787)) := by
  have c125100 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125101 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125102 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125103 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125104 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125105 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_16
  simp only [← c125100, ← c125101, ← c125102] at e_2051_16
  have e_2051_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_17
  simp only [← c125103, ← c125104, k2, ← c125105] at e_2051_17
  have hr := e_2051_17
  simp only [e_2051_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17788)) := by
  have c125106 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125107 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125108 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125109 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125110 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125111 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_18
  simp only [← c125106, ← c125107, ← c125108] at e_2051_18
  have e_2051_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_19
  simp only [← c125109, ← c125110, k2, ← c125111] at e_2051_19
  have hr := e_2051_19
  simp only [e_2051_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17789)) := by
  have c125112 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125113 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125114 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125115 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125116 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125117 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_0
  simp only [← c125112, ← c125113, ← c125114] at e_2052_0
  have e_2052_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_1
  simp only [← c125115, ← c125116, k2, ← c125117] at e_2052_1
  have hr := e_2052_1
  simp only [e_2052_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17790)) := by
  have c125118 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125119 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125120 := (publicBatchWrapper89_copies3910 a h).1
  have c125121 := (publicBatchWrapper89_copies3910 a h).2.1
  have c125122 := (publicBatchWrapper89_copies3910 a h).2.2.1
  have c125123 := (publicBatchWrapper89_copies3910 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_2
  simp only [← c125118, ← c125119, ← c125120] at e_2052_2
  have e_2052_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_3
  simp only [← c125121, ← c125122, k2, ← c125123] at e_2052_3
  have hr := e_2052_3
  simp only [e_2052_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17791)) := by
  have c125124 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.1
  have c125125 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.1
  have c125126 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.1
  have c125127 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.1
  have c125128 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.1
  have c125129 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_4
  simp only [← c125124, ← c125125, ← c125126] at e_2052_4
  have e_2052_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_5
  simp only [← c125127, ← c125128, k2, ← c125129] at e_2052_5
  have hr := e_2052_5
  simp only [e_2052_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17792)) := by
  have c125130 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125131 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125132 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125133 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125134 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125135 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_6
  simp only [← c125130, ← c125131, ← c125132] at e_2052_6
  have e_2052_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_7
  simp only [← c125133, ← c125134, k2, ← c125135] at e_2052_7
  have hr := e_2052_7
  simp only [e_2052_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17793)) := by
  have c125136 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125137 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125138 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125139 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125140 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125141 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_8
  simp only [← c125136, ← c125137, ← c125138] at e_2052_8
  have e_2052_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_9
  simp only [← c125139, ← c125140, k2, ← c125141] at e_2052_9
  have hr := e_2052_9
  simp only [e_2052_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17794)) := by
  have c125142 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125143 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125144 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125145 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125146 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125147 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_10
  simp only [← c125142, ← c125143, ← c125144] at e_2052_10
  have e_2052_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_11
  simp only [← c125145, ← c125146, k2, ← c125147] at e_2052_11
  have hr := e_2052_11
  simp only [e_2052_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17795)) := by
  have c125148 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125149 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125150 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125151 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125152 := (publicBatchWrapper89_copies3911 a h).1
  have c125153 := (publicBatchWrapper89_copies3911 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_12
  simp only [← c125148, ← c125149, ← c125150] at e_2052_12
  have e_2052_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_13
  simp only [← c125151, ← c125152, k2, ← c125153] at e_2052_13
  have hr := e_2052_13
  simp only [e_2052_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17796)) := by
  have c125154 := (publicBatchWrapper89_copies3911 a h).2.2.1
  have c125155 := (publicBatchWrapper89_copies3911 a h).2.2.2.1
  have c125156 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.1
  have c125157 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.1
  have c125158 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.1
  have c125159 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_14
  simp only [← c125154, ← c125155, ← c125156] at e_2052_14
  have e_2052_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_15
  simp only [← c125157, ← c125158, k2, ← c125159] at e_2052_15
  have hr := e_2052_15
  simp only [e_2052_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17797)) := by
  have c125160 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.1
  have c125161 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.1
  have c125162 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125163 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125164 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125165 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_16
  simp only [← c125160, ← c125161, ← c125162] at e_2052_16
  have e_2052_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_17
  simp only [← c125163, ← c125164, k2, ← c125165] at e_2052_17
  have hr := e_2052_17
  simp only [e_2052_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17798)) := by
  have c125166 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125167 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125168 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125169 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125170 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125171 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_18
  simp only [← c125166, ← c125167, ← c125168] at e_2052_18
  have e_2052_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_19
  simp only [← c125169, ← c125170, k2, ← c125171] at e_2052_19
  have hr := e_2052_19
  simp only [e_2052_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17799)) := by
  have c125172 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125173 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125174 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125175 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125176 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125177 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_0
  simp only [← c125172, ← c125173, ← c125174] at e_2053_0
  have e_2053_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_1
  simp only [← c125175, ← c125176, k2, ← c125177] at e_2053_1
  have hr := e_2053_1
  simp only [e_2053_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17800)) := by
  have c125178 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125179 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125180 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125181 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125182 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125183 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_2
  simp only [← c125178, ← c125179, ← c125180] at e_2053_2
  have e_2053_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_3
  simp only [← c125181, ← c125182, k2, ← c125183] at e_2053_3
  have hr := e_2053_3
  simp only [e_2053_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2053 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17801)) := by
  have c125184 := (publicBatchWrapper89_copies3912 a h).1
  have c125185 := (publicBatchWrapper89_copies3912 a h).2.1
  have c125186 := (publicBatchWrapper89_copies3912 a h).2.2.1
  have c125187 := (publicBatchWrapper89_copies3912 a h).2.2.2.1
  have c125188 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.1
  have c125189 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2053_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_4
  simp only [← c125184, ← c125185, ← c125186] at e_2053_4
  have e_2053_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2053_5
  simp only [← c125187, ← c125188, k2, ← c125189] at e_2053_5
  have hr := e_2053_5
  simp only [e_2053_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
